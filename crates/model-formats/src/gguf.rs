use adaforge_application::{DiagnosticError, MetadataValue as V, ModelInfo, ModelInspector};
use adaforge_domain::{TensorInfo, WeightType};
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Default)]
pub struct GgufInspector {
    index: std::cell::RefCell<Option<CachedIndex>>,
}
struct CachedIndex {
    path: std::path::PathBuf,
    size: u64,
    modified: std::time::SystemTime,
    info: ModelInfo,
}
impl GgufInspector {
    fn index(&self, path: &Path, file: &File) -> io::Result<ModelInfo> {
        let metadata = file.metadata()?;
        let size = metadata.len();
        let modified = metadata.modified()?;
        let path = path.canonicalize()?;
        if let Some(cached) = self.index.borrow().as_ref()
            && cached.path == path
            && cached.size == size
            && cached.modified == modified
        {
            return Ok(cached.info.clone());
        }
        let info = parse(BufReader::new(file.try_clone()?), size)?;
        *self.index.borrow_mut() = Some(CachedIndex {
            path,
            size,
            modified,
            info: info.clone(),
        });
        Ok(info)
    }
}
impl ModelInspector for GgufInspector {
    fn read_q6_matrix(
        &self,
        path: &Path,
        name: &str,
    ) -> Result<adaforge_domain::Q6Matrix, DiagnosticError> {
        let read = || -> io::Result<adaforge_domain::Q6Matrix> {
            let mut file = File::open(path)?;
            let info = self.index(path, &file)?;
            let t = info
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| invalid("Matrice Q6_K absente"))?;
            if t.weight_type != WeightType::Q6K || t.dimensions.len() != 2 {
                return Err(invalid("Matrice Q6_K de rang 2 requise"));
            }
            let columns =
                u32::try_from(t.dimensions[0]).map_err(|_| invalid("Largeur Q6_K excessive"))?;
            let rows =
                u32::try_from(t.dimensions[1]).map_err(|_| invalid("Lignes Q6_K excessives"))?;
            if columns > 65536 || rows > 1000000 || t.byte_length > 4 * 1024 * 1024 * 1024 {
                return Err(invalid("Matrice Q6_K trop grande"));
            }
            file.seek(SeekFrom::Start(t.file_offset))?;
            let mut bytes = vec![0; t.byte_length as usize];
            file.read_exact(&mut bytes)?;
            adaforge_domain::Q6Matrix::new(columns, rows, bytes).map_err(|e| invalid(&e))
        };
        read().map_err(|e| DiagnosticError(format!("Lecture tête Q6_K : {e}")))
    }

    fn read_tokenizer(
        &self,
        path: &Path,
    ) -> Result<adaforge_domain::GemmaTokenizer, DiagnosticError> {
        let read = || -> io::Result<adaforge_domain::GemmaTokenizer> {
            let file = File::open(path)?;
            let size = file.metadata()?.len();
            let mut r = Reader {
                inner: BufReader::new(file),
                size,
                budget: 256 * 1024 * 1024,
                retain_arrays: true,
            };
            if r.bytes::<4>()? != *b"GGUF" || r.u32()? != 3 {
                return Err(invalid("Tokenizer : GGUF v3 little-endian requis"));
            }
            let _tensors = r.u64()?;
            let count = r.u64()?;
            if count > 100000 {
                return Err(invalid("Trop de métadonnées"));
            }
            let mut entries = BTreeMap::new();
            for _ in 0..count {
                let key = r.string(true)?;
                let kind = r.u32()?;
                let keep = matches!(
                    key.as_str(),
                    "tokenizer.ggml.model"
                        | "tokenizer.ggml.tokens"
                        | "tokenizer.ggml.token_type"
                        | "tokenizer.ggml.merges"
                        | "tokenizer.ggml.bos_token_id"
                        | "tokenizer.ggml.add_space_prefix"
                        | "tokenizer.ggml.add_bos_token"
                );
                let value = r.value(kind, keep)?;
                if keep && entries.insert(key, value).is_some() {
                    return Err(invalid("Métadonnée tokenizer dupliquée"));
                }
            }
            if !matches!(entries.get("tokenizer.ggml.model"),Some(V::Text(m)) if m=="gemma4") {
                return Err(invalid("Tokenizer Gemma 4 requis"));
            }
            if !matches!(
                entries.get("tokenizer.ggml.add_space_prefix"),
                Some(V::Bool(false))
            ) {
                return Err(invalid("Préfixe espace non pris en charge"));
            }
            let strings = |key: &str, map: &mut BTreeMap<String, V>| -> io::Result<Vec<String>> {
                match map.remove(key) {
                    Some(V::Array(values)) => values
                        .into_iter()
                        .map(|v| match v {
                            V::Text(t) => Ok(t),
                            _ => Err(invalid("Table de chaînes invalide")),
                        })
                        .collect(),
                    _ => Err(invalid("Table de chaînes absente")),
                }
            };
            let tokens = strings("tokenizer.ggml.tokens", &mut entries)?;
            let merges = strings("tokenizer.ggml.merges", &mut entries)?;
            let types = match entries.remove("tokenizer.ggml.token_type") {
                Some(V::Array(values)) => values
                    .into_iter()
                    .map(|v| match v {
                        V::Signed(n) => {
                            i32::try_from(n).map_err(|_| invalid("Type token invalide"))
                        }
                        _ => Err(invalid("Types tokenizer invalides")),
                    })
                    .collect::<io::Result<Vec<_>>>()?,
                _ => return Err(invalid("Types tokenizer absents")),
            };
            let bos = match entries.get("tokenizer.ggml.bos_token_id") {
                Some(V::Unsigned(n)) => u32::try_from(*n).map_err(|_| invalid("BOS invalide"))?,
                _ => return Err(invalid("BOS absent")),
            };
            adaforge_domain::GemmaTokenizer::new(tokens, types, merges, bos)
                .map_err(|e| invalid(&e))
        };
        read().map_err(|e| DiagnosticError(format!("Lecture tokenizer : {e}")))
    }

    fn read_embeddings(
        &self,
        path: &Path,
        ids: &[u32],
    ) -> Result<adaforge_domain::EmbeddingBatch, DiagnosticError> {
        let read = || -> io::Result<adaforge_domain::EmbeddingBatch> {
            let mut file = File::open(path)?;
            let info = self.index(path, &file)?;
            if !matches!(info.metadata.get("general.architecture"),Some(V::Text(a)) if a=="gemma4")
            {
                return Err(invalid("Embeddings : modèle Gemma 4 requis"));
            }
            let t = info
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .ok_or_else(|| invalid("Table token_embd.weight absente"))?;
            if t.weight_type != WeightType::Q6K || t.dimensions.len() != 2 {
                return Err(invalid("Embeddings : table Q6_K de rang 2 requise"));
            }
            let width = u32::try_from(t.dimensions[0])
                .map_err(|_| invalid("Largeur embeddings excessive"))?;
            let vocabulary =
                u32::try_from(t.dimensions[1]).map_err(|_| invalid("Vocabulaire excessif"))?;
            if width > 65536 || vocabulary == 0 {
                return Err(invalid("Dimensions embeddings invalides"));
            }
            let ids = if ids.is_empty() {
                vec![0, 1, 2, 255, 256, 1023, vocabulary - 1, 1]
            } else {
                ids.to_vec()
            };
            if ids.len() > 256 || ids.iter().any(|&i| i >= vocabulary) {
                return Err(invalid(
                    "ID token hors vocabulaire ou plus de 256 IDs demandés",
                ));
            }
            let row_bytes = t.byte_length / u64::from(vocabulary);
            let mut bytes = vec![0; row_bytes as usize * ids.len()];
            for (row, &token) in ids.iter().enumerate() {
                let offset = t
                    .file_offset
                    .checked_add(u64::from(token) * row_bytes)
                    .ok_or_else(|| invalid("Offset embeddings trop grand"))?;
                file.seek(SeekFrom::Start(offset))?;
                file.read_exact(
                    &mut bytes[row * row_bytes as usize..(row + 1) * row_bytes as usize],
                )?;
            }
            adaforge_domain::EmbeddingBatch::new(width, vocabulary, ids, bytes)
                .map_err(|e| invalid(&e))
        };
        read().map_err(|e| DiagnosticError(format!("Lecture embeddings : {e}")))
    }

    fn read_f32_vector(&self, path: &Path, name: &str) -> Result<Vec<f32>, DiagnosticError> {
        let read = || -> io::Result<Vec<f32>> {
            let mut file = File::open(path)?;
            let info = self.index(path, &file)?;
            let t = info
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| invalid("Vecteur F32 absent"))?;
            if t.weight_type != WeightType::F32
                || t.dimensions.len() != 1
                || t.dimensions[0] > 65536
            {
                return Err(invalid("Vecteur F32 requis, 65536 éléments maximum"));
            }
            file.seek(SeekFrom::Start(t.file_offset))?;
            let mut bytes = vec![0; t.byte_length as usize];
            file.read_exact(&mut bytes)?;
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("Valeur F32 non finie"));
            }
            Ok(values)
        };
        read().map_err(|e| DiagnosticError(format!("Lecture {name}: {e}")))
    }

    fn read_rms_norm(
        &self,
        path: &Path,
        name: &str,
    ) -> Result<adaforge_domain::RmsNorm, DiagnosticError> {
        let read = || -> io::Result<adaforge_domain::RmsNorm> {
            let mut file = File::open(path)?;
            let info = self.index(path, &file)?;
            if !matches!(info.metadata.get("general.architecture"),Some(V::Text(a)) if a=="gemma4")
            {
                return Err(invalid("Le diagnostic RMSNorm modèle cible Gemma 4"));
            }
            let epsilon = match info.metadata.get("gemma4.attention.layer_norm_rms_epsilon") {
                Some(V::Float(v)) => *v as f32,
                _ => return Err(invalid("Epsilon RMSNorm absent ou invalide")),
            };
            let t = info
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| invalid("Tenseur RMSNorm absent"))?;
            if t.weight_type != WeightType::F32
                || t.dimensions.len() != 1
                || t.dimensions[0] > 65536
            {
                return Err(invalid(
                    "RMSNorm attend un vecteur F32 de 1 à 65536 éléments",
                ));
            }
            file.seek(SeekFrom::Start(t.file_offset))?;
            let mut bytes = vec![0; t.byte_length as usize];
            file.read_exact(&mut bytes)?;
            let weights = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            adaforge_domain::RmsNorm::new(weights, epsilon).map_err(|e| invalid(&e))
        };
        read().map_err(|e| DiagnosticError(format!("RMSNorm {name} : {e}")))
    }

    fn read_q4_matrix(
        &self,
        path: &Path,
        name: &str,
    ) -> Result<adaforge_domain::Q4Matrix, DiagnosticError> {
        let read = || -> io::Result<adaforge_domain::Q4Matrix> {
            let mut file = File::open(path)?;
            let info = self.index(path, &file)?;
            let tensor = info
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| invalid("Tenseur absent"))?;
            if tensor.weight_type != WeightType::Q4_0 || tensor.dimensions.len() != 2 {
                return Err(invalid("Le diagnostic attend une matrice Q4_0 de rang 2"));
            }
            // Limite volontaire du diagnostic, avant toute allocation.
            if tensor.byte_length > 256 * 1024 * 1024 {
                return Err(invalid(
                    "Matrice dépassant la limite de diagnostic de 256 Mio",
                ));
            }
            let columns =
                u32::try_from(tensor.dimensions[0]).map_err(|_| invalid("Trop de colonnes"))?;
            let rows =
                u32::try_from(tensor.dimensions[1]).map_err(|_| invalid("Trop de lignes"))?;
            file.seek(SeekFrom::Start(tensor.file_offset))?;
            let mut bytes = vec![0; tensor.byte_length as usize];
            file.read_exact(&mut bytes)?;
            adaforge_domain::Q4Matrix::new(columns, rows, bytes).map_err(|e| invalid(&e))
        };
        read().map_err(|e| DiagnosticError(format!("Lecture {name} dans {}: {e}", path.display())))
    }

    fn inspect(&self, path: &Path) -> Result<ModelInfo, DiagnosticError> {
        let file =
            File::open(path).map_err(|e| DiagnosticError(format!("{}: {e}", path.display())))?;
        self.index(path, &file)
            .map_err(|e| DiagnosticError(format!("GGUF {}: {e}", path.display())))
    }
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
struct Reader<R> {
    inner: R,
    size: u64,
    budget: u64,
    retain_arrays: bool,
}
impl<R: Read + Seek> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        if self.budget < N as u64 {
            return Err(invalid("En-tête trop volumineux (limite 256 Mio)"));
        }
        self.budget -= N as u64;
        let mut b = [0; N];
        self.inner.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self, keep: bool) -> io::Result<String> {
        let n = self.u64()?;
        let pos = self.inner.stream_position()?;
        if n > 16 * 1024 * 1024 || n > self.budget || n > self.size.saturating_sub(pos) {
            return Err(invalid("Longueur de chaîne invalide ou excessive"));
        }
        self.budget -= n;
        if !keep {
            self.inner.seek(SeekFrom::Current(n as i64))?;
            return Ok(String::new());
        }
        let mut bytes = vec![0; n as usize];
        self.inner.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| invalid("Chaîne UTF-8 invalide"))
    }
    fn value(&mut self, kind: u32, keep: bool) -> io::Result<V> {
        Ok(match kind {
            0 => V::Unsigned(self.bytes::<1>()?[0] as u64),
            1 => V::Signed(i8::from_le_bytes(self.bytes()?) as i64),
            2 => V::Unsigned(u16::from_le_bytes(self.bytes()?) as u64),
            3 => V::Signed(i16::from_le_bytes(self.bytes()?) as i64),
            4 => V::Unsigned(self.u32()? as u64),
            5 => V::Signed(i32::from_le_bytes(self.bytes()?) as i64),
            6 => V::Float(f32::from_le_bytes(self.bytes()?) as f64),
            7 => match self.bytes::<1>()?[0] {
                0 => V::Bool(false),
                1 => V::Bool(true),
                _ => return Err(invalid("Booléen invalide")),
            },
            8 => V::Text(self.string(keep)?),
            9 => {
                let t = self.u32()?;
                let n = self.u64()?;
                if t > 12 || t == 9 || n > 4_000_000 {
                    return Err(invalid("Type ou taille de tableau non pris en charge"));
                }
                let collect = keep && (n <= 4096 || self.retain_arrays);
                let mut values = Vec::new();
                for _ in 0..n {
                    let v = self.value(t, collect)?;
                    if collect {
                        values.push(v);
                    }
                }
                if collect {
                    V::Array(values)
                } else {
                    V::ArraySummary {
                        element_type: t,
                        length: n,
                    }
                }
            }
            10 => V::Unsigned(self.u64()?),
            11 => V::Signed(i64::from_le_bytes(self.bytes()?)),
            12 => V::Float(f64::from_le_bytes(self.bytes()?)),
            _ => return Err(invalid("Type de métadonnée inconnu")),
        })
    }
}
fn parse<R: Read + Seek>(inner: R, size: u64) -> io::Result<ModelInfo> {
    let mut r = Reader {
        inner,
        size,
        budget: 256 * 1024 * 1024,
        retain_arrays: false,
    };
    if r.bytes::<4>()? != *b"GGUF" {
        return Err(invalid("Signature GGUF absente"));
    }
    let version = r.u32()?;
    if version != 3 {
        return Err(invalid("Seul GGUF v3 little-endian est pris en charge"));
    }
    let nt = r.u64()?;
    let nk = r.u64()?;
    if nt > 100_000 || nk > 100_000 {
        return Err(invalid("Nombre d’entrées excessif"));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..nk {
        let key = r.string(true)?;
        let kind = r.u32()?;
        let value = r.value(kind, true)?;
        if metadata.insert(key, value).is_some() {
            return Err(invalid("Clé de métadonnée dupliquée"));
        }
    }
    let alignment = match metadata.get("general.alignment") {
        None => 32,
        Some(V::Unsigned(n)) if *n > 0 && n.is_power_of_two() && *n <= 1024 * 1024 => *n,
        _ => return Err(invalid("Alignement invalide")),
    };
    let mut tensors = Vec::new();
    let mut names = HashSet::new();
    for _ in 0..nt {
        let name = r.string(true)?;
        if name.is_empty() || !names.insert(name.clone()) {
            return Err(invalid("Nom de tenseur vide ou dupliqué"));
        }
        let nd = r.u32()?;
        if !(1..=4).contains(&nd) {
            return Err(invalid("Nombre de dimensions invalide"));
        }
        let mut dimensions = Vec::new();
        let mut elements = 1_u64;
        for _ in 0..nd {
            let d = r.u64()?;
            if d == 0 {
                return Err(invalid("Dimension nulle"));
            }
            elements = elements
                .checked_mul(d)
                .ok_or_else(|| invalid("Dimensions trop grandes"))?;
            dimensions.push(d);
        }
        let type_id = r.u32()?;
        let weight_type = match type_id {
            0 => WeightType::F32,
            1 => WeightType::F16,
            2 => WeightType::Q4_0,
            8 => WeightType::Q8_0,
            14 => WeightType::Q6K,
            30 => WeightType::Bf16,
            _ => {
                return Err(invalid(&format!(
                    "Type de poids non pris en charge : {type_id} ({name})"
                )));
            }
        };
        let (block, bytes) = weight_type.block_layout();
        if dimensions[0] % block != 0 {
            return Err(invalid(
                "Ligne incompatible avec les blocs de quantification",
            ));
        }
        let byte_length = (elements / block)
            .checked_mul(bytes)
            .ok_or_else(|| invalid("Tenseur trop grand"))?;
        let offset = r.u64()?;
        if offset % alignment != 0 {
            return Err(invalid("Offset de tenseur non aligné"));
        }
        tensors.push(TensorInfo {
            name,
            dimensions,
            weight_type,
            file_offset: offset,
            byte_length,
        });
    }
    let pos = r.inner.stream_position()?;
    let data_offset = pos
        .checked_add(alignment - 1)
        .ok_or_else(|| invalid("Offset trop grand"))?
        / alignment
        * alignment;
    if data_offset > size {
        return Err(invalid("Section de données absente"));
    }
    let mut ranges = Vec::new();
    for t in &mut tensors {
        t.file_offset = data_offset
            .checked_add(t.file_offset)
            .ok_or_else(|| invalid("Offset trop grand"))?;
        let end = t
            .file_offset
            .checked_add(t.byte_length)
            .ok_or_else(|| invalid("Fin de tenseur trop grande"))?;
        if end > size {
            return Err(invalid("Tenseur hors du fichier (fichier tronqué)"));
        }
        ranges.push((t.file_offset, end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|p| p[0].1 > p[1].0) {
        return Err(invalid("Tenseurs qui se chevauchent"));
    }
    Ok(ModelInfo {
        format_version: version,
        file_bytes: size,
        data_offset,
        metadata,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    fn fixture() -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3_u32.to_le_bytes());
        b.extend(1_u64.to_le_bytes());
        b.extend(0_u64.to_le_bytes());
        b.extend(1_u64.to_le_bytes());
        b.push(b'x');
        b.extend(1_u32.to_le_bytes());
        b.extend(32_u64.to_le_bytes());
        b.extend(2_u32.to_le_bytes());
        b.extend(0_u64.to_le_bytes());
        b.resize(64 + 18, 0);
        b
    }
    #[test]
    fn cached_index_is_owned_and_invalidates_on_file_changes() {
        let dir = std::env::temp_dir().join(format!("adaforge-index-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.gguf");
        let other = dir.join("b.gguf");
        let inspector = GgufInspector::default();
        let bytes = fixture();
        std::fs::write(&path, &bytes).unwrap();
        let mut first = inspector.inspect(&path).unwrap();
        first.tensors[0].name = "modified copy".into();
        assert_eq!(inspector.inspect(&path).unwrap().tensors[0].name, "x");
        let mut renamed = bytes.clone();
        renamed[32] = b'y';
        std::fs::write(&other, &renamed).unwrap();
        assert_eq!(inspector.inspect(&other).unwrap().tensors[0].name, "y");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&other)
            .unwrap();
        file.set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(100))
            .unwrap();
        inspector.inspect(&other).unwrap();
        let mut changed = renamed;
        changed[32] = b'z';
        std::fs::write(&other, &changed).unwrap();
        assert_eq!(inspector.inspect(&other).unwrap().tensors[0].name, "z");
        std::fs::write(&other, b"GGUF").unwrap();
        assert!(inspector.inspect(&other).is_err());
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&other).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
    #[test]
    fn reads_quantized_tensor_and_rejects_truncation() {
        let b = fixture();
        let m = parse(Cursor::new(&b), b.len() as u64).unwrap();
        assert_eq!(m.data_offset, 64);
        assert_eq!(m.tensors[0].byte_length, 18);
        assert_eq!(m.tensors[0].dimensions, vec![32]);
        for n in 0..b.len() {
            assert!(parse(Cursor::new(&b[..n]), n as u64).is_err(), "length {n}");
        }
    }
    #[test]
    fn rejects_bad_signature_version_type_and_offset() {
        for (pos, value) in [(0, 0), (4, 2), (45, 99), (49, 1)] {
            let mut b = fixture();
            b[pos] = value;
            assert!(parse(Cursor::new(&b), b.len() as u64).is_err());
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use std::io::Cursor;
    fn string(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }
    fn model(offset: u64, alignment: u32) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3_u32.to_le_bytes());
        b.extend(2_u64.to_le_bytes());
        b.extend(2_u64.to_le_bytes());
        string(&mut b, "general.alignment");
        b.extend(4_u32.to_le_bytes());
        b.extend(alignment.to_le_bytes());
        string(&mut b, "gemma4.test");
        b.extend(9_u32.to_le_bytes());
        b.extend(7_u32.to_le_bytes());
        b.extend(2_u64.to_le_bytes());
        b.extend([1, 0]);
        for (name, off) in [("a", 0), ("b", offset)] {
            string(&mut b, name);
            b.extend(1_u32.to_le_bytes());
            b.extend(256_u64.to_le_bytes());
            b.extend(14_u32.to_le_bytes());
            b.extend(off.to_le_bytes());
        }
        let start = b.len().div_ceil(32) * 32;
        b.resize(start + 512, 0);
        b
    }
    #[test]
    fn preserves_array_values_and_q6k_layout() {
        let b = model(224, 32);
        let m = parse(Cursor::new(&b), b.len() as u64).unwrap();
        assert_eq!(m.tensors[0].byte_length, 210);
        assert!(
            matches!(m.metadata.get("gemma4.test"),Some(V::Array(v)) if matches!(v.as_slice(),[V::Bool(true),V::Bool(false)]))
        );
    }
    #[test]
    fn rejects_overlaps_unaligned_offsets_and_invalid_alignment() {
        for (offset, alignment) in [(0, 32), (1, 32), (224, 0), (224, 3)] {
            let b = model(offset, alignment);
            assert!(parse(Cursor::new(&b), b.len() as u64).is_err());
        }
    }
}
