//! Tokenizer BPE Gemma 4 : UTF-8, metaspace, rangs GGUF et fallback octets.
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};
pub struct GemmaTokenizer {
    tokens: Vec<String>,
    types: Vec<i32>,
    lookup: HashMap<String, u32>,
    ranks: HashMap<String, HashMap<String, u32>>,
    bytes: [u32; 256],
    specials: Vec<u32>,
    bos: u32,
}
#[derive(Clone, Copy, Default)]
pub struct TokenizeOptions {
    pub add_bos: bool,
    pub parse_special: bool,
}
struct Node {
    start: usize,
    end: usize,
    previous: Option<usize>,
    next: Option<usize>,
    generation: u32,
    alive: bool,
}
type Candidate = Reverse<(u32, usize, u32, usize, u32)>;
impl GemmaTokenizer {
    pub fn new(
        tokens: Vec<String>,
        mut types: Vec<i32>,
        merges: Vec<String>,
        bos: u32,
    ) -> Result<Self, String> {
        if tokens.is_empty()
            || tokens.len() > 1_000_000
            || tokens.len() != types.len()
            || bos as usize >= tokens.len()
            || tokens[bos as usize] != "<bos>"
        {
            return Err("Vocabulaire Gemma invalide".into());
        }
        let mut lookup = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            if token.is_empty() || lookup.insert(token.clone(), id as u32).is_some() {
                return Err("Token vide ou dupliqué".into());
            }
        }
        // Gemma control markers may carry USER_DEFINED/NORMAL attributes in GGUF.
        // Classify them consistently with the reference Gemma vocabulary loader.
        for marker in ["<|tool_response>"] {
            if let Some(&id) = lookup.get(marker) {
                types[id as usize] = 3;
            }
        }
        let mut bytes = [u32::MAX; 256];
        let mut specials = Vec::new();
        for (id, (text, kind)) in tokens.iter().zip(&types).enumerate() {
            if *kind == 6 {
                let hex = text
                    .strip_prefix("<0x")
                    .and_then(|s| s.strip_suffix('>'))
                    .filter(|s| s.len() == 2)
                    .ok_or("Token octet invalide")?;
                let byte = u8::from_str_radix(hex, 16).map_err(|_| "Token octet invalide")?;
                if bytes[byte as usize] != u32::MAX {
                    return Err("Token octet dupliqué".into());
                }
                bytes[byte as usize] = id as u32;
            }
            if *kind == 3 || *kind == 4 {
                specials.push(id as u32);
            }
            if !(1..=6).contains(kind) {
                return Err("Type de token inconnu".into());
            }
        }
        if bytes.contains(&u32::MAX) {
            return Err("Fallback des 256 octets incomplet".into());
        }
        let mut ranks: HashMap<String, HashMap<String, u32>> = HashMap::new();
        for (rank, merge) in merges.into_iter().enumerate() {
            let pos = merge
                .as_bytes()
                .iter()
                .enumerate()
                .skip(1)
                .find_map(|(i, &b)| (b == b' ').then_some(i))
                .ok_or("Paire BPE invalide")?;
            let (left, right) = (&merge[..pos], &merge[pos + 1..]);
            if left.is_empty() || right.is_empty() {
                return Err("Paire BPE vide".into());
            }
            ranks
                .entry(left.into())
                .or_default()
                .entry(right.into())
                .or_insert(rank as u32);
        }
        Ok(Self {
            tokens,
            types,
            lookup,
            ranks,
            bytes,
            specials,
            bos,
        })
    }
    /// Sous-ensemble du template Gemma : un tour utilisateur, sans outils ni pensée.
    pub fn encode_user_turn(&self, text: &str) -> Result<Vec<u32>, String> {
        for marker in ["<|turn>", "<turn|>", "<|channel>", "<channel|>"] {
            if self.token_id(marker).is_none() {
                return Err("Marqueur de conversation Gemma absent".into());
            }
        }
        let mut ids = self.encode(
            "<|turn>user\n",
            TokenizeOptions {
                add_bos: true,
                parse_special: true,
            },
        )?;
        ids.extend(self.encode(
            text.trim(),
            TokenizeOptions {
                add_bos: false,
                parse_special: false,
            },
        )?);
        ids.extend(self.encode(
            "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
            TokenizeOptions {
                add_bos: false,
                parse_special: true,
            },
        )?);
        Ok(ids)
    }
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.lookup.get(text).copied()
    }
    pub fn vocabulary(&self) -> usize {
        self.tokens.len()
    }
    pub fn encode(&self, text: &str, options: TokenizeOptions) -> Result<Vec<u32>, String> {
        if text.len() > 65536 {
            return Err("Texte trop long pour le diagnostic (64 Kio maximum)".into());
        }
        let mut out = Vec::new();
        if options.add_bos {
            out.push(self.bos);
        }
        let mut remaining = text;
        while !remaining.is_empty() {
            let found = self
                .specials
                .iter()
                .filter(|&&id| options.parse_special || self.types[id as usize] == 4)
                .filter_map(|&id| {
                    remaining
                        .find(&self.tokens[id as usize])
                        .map(|p| (p, self.tokens[id as usize].len(), id))
                })
                .min_by_key(|&(p, len, _)| (p, Reverse(len)));
            if let Some((position, len, id)) = found {
                self.encode_raw(&remaining[..position], &mut out);
                out.push(id);
                remaining = &remaining[position + len..];
            } else {
                self.encode_raw(remaining, &mut out);
                break;
            }
        }
        Ok(out)
    }
    fn encode_raw(&self, text: &str, out: &mut Vec<u32>) {
        let text = text.replace(' ', "▁");
        let mut start = 0;
        let mut kind = None;
        for (i, ch) in text.char_indices() {
            let newline = ch == '\n';
            if kind.is_some_and(|k| k != newline) {
                self.bpe(&text[start..i], out);
                start = i;
            }
            kind = Some(newline);
        }
        if start < text.len() {
            self.bpe(&text[start..], out);
        }
    }
    fn offer(&self, text: &str, nodes: &[Node], left: usize, queue: &mut BinaryHeap<Candidate>) {
        let l = &nodes[left];
        if !l.alive {
            return;
        }
        if let Some(right) = l.next {
            let r = &nodes[right];
            if let Some(&rank) = self
                .ranks
                .get(&text[l.start..l.end])
                .and_then(|m| m.get(&text[r.start..r.end]))
            {
                queue.push(Reverse((rank, left, l.generation, right, r.generation)));
            }
        }
    }
    fn bpe(&self, text: &str, out: &mut Vec<u32>) {
        if text.chars().all(|c| c == '\n')
            && let Some(&id) = self.lookup.get(text)
        {
            out.push(id);
            return;
        }
        let mut nodes: Vec<Node> = text
            .char_indices()
            .enumerate()
            .map(|(i, (start, c))| Node {
                start,
                end: start + c.len_utf8(),
                previous: i.checked_sub(1),
                next: Some(i + 1),
                generation: 0,
                alive: true,
            })
            .collect();
        if let Some(last) = nodes.last_mut() {
            last.next = None;
        } else {
            return;
        }
        let mut queue = BinaryHeap::new();
        for i in 0..nodes.len() {
            self.offer(text, &nodes, i, &mut queue);
        }
        while let Some(Reverse((_, left, lg, right, rg))) = queue.pop() {
            if !nodes[left].alive
                || !nodes[right].alive
                || nodes[left].next != Some(right)
                || nodes[left].generation != lg
                || nodes[right].generation != rg
            {
                continue;
            }
            nodes[left].end = nodes[right].end;
            nodes[left].next = nodes[right].next;
            nodes[left].generation += 1;
            nodes[right].alive = false;
            if let Some(next) = nodes[left].next {
                nodes[next].previous = Some(left);
            }
            if let Some(previous) = nodes[left].previous {
                self.offer(text, &nodes, previous, &mut queue);
            }
            self.offer(text, &nodes, left, &mut queue);
        }
        for node in nodes.iter().filter(|n| n.alive) {
            let piece = &text[node.start..node.end];
            if let Some(&id) = self.lookup.get(piece) {
                out.push(id);
            } else {
                out.extend(piece.as_bytes().iter().map(|&b| self.bytes[b as usize]));
            }
        }
    }
    pub fn decode(&self, ids: &[u32], render_special: bool) -> Result<String, String> {
        String::from_utf8(self.decode_bytes(ids, render_special)?)
            .map_err(|_| "Séquence de tokens UTF-8 incomplète ou invalide".into())
    }
    /// N’émet que le préfixe UTF-8 complet ; une fin coupée attend le token suivant.
    pub fn decode_generated_prefix(&self, ids: &[u32]) -> Result<String, String> {
        let bytes = self.decode_bytes(ids, false)?;
        let length = std::str::from_utf8(&bytes).map_or_else(|e| e.valid_up_to(), |_| bytes.len());
        Ok(std::str::from_utf8(&bytes[..length])
            .map_err(|e| e.to_string())?
            .to_owned())
    }
    pub fn decode_generated(&self, ids: &[u32]) -> Result<String, String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids, false)?).into_owned())
    }
    fn decode_bytes(&self, ids: &[u32], render_special: bool) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();
        for &id in ids {
            let text = self.tokens.get(id as usize).ok_or("ID hors vocabulaire")?;
            match self.types[id as usize] {
                6 => {
                    let b =
                        u8::from_str_radix(&text[3..5], 16).map_err(|_| "Token octet invalide")?;
                    output.push(b);
                }
                3 if !render_special => {}
                _ => output.extend(text.replace('▁', " ").bytes()),
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tokenizer() -> GemmaTokenizer {
        let mut tokens = vec!["<bos>".into(), "<eos>".into(), "<tag>".into()];
        let mut types = vec![3, 3, 4];
        for byte in 0..=255 {
            tokens.push(format!("<0x{byte:02X}>"));
            types.push(6);
        }
        for token in ["a", "b", "ab", "aba", "▁", "▁a", "\n", "\n\n"] {
            tokens.push(token.into());
            types.push(1);
        }
        GemmaTokenizer::new(
            tokens,
            types,
            vec!["a b".into(), "ab a".into(), "▁ a".into()],
            0,
        )
        .unwrap()
    }
    #[test]
    fn bpe_rank_leftmost_ties_and_metaspace() {
        let t = tokenizer();
        let o = TokenizeOptions::default();
        assert_eq!(
            t.encode("abab", o).unwrap(),
            vec![t.lookup["ab"], t.lookup["ab"]]
        );
        assert_eq!(t.encode("aba", o).unwrap(), vec![t.lookup["aba"]]);
        assert_eq!(t.encode(" a", o).unwrap(), vec![t.lookup["▁a"]]);
        assert_eq!(t.encode("\n\n", o).unwrap(), vec![t.lookup["\n\n"]]);
    }
    #[test]
    fn bytes_unicode_specials_bos_and_round_trip() {
        let t = tokenizer();
        for text in ["é🙂", "a\0b", " a  b\n\t", ""] {
            let ids = t.encode(text, TokenizeOptions::default()).unwrap();
            assert_eq!(t.decode(&ids, true).unwrap(), text);
        }
        assert_eq!(
            t.encode(
                "",
                TokenizeOptions {
                    add_bos: true,
                    parse_special: false
                }
            )
            .unwrap(),
            vec![0]
        );
        assert_eq!(
            t.encode(
                "<eos>",
                TokenizeOptions {
                    add_bos: false,
                    parse_special: true
                }
            )
            .unwrap(),
            vec![1]
        );
        assert_ne!(
            t.encode("<eos>", TokenizeOptions::default()).unwrap(),
            vec![1]
        );
        assert_eq!(
            t.encode("<tag>", TokenizeOptions::default()).unwrap(),
            vec![2]
        );
        assert_eq!(t.decode_generated_prefix(&[t.bytes[0xC3]]).unwrap(), "");
        assert_eq!(
            t.decode_generated_prefix(&[t.bytes[0xC3], t.bytes[0xA9]])
                .unwrap(),
            "é"
        );
        assert!(t.decode(&[t.bytes[0xC3]], true).is_err());
        assert_eq!(t.decode_generated(&[t.bytes[0xC3]]).unwrap(), "�");
        assert!(t.decode(&[u32::MAX], true).is_err());
        assert!(
            t.encode(&"a".repeat(65537), TokenizeOptions::default())
                .is_err()
        );
    }
}
