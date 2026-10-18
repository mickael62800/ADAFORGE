//! Cache KV FP32 contigu : [token][tête KV][canal], buffers K et V distincts.
use super::failure;
use adaforge_application::{DiagnosticError, KvCacheReport};
use adaforge_domain::{KvLayout, KvState};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, sys};
use std::sync::Arc;

pub(crate) struct ResidentKv<'a> {
    pub keys: &'a CudaSlice<f32>,
    pub values: &'a CudaSlice<f32>,
    pub tokens: u32,
    pub layout: KvLayout,
    /// Physical slots: token t is stored in slot t % ring.
    pub ring: u32,
}
pub(crate) struct KvCache {
    layout: KvLayout,
    state: KvState,
    stream: Arc<CudaStream>,
    keys: CudaSlice<f32>,
    values: CudaSlice<f32>,
    /// Physical slots (<= logical capacity). Token t lives in slot t % ring.
    ring: u32,
    poisoned: bool,
}
impl KvCache {
    pub(crate) fn new(stream: Arc<CudaStream>, layout: KvLayout) -> Result<Self, DiagnosticError> {
        Self::with_ring(stream, layout, layout.capacity())
    }
    /// Circular cache for sliding-window layers: only `ring` token slots are
    /// allocated. The caller must keep ring >= window + (largest batch - 1)
    /// so that no write ever overwrites a token that is still visible.
    pub(crate) fn with_ring(
        stream: Arc<CudaStream>,
        layout: KvLayout,
        ring: u32,
    ) -> Result<Self, DiagnosticError> {
        let ring = ring.clamp(1, layout.capacity());
        let elements = ring as usize * layout.token_elements();
        let keys = stream
            .alloc_zeros::<f32>(elements)
            .map_err(|e| failure("Allocation cache K", e))?;
        let values = stream
            .alloc_zeros::<f32>(elements)
            .map_err(|e| failure("Allocation cache V", e))?;
        Ok(Self {
            layout,
            state: KvState::new(layout),
            stream,
            keys,
            values,
            ring,
            poisoned: false,
        })
    }
    pub(crate) fn ring(&self) -> u32 {
        self.ring
    }
    pub(crate) fn graph_buffers(
        &mut self,
    ) -> Result<(&mut CudaSlice<f32>, &mut CudaSlice<f32>), DiagnosticError> {
        self.ensure_valid()?;
        Ok((&mut self.keys, &mut self.values))
    }
    pub(crate) fn graph_commit(&mut self, position: u32) -> Result<(), DiagnosticError> {
        self.ensure_valid()?;
        if self.state.len() != position {
            return Err(DiagnosticError("Position cache graphe invalide".into()));
        }
        self.state.append_range(1).map_err(DiagnosticError)?;
        self.state.commit(1).map_err(DiagnosticError)
    }
    /// Ajoute des buffers déjà sur le GPU. La longueur décrit le préfixe en file
    /// d’exécution ; les futurs lecteurs doivent utiliser ce même stream.
    pub(crate) fn append(
        &mut self,
        keys: &CudaSlice<f32>,
        values: &CudaSlice<f32>,
    ) -> Result<(), DiagnosticError> {
        self.ensure_valid()?;
        if keys.context() != self.stream.context() || values.context() != self.stream.context() {
            return Err(DiagnosticError(
                "Contexte CUDA du cache incompatible".into(),
            ));
        }
        let token_elements = self.layout.token_elements();
        if keys.len() != values.len() || !keys.len().is_multiple_of(token_elements) {
            return Err(DiagnosticError("Tailles des entrées KV invalides".into()));
        }
        let tokens = u32::try_from(keys.len() / token_elements)
            .map_err(|_| DiagnosticError("Ajout KV trop grand".into()))?;
        self.state.append_range(tokens).map_err(DiagnosticError)?;
        let start = self.state.len();
        let ring = self.ring;
        let mut copy = || {
            // One copy per token: slots wrap around the ring.
            for t in 0..tokens as usize {
                let slot = ((start as usize + t) % ring as usize) * token_elements;
                let src = t * token_elements..(t + 1) * token_elements;
                self.stream.memcpy_dtod(
                    &keys.slice(src.clone()),
                    &mut self.keys.slice_mut(slot..slot + token_elements),
                )?;
                self.stream.memcpy_dtod(
                    &values.slice(src),
                    &mut self.values.slice_mut(slot..slot + token_elements),
                )?;
            }
            Ok::<(), cudarc::driver::DriverError>(())
        };
        if let Err(e) = copy() {
            self.poisoned = true;
            return Err(failure("Copie KV (cache invalidé)", e));
        }
        self.state.commit(tokens).map_err(DiagnosticError)
    }
    pub(crate) fn resident(&self) -> Result<ResidentKv<'_>, DiagnosticError> {
        self.ensure_valid()?;
        Ok(ResidentKv {
            keys: &self.keys,
            values: &self.values,
            tokens: self.state.len(),
            layout: self.layout,
            ring: self.ring,
        })
    }
    fn ensure_valid(&self) -> Result<(), DiagnosticError> {
        if self.poisoned {
            Err(DiagnosticError(
                "Cache KV invalidé par erreur CUDA ; recréer le cache".into(),
            ))
        } else {
            Ok(())
        }
    }
    /// Records `tokens` entries written directly into the buffers (batched
    /// prefill) starting at `position`, which must be the current length.
    pub(crate) fn commit_batch(&mut self, position: u32, tokens: u32) -> Result<(), DiagnosticError> {
        self.ensure_valid()?;
        if self.state.len() != position {
            return Err(DiagnosticError("Position cache lot invalide".into()));
        }
        self.state.append_range(tokens).map_err(DiagnosticError)?;
        self.state.commit(tokens).map_err(DiagnosticError)
    }
    pub(crate) fn truncate(&mut self, len: u32) -> Result<(), DiagnosticError> {
        self.ensure_valid()?;
        self.state.truncate(len).map_err(DiagnosticError)
    }
    pub(crate) fn reset(&mut self) -> Result<(), DiagnosticError> {
        self.ensure_valid()?;
        self.state.reset();
        Ok(())
    }
    /// Diagnostic seulement. Le moteur utilisera les données résidentes.
    pub(crate) fn snapshot(&self) -> Result<(Vec<f32>, Vec<f32>), DiagnosticError> {
        self.ensure_valid()?;
        // Physical contents (slot order) of the occupied slots.
        let end = self.state.len().min(self.ring) as usize * self.layout.token_elements();
        if end == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let keys = self
            .stream
            .clone_dtoh(&self.keys.slice(..end))
            .map_err(|e| failure("Lecture cache K", e))?;
        let values = self
            .stream
            .clone_dtoh(&self.values.slice(..end))
            .map_err(|e| failure("Lecture cache V", e))?;
        self.stream
            .synchronize()
            .map_err(|e| failure("Synchronisation cache", e))?;
        Ok((keys, values))
    }
}
pub(crate) fn check(layout: KvLayout) -> Result<KvCacheReport, DiagnosticError> {
    if layout.capacity() != 8 {
        return Err(DiagnosticError(
            "Le diagnostic KV utilise une capacité de huit tokens".into(),
        ));
    }
    let ctx = CudaContext::new(0).map_err(|e| failure("Initialisation cache", e))?;
    let name = ctx.name().map_err(|e| failure("Nom GPU", e))?;
    let major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| failure("CUDA", e))?;
    let minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| failure("CUDA", e))?;
    if !name.contains("RTX 4090") || (major, minor) != (8, 9) {
        return Err(DiagnosticError("RTX 4090 sm_89 requise".into()));
    }
    let stream = ctx.default_stream();
    let mut cache = KvCache::new(stream.clone(), layout)?;
    let mut expected_k = Vec::new();
    let mut expected_v = Vec::new();
    let mut verified = 0;
    for count in [3, 1, 4] {
        let start = expected_k.len() / layout.token_elements();
        let (k, v) = data(layout, start, count);
        let kg = stream
            .clone_htod(&k)
            .map_err(|e| failure("Entrée K test", e))?;
        let vg = stream
            .clone_htod(&v)
            .map_err(|e| failure("Entrée V test", e))?;
        cache.append(&kg, &vg)?;
        expected_k.extend(k);
        expected_v.extend(v);
        verified += verify(&cache, &expected_k, &expected_v)?;
    }
    let (k, v) = data(layout, 8, 1);
    let kg = stream
        .clone_htod(&k)
        .map_err(|e| failure("Entrée dépassement K", e))?;
    let vg = stream
        .clone_htod(&v)
        .map_err(|e| failure("Entrée dépassement V", e))?;
    if cache.append(&kg, &vg).is_ok() {
        return Err(DiagnosticError("Dépassement KV accepté".into()));
    }
    verified += verify(&cache, &expected_k, &expected_v)?;
    cache.reset()?;
    if cache.snapshot()? != (Vec::new(), Vec::new()) {
        return Err(DiagnosticError("Cache non vide après reset".into()));
    }
    let (k, v) = data(layout, 100, 2);
    let kg = stream
        .clone_htod(&k)
        .map_err(|e| failure("Réutilisation K", e))?;
    let vg = stream
        .clone_htod(&v)
        .map_err(|e| failure("Réutilisation V", e))?;
    cache.append(&kg, &vg)?;
    verified += verify(&cache, &k, &v)?;
    Ok(KvCacheReport {
        capacity: layout.capacity(),
        heads: layout.heads(),
        width: layout.width(),
        allocated_bytes: layout.total_bytes(),
        verified_values: verified,
        overflow_rejected: true,
        reset_verified: true,
    })
}
fn data(layout: KvLayout, start: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
    let n = layout.token_elements();
    let keys = (start * n..(start + count) * n)
        .map(|i| (i % 4093) as f32 * 0.25 - 512.0)
        .collect();
    let values = (start * n..(start + count) * n)
        .map(|i| 1024.0 - (i % 2039) as f32 * 0.5)
        .collect();
    (keys, values)
}
fn verify(cache: &KvCache, keys: &[f32], values: &[f32]) -> Result<usize, DiagnosticError> {
    let (actual_k, actual_v) = cache.snapshot()?;
    if actual_k != keys || actual_v != values {
        return Err(DiagnosticError("Contenu du cache KV incorrect".into()));
    }
    Ok(keys.len() + values.len())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_cache_keeps_prefix_rejects_overflow_and_resets() {
        for (heads, width) in [(2, 3), (8, 256), (1, 512)] {
            let report = check(KvLayout::new(8, heads, width).unwrap()).unwrap();
            assert!(report.overflow_rejected && report.reset_verified);
        }
    }
}

#[cfg(test)]
mod invalid_input_tests {
    use super::*;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn wrong_sizes_do_not_modify_cache_state() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut cache = KvCache::new(stream.clone(), KvLayout::new(8, 2, 3).unwrap()).unwrap();
        let short = stream.clone_htod(&[1.0_f32; 5]).unwrap();
        let valid = stream.clone_htod(&[2.0_f32; 6]).unwrap();
        assert!(cache.append(&short, &short).is_err());
        assert!(cache.append(&short, &valid).is_err());
        assert_eq!(cache.state.len(), 0);
        cache.append(&valid, &valid).unwrap();
        assert_eq!(cache.snapshot().unwrap(), (vec![2.0; 6], vec![2.0; 6]));
    }
}
