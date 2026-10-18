//! Géométrie et état logique d’un cache KV contigu pour une couche/séquence.
#[derive(Debug, Clone, Copy)]
pub struct KvLayout {
    capacity: u32,
    heads: u32,
    width: u32,
    elements: usize,
}
impl KvLayout {
    pub fn new(capacity: u32, heads: u32, width: u32) -> Result<Self, String> {
        if capacity == 0 || heads == 0 || width == 0 {
            return Err("Dimensions cache KV nulles".into());
        }
        let elements = u64::from(capacity)
            .checked_mul(u64::from(heads))
            .and_then(|v| v.checked_mul(u64::from(width)))
            .ok_or("Cache KV trop grand")?;
        elements
            .checked_mul(8)
            .ok_or("Taille KV en octets excessive")?;
        let elements = usize::try_from(elements).map_err(|_| "Cache KV non adressable")?;
        Ok(Self {
            capacity,
            heads,
            width,
            elements,
        })
    }
    pub fn capacity(self) -> u32 {
        self.capacity
    }
    pub fn heads(self) -> u32 {
        self.heads
    }
    pub fn width(self) -> u32 {
        self.width
    }
    pub fn elements(self) -> usize {
        self.elements
    }
    pub fn token_elements(self) -> usize {
        self.elements / self.capacity as usize
    }
    pub fn total_bytes(self) -> u64 {
        self.elements as u64 * 8
    }
}
#[derive(Debug)]
pub struct KvState {
    layout: KvLayout,
    len: u32,
}
impl KvState {
    pub fn new(layout: KvLayout) -> Self {
        Self { layout, len: 0 }
    }
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn append_range(&self, tokens: u32) -> Result<std::ops::Range<usize>, String> {
        let end = self
            .len
            .checked_add(tokens)
            .ok_or("Dépassement capacité KV")?;
        if tokens == 0 || end > self.layout.capacity {
            return Err("Dépassement capacité KV ou ajout vide".into());
        }
        let n = self.layout.token_elements();
        Ok(self.len as usize * n..end as usize * n)
    }
    pub fn commit(&mut self, tokens: u32) -> Result<(), String> {
        self.append_range(tokens)?;
        self.len += tokens;
        Ok(())
    }
    pub fn reset(&mut self) {
        self.len = 0;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tracks_prefix_and_rejects_overflow_without_mutation() {
        let layout = KvLayout::new(4, 2, 3).unwrap();
        let mut state = KvState::new(layout);
        assert_eq!(layout.total_bytes(), 192);
        assert_eq!(state.append_range(3).unwrap(), 0..18);
        state.commit(3).unwrap();
        assert_eq!(state.append_range(1).unwrap(), 18..24);
        assert!(state.commit(2).is_err());
        assert_eq!(state.len(), 3);
        state.commit(1).unwrap();
        assert!(state.commit(1).is_err());
        assert_eq!(state.len(), 4);
        state.reset();
        assert!(state.is_empty());
        assert!(state.append_range(0).is_err());
    }
    #[test]
    fn rejects_zero_and_overflowing_layouts() {
        for dims in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (u32::MAX, u32::MAX, u32::MAX),
        ] {
            assert!(KvLayout::new(dims.0, dims.1, dims.2).is_err());
        }
    }
}
