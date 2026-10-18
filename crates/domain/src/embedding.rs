//! Lignes sélectionnées d’une table d’embeddings Q6_K (256 poids / 210 octets).
#[derive(Debug)]
pub struct EmbeddingBatch {
    width: u32,
    vocabulary: u32,
    token_ids: Vec<u32>,
    bytes: Vec<u8>,
}
impl EmbeddingBatch {
    pub fn new(
        width: u32,
        vocabulary: u32,
        token_ids: Vec<u32>,
        bytes: Vec<u8>,
    ) -> Result<Self, String> {
        let size = u64::from(width / 256) * 210 * token_ids.len() as u64;
        if width == 0
            || width > 65536
            || !width.is_multiple_of(256)
            || vocabulary == 0
            || token_ids.is_empty()
            || token_ids.len() > 256
            || token_ids.iter().any(|&i| i >= vocabulary)
            || size != bytes.len() as u64
        {
            return Err("Embeddings Q6_K : dimensions, IDs ou taille invalides".into());
        }
        if bytes
            .chunks_exact(210)
            .any(|b| !crate::q4::half_to_f32(u16::from_le_bytes([b[208], b[209]])).is_finite())
        {
            return Err("Échelle Q6_K non finie".into());
        }
        Ok(Self {
            width,
            vocabulary,
            token_ids,
            bytes,
        })
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn vocabulary(&self) -> u32 {
        self.vocabulary
    }
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn scale(&self) -> f32 {
        (self.width as f32).sqrt()
    }
    pub fn reference(&self, scaled: bool) -> Vec<f32> {
        let mut output = vec![0.0; self.token_ids.len() * self.width as usize];
        for (index, block) in self.bytes.chunks_exact(210).enumerate() {
            let d = crate::q4::half_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let out = &mut output[index * 256..(index + 1) * 256];
            // CPU walks two 128-element groups and four 32-element lanes.
            for group in 0..2 {
                for lane in 0..32 {
                    for quarter in 0..4 {
                        let low = block[group * 64 + lane + (quarter % 2) * 32];
                        let low = if quarter < 2 { low & 15 } else { low >> 4 };
                        let high = (block[128 + group * 32 + lane] >> (quarter * 2)) & 3;
                        let quant = i32::from(low | high << 4) - 32;
                        let scale = block[192 + group * 8 + quarter * 2 + lane / 16] as i8;
                        let value = (d * f32::from(scale)) * quant as f32;
                        out[group * 128 + quarter * 32 + lane] =
                            if scaled { value * self.scale() } else { value };
                    }
                }
            }
        }
        output
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn known_block() -> Vec<u8> {
        let mut b = vec![0; 210];
        b[208..].copy_from_slice(&0x3800_u16.to_le_bytes());
        // Construct packed data from independently specified 6-bit integer codes.
        for i in 0..256 {
            let code = (i % 64) as u8;
            let g = i / 128;
            let q = (i % 128) / 32;
            let l = i % 32;
            b[g * 64 + l + (q % 2) * 32] |= (code & 15) << if q < 2 { 0 } else { 4 };
            b[128 + g * 32 + l] |= (code >> 4) << (q * 2);
        }
        for i in 0..16 {
            b[192 + i] = ((i as i8) - 8) as u8;
        }
        b
    }
    #[test]
    fn decodes_all_codes_and_signed_subscales_in_the_right_order() {
        let batch = EmbeddingBatch::new(256, 1, vec![0], known_block()).unwrap();
        let decoded = batch.reference(false);
        for (i, &v) in decoded.iter().enumerate() {
            let expected = 0.5 * ((i / 16) as i32 - 8) as f32 * ((i % 64) as i32 - 32) as f32;
            assert_eq!(v, expected, "channel {i}");
        }
        for (&raw, &scaled) in decoded.iter().zip(&batch.reference(true)) {
            assert_eq!(scaled, raw * 16.0);
        }
    }
    #[test]
    fn validates_rows_ids_and_scales() {
        assert!(EmbeddingBatch::new(255, 1, vec![0], known_block()).is_err());
        assert!(EmbeddingBatch::new(256, 1, vec![1], known_block()).is_err());
        let mut b = known_block();
        b[208..].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(EmbeddingBatch::new(256, 1, vec![0], b).is_err());
    }
}
