//! Matrice Q6_K pour la tête de sortie, sans échelle des embeddings.
pub struct Q6Matrix {
    columns: u32,
    rows: u32,
    bytes: Vec<u8>,
}
impl Q6Matrix {
    pub fn new(columns: u32, rows: u32, bytes: Vec<u8>) -> Result<Self, String> {
        if columns == 0
            || columns > 65536
            || !columns.is_multiple_of(256)
            || rows == 0
            || rows > 1000000
            || u64::from(columns / 256) * u64::from(rows) * 210 != bytes.len() as u64
            || bytes
                .chunks_exact(210)
                .any(|b| !crate::q4::half_to_f32(u16::from_le_bytes([b[208], b[209]])).is_finite())
        {
            return Err("Matrice Q6_K invalide".into());
        }
        Ok(Self {
            columns,
            rows,
            bytes,
        })
    }
    pub fn columns(&self) -> u32 {
        self.columns
    }
    pub fn rows(&self) -> u32 {
        self.rows
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn reference(&self, x: &[f32]) -> Result<Vec<f64>, String> {
        if x.len() != self.columns as usize || x.iter().any(|v| !v.is_finite()) {
            return Err("Entrée Q6_K invalide".into());
        }
        let mut result = Vec::with_capacity(self.rows as usize);
        let row_bytes = self.columns as usize / 256 * 210;
        for row in self.bytes.chunks_exact(row_bytes) {
            let mut sum = 0.0;
            for (index, b) in row.chunks_exact(210).enumerate() {
                let d = crate::q4::half_to_f32(u16::from_le_bytes([b[208], b[209]]));
                for group in 0..2 {
                    for quarter in 0..4 {
                        for lane in 0..32 {
                            let low = b[group * 64 + lane + (quarter % 2) * 32];
                            let low = if quarter < 2 { low & 15 } else { low >> 4 };
                            let high = (b[128 + group * 32 + lane] >> (quarter * 2)) & 3;
                            let code = i32::from(low | high << 4) - 32;
                            let scale = b[192 + group * 8 + quarter * 2 + lane / 16] as i8;
                            let value = (d * f32::from(scale)) * code as f32;
                            sum += f64::from(value)
                                * f64::from(x[index * 256 + group * 128 + quarter * 32 + lane]);
                        }
                    }
                }
            }
            result.push(sum);
        }
        Ok(result)
    }
}
pub fn softcap(value: f32, cap: f32) -> Result<f32, String> {
    if !value.is_finite() || !cap.is_finite() || cap <= 0.0 {
        return Err("Softcapping invalide".into());
    }
    Ok((value / cap).tanh() * cap)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_constant_row_and_softcap() {
        let mut b = vec![0; 210];
        b[192..208].fill(1);
        b[208..].copy_from_slice(&0x3800_u16.to_le_bytes());
        let m = Q6Matrix::new(256, 1, b).unwrap();
        assert_eq!(m.reference(&vec![1.0; 256]).unwrap(), vec![-4096.0]);
        assert!(m.reference(&[1.0]).is_err());
        assert!(Q6Matrix::new(32, 1, vec![]).is_err());
        assert_eq!(softcap(0.0, 30.0).unwrap(), 0.0);
        assert_eq!(softcap(3000.0, 30.0).unwrap(), 30.0);
        assert!(softcap(1.0, 0.0).is_err());
        assert!(softcap(f32::NAN, 30.0).is_err());
    }
}
