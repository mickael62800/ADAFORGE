//! Stockage Q4_0 : blocs de 32 poids, échelle FP16 et 16 octets de nibbles.
#[derive(Debug)]
pub struct Q4Matrix {
    columns: u32,
    rows: u32,
    bytes: Vec<u8>,
}
impl Q4Matrix {
    pub fn new(columns: u32, rows: u32, bytes: Vec<u8>) -> Result<Self, String> {
        if columns == 0 || rows == 0 || !columns.is_multiple_of(32) {
            return Err("Dimensions Q4_0 invalides".into());
        }
        let size = u64::from(columns / 32) * u64::from(rows) * 18;
        if size != bytes.len() as u64 {
            return Err("Taille Q4_0 incohérente".into());
        }
        if bytes
            .chunks_exact(18)
            .any(|b| !half_to_f32(u16::from_le_bytes([b[0], b[1]])).is_finite())
        {
            return Err("Échelle Q4_0 non finie".into());
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

    /// Oracle CPU : déquantification FP32 exacte et accumulation FP64.
    /// Aucune matrice déquantifiée complète n’est matérialisée.
    pub fn reference_matvec(&self, input: &[f32]) -> Result<Vec<f64>, String> {
        if input.len() != self.columns as usize || input.iter().any(|v| !v.is_finite()) {
            return Err("Vecteur d’entrée invalide".into());
        }
        let blocks_per_row = self.columns as usize / 32;
        let mut result = Vec::with_capacity(self.rows as usize);
        for row in self.bytes.chunks_exact(blocks_per_row * 18) {
            let mut sum = 0.0_f64;
            for (block_index, block) in row.chunks_exact(18).enumerate() {
                let scale = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for j in 0..16 {
                    let lo = i32::from(block[2 + j] & 15) - 8;
                    let hi = i32::from(block[2 + j] >> 4) - 8;
                    sum += f64::from(scale * lo as f32) * f64::from(input[block_index * 32 + j]);
                    sum +=
                        f64::from(scale * hi as f32) * f64::from(input[block_index * 32 + 16 + j]);
                }
            }
            if !sum.is_finite() {
                return Err("Résultat CPU non fini".into());
            }
            result.push(sum);
        }
        Ok(result)
    }
}

pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    match exponent {
        0 => sign * f32::from(mantissa) * 2.0_f32.powi(-24),
        31 => {
            if mantissa == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * f32::from_bits((u32::from(exponent) + 112) << 23 | u32::from(mantissa) << 13),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_nibbles_scale_sign_and_order() {
        let mut b = vec![0; 18];
        b[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        for j in 0..16 {
            b[2 + j] = j as u8 | ((15 - j) as u8) << 4;
        }
        let m = Q4Matrix::new(32, 1, b).unwrap();
        for column in 0..32 {
            let mut x = vec![0.0; 32];
            x[column] = 1.0;
            let q = if column < 16 {
                column as i32 - 8
            } else {
                23 - column as i32
            };
            assert_eq!(m.reference_matvec(&x).unwrap(), vec![f64::from(q) * 0.5]);
        }
    }
    #[test]
    fn validates_dimensions_bytes_inputs_and_scales() {
        assert!(Q4Matrix::new(31, 1, vec![0; 18]).is_err());
        assert!(Q4Matrix::new(32, 0, vec![]).is_err());
        assert!(Q4Matrix::new(32, 1, vec![0; 17]).is_err());
        let mut b = vec![0; 18];
        b[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(Q4Matrix::new(32, 1, b).is_err());
        let m = Q4Matrix::new(32, 1, vec![0; 18]).unwrap();
        assert!(m.reference_matvec(&[0.0]).is_err());
        assert!(m.reference_matvec(&[f32::NAN; 32]).is_err());
    }
    #[test]
    fn half_edge_cases() {
        assert_eq!(half_to_f32(1), 2.0_f32.powi(-24));
        assert_eq!(half_to_f32(0x8001), -2.0_f32.powi(-24));
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x7bff), 65504.0);
        assert!(half_to_f32(0x7e00).is_nan());
    }
}
