//! Projection de sortie d’attention, RMSNorm post-attention puis résiduel.
use crate::{Q4Matrix, RmsNorm};
#[derive(Debug)]
pub struct AttentionOutput {
    matrix: Q4Matrix,
    norm: RmsNorm,
}
impl AttentionOutput {
    pub fn new(matrix: Q4Matrix, norm: RmsNorm) -> Result<Self, String> {
        if matrix.rows() as usize != norm.weights().len() {
            return Err("Dimensions projection/norme post-attention incompatibles".into());
        }
        Ok(Self { matrix, norm })
    }
    pub fn matrix(&self) -> &Q4Matrix {
        &self.matrix
    }
    pub fn norm(&self) -> &RmsNorm {
        &self.norm
    }
    pub fn reference(&self, attention: &[f32], residual: &[f32]) -> Result<Vec<f64>, String> {
        if residual.len() != self.norm.weights().len() || residual.iter().any(|v| !v.is_finite()) {
            return Err("Résiduel invalide".into());
        }
        let projected: Vec<f32> = self
            .matrix
            .reference_matvec(attention)?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let normalized = self.norm.reference(&projected)?;
        Ok(normalized
            .iter()
            .zip(residual)
            .map(|(&a, &b)| f64::from(a as f32) + f64::from(b))
            .collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_shape_mismatch_and_preserves_residual_for_zero_branch() {
        let matrix = Q4Matrix::new(32, 3, vec![0; 54]).unwrap();
        assert!(AttentionOutput::new(matrix, RmsNorm::new(vec![1.0; 2], 1e-6).unwrap()).is_err());
        let branch = AttentionOutput::new(
            Q4Matrix::new(32, 3, vec![0; 54]).unwrap(),
            RmsNorm::new(vec![2.0; 3], 1e-6).unwrap(),
        )
        .unwrap();
        assert_eq!(
            branch.reference(&[1.0; 32], &[1.0, -2.0, 3.0]).unwrap(),
            vec![1.0, -2.0, 3.0]
        );
        assert!(branch.reference(&[1.0; 32], &[1.0]).is_err());
    }
}
