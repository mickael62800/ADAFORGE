//! RMSNorm à échelle multiplicative directe, comme les poids GGUF Gemma 4.
#[derive(Debug)]
pub struct RmsNorm {
    weights: Vec<f32>,
    epsilon: f32,
}
impl RmsNorm {
    pub fn new(weights: Vec<f32>, epsilon: f32) -> Result<Self, String> {
        if weights.is_empty() || weights.len() > 65536 || weights.iter().any(|v| !v.is_finite()) {
            return Err("Poids RMSNorm invalides (1 à 65536 éléments finis requis)".into());
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("Epsilon RMSNorm doit être fini et positif".into());
        }
        Ok(Self { weights, epsilon })
    }
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
    /// Référence indépendante FP64, pour un ou plusieurs vecteurs contigus.
    pub fn reference(&self, input: &[f32]) -> Result<Vec<f64>, String> {
        let n = self.weights.len();
        if input.is_empty()
            || !input.len().is_multiple_of(n)
            || input.iter().any(|v| !v.is_finite())
        {
            return Err("Entrée RMSNorm invalide".into());
        }
        let mut output = Vec::with_capacity(input.len());
        for row in input.chunks_exact(n) {
            let mean = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / n as f64;
            let scale = 1.0 / (mean + f64::from(self.epsilon)).sqrt();
            for (&x, &w) in row.iter().zip(&self.weights) {
                output.push(f64::from(x) * scale * f64::from(w));
            }
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_result_and_batch_independence() {
        let n = RmsNorm::new(vec![2.0, -3.0], 1.0).unwrap();
        let y = n.reference(&[1.0, 1.0, 0.0, 0.0]).unwrap();
        assert!((y[0] - 2.0 / 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((y[1] + 3.0 / 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(&y[2..], &[0.0, 0.0]);
        // Direct multiplication : adding 1 to these weights would fail this oracle.
        assert_eq!(
            RmsNorm::new(vec![0.0], 1e-6)
                .unwrap()
                .reference(&[2.0])
                .unwrap(),
            vec![0.0]
        );
    }
    #[test]
    fn rejects_invalid_parameters_and_inputs() {
        for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(RmsNorm::new(vec![1.0], eps).is_err());
        }
        assert!(RmsNorm::new(vec![], 1e-6).is_err());
        assert!(RmsNorm::new(vec![f32::NAN], 1e-6).is_err());
        let n = RmsNorm::new(vec![1.0; 3], 1e-6).unwrap();
        for x in [vec![], vec![0.0; 2], vec![f32::INFINITY; 3]] {
            assert!(n.reference(&x).is_err());
        }
    }
}
