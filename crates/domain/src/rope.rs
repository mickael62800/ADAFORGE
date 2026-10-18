//! RoPE NeoX : paires entre les deux moitiés de chaque tête.
#[derive(Debug)]
pub struct Rope {
    width: u32,
    base: f64,
    factors: Vec<f32>,
    inverse_frequencies: Vec<f64>,
    context: u32,
}
impl Rope {
    pub fn new(width: u32, base: f64, factors: Vec<f32>, context: u32) -> Result<Self, String> {
        if width == 0
            || width > 65536
            || !width.is_multiple_of(2)
            || factors.len() != width as usize / 2
            || !base.is_finite()
            || base <= 1.0
            || context == 0
            || factors.iter().any(|f| !f.is_finite() || *f <= 0.0)
        {
            return Err("Paramètres RoPE invalides".into());
        }
        let inverse_frequencies = factors
            .iter()
            .enumerate()
            .map(|(i, f)| base.powf(-2.0 * i as f64 / f64::from(width)) / f64::from(*f))
            .collect();
        Ok(Self {
            width,
            base,
            factors,
            inverse_frequencies,
            context,
        })
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn base(&self) -> f64 {
        self.base
    }
    pub fn context(&self) -> u32 {
        self.context
    }
    pub fn factors(&self) -> &[f32] {
        &self.factors
    }
    pub fn inverse_frequencies(&self) -> &[f64] {
        &self.inverse_frequencies
    }
    pub fn reference(&self, input: &[f32], position: u32) -> Result<Vec<f64>, String> {
        if input.is_empty()
            || !input.len().is_multiple_of(self.width as usize)
            || position >= self.context
            || input.iter().any(|v| !v.is_finite())
        {
            return Err("Entrée ou position RoPE invalide".into());
        }
        let width = self.width as usize;
        let half = width / 2;
        let mut output = vec![0.0; input.len()];
        for (src, dst) in input
            .chunks_exact(width)
            .zip(output.chunks_exact_mut(width))
        {
            for i in 0..half {
                let angle = f64::from(position)
                    * self.base.powf(-2.0 * i as f64 / f64::from(self.width))
                    / f64::from(self.factors[i]);
                let (sin, cos) = angle.sin_cos();
                let a = f64::from(src[i]);
                let b = f64::from(src[i + half]);
                dst[i] = a * cos - b * sin;
                dst[i + half] = a * sin + b * cos;
            }
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_and_known_rotation_use_split_half_pairs() {
        let r = Rope::new(4, 10000.0, vec![1.0; 2], 100).unwrap();
        assert_eq!(
            r.reference(&[1.0, 2.0, 3.0, 4.0], 0).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        let y = r.reference(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert!((y[0] - 1.0_f64.cos()).abs() < 1e-12);
        assert!((y[2] - 1.0_f64.sin()).abs() < 1e-12);
        assert_eq!(y[1], 0.0);
        assert_eq!(y[3], 0.0);
    }
    #[test]
    fn preserves_norm_and_suppressed_pairs() {
        let r = Rope::new(4, 1000000.0, vec![1.0, 1e30], 262144).unwrap();
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = r.reference(&x, 262143).unwrap();
        assert!((y.iter().map(|v| v * v).sum::<f64>() - 30.0).abs() < 1e-10);
        assert_eq!(y[1], 2.0);
        assert_eq!(y[3], 4.0);
        assert!(r.reference(&x, 262144).is_err());
    }
    #[test]
    fn rejects_invalid_parameters_and_data() {
        assert!(Rope::new(3, 10000.0, vec![1.0], 10).is_err());
        assert!(Rope::new(4, 1.0, vec![1.0; 2], 10).is_err());
        for factor in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(Rope::new(4, 10000.0, vec![factor; 2], 10).is_err());
        }
        let r = Rope::new(4, 10000.0, vec![1.0; 2], 10).unwrap();
        assert!(r.reference(&[f32::NAN; 4], 0).is_err());
        assert!(r.reference(&[0.0; 3], 0).is_err());
    }
}
