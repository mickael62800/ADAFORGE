//! Attention causale pour une requête et un cache contigu depuis la position zéro.
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    q_heads: u32,
    kv_heads: u32,
    width: u32,
    window: Option<u32>,
    scale: f32,
}
impl AttentionConfig {
    pub fn new(
        q_heads: u32,
        kv_heads: u32,
        width: u32,
        window: Option<u32>,
        scale: f32,
    ) -> Result<Self, String> {
        if q_heads == 0
            || kv_heads == 0
            || width == 0
            || !q_heads.is_multiple_of(kv_heads)
            || window == Some(0)
            || !scale.is_finite()
            || scale <= 0.0
            || u64::from(q_heads) * u64::from(width) > u64::from(u32::MAX)
        {
            return Err("Configuration attention invalide".into());
        }
        Ok(Self {
            q_heads,
            kv_heads,
            width,
            window,
            scale,
        })
    }
    pub fn q_heads(self) -> u32 {
        self.q_heads
    }
    pub fn kv_heads(self) -> u32 {
        self.kv_heads
    }
    pub fn width(self) -> u32 {
        self.width
    }
    pub fn window(self) -> Option<u32> {
        self.window
    }
    pub fn scale(self) -> f32 {
        self.scale
    }
    pub fn visible_range(
        self,
        tokens: u32,
        position: u32,
    ) -> Result<std::ops::Range<usize>, String> {
        if tokens == 0 || position >= tokens {
            return Err("Position attention hors du cache".into());
        }
        let end = position as usize + 1;
        let start = self
            .window
            .map(|w| end.saturating_sub(w as usize))
            .unwrap_or(0);
        Ok(start..end)
    }
    pub fn reference(
        self,
        q: &[f32],
        keys: &[f32],
        values: &[f32],
        tokens: u32,
        position: u32,
    ) -> Result<Vec<f64>, String> {
        let visible = self.visible_range(tokens, position)?;
        let width = self.width as usize;
        let stride = self.kv_heads as usize * width;
        if q.len() != self.q_heads as usize * width
            || keys.len() != tokens as usize * stride
            || values.len() != keys.len()
            || q.iter().chain(keys).chain(values).any(|v| !v.is_finite())
        {
            return Err("Entrées attention invalides".into());
        }
        let mut output = vec![0.0; q.len()];
        for head in 0..self.q_heads as usize {
            let kv = head / (self.q_heads / self.kv_heads) as usize;
            let query = &q[head * width..(head + 1) * width];
            let scores: Vec<f64> = visible
                .clone()
                .map(|t| {
                    query
                        .iter()
                        .zip(&keys[t * stride + kv * width..t * stride + (kv + 1) * width])
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>()
                        * f64::from(self.scale)
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probabilities: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let denominator = probabilities.iter().sum::<f64>();
            for (t, p) in visible.clone().zip(probabilities) {
                for channel in 0..width {
                    output[head * width + channel] +=
                        p / denominator * f64::from(values[t * stride + kv * width + channel]);
                }
            }
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masks_future_tokens_and_maps_grouped_heads() {
        let c = AttentionConfig::new(4, 2, 1, None, 1.0).unwrap();
        let q = [0.0; 4];
        let k = [0.0; 6];
        let v = [1.0, 10.0, 3.0, 30.0, 999.0, 999.0];
        assert_eq!(
            c.reference(&q, &k, &v, 3, 1).unwrap(),
            vec![2.0, 2.0, 20.0, 20.0]
        );
        let c = AttentionConfig::new(4, 2, 1, Some(1), 1.0).unwrap();
        assert_eq!(
            c.reference(&q, &k, &v, 3, 1).unwrap(),
            vec![3.0, 3.0, 30.0, 30.0]
        );
    }
    #[test]
    fn softmax_is_stable_and_scale_is_applied() {
        let c = AttentionConfig::new(1, 1, 1, None, 1.0).unwrap();
        assert_eq!(
            c.reference(&[1000.0], &[1000.0, 1001.0], &[2.0, 3.0], 2, 1)
                .unwrap(),
            vec![3.0]
        );
        let c = AttentionConfig::new(1, 1, 1, None, 2.0).unwrap();
        let y = c.reference(&[1.0], &[0.0, 1.0], &[0.0, 1.0], 2, 1).unwrap()[0];
        assert!((y - 2.0_f64.exp() / (1.0 + 2.0_f64.exp())).abs() < 1e-12);
    }
    #[test]
    fn rejects_invalid_layouts_positions_and_values() {
        assert!(AttentionConfig::new(3, 2, 1, None, 1.0).is_err());
        assert!(AttentionConfig::new(1, 1, 1, Some(0), 1.0).is_err());
        let c = AttentionConfig::new(1, 1, 1, None, 1.0).unwrap();
        assert!(c.reference(&[1.0], &[1.0], &[1.0], 1, 1).is_err());
        assert!(c.reference(&[f32::NAN], &[1.0], &[1.0], 1, 0).is_err());
    }
}
