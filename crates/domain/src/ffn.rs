//! FFN dense Gemma : RMSNorm, GELU(tanh)(gate) * up, down, norme et résiduel.
use crate::{Q4Matrix, RmsNorm};
#[derive(Debug)]
pub struct Ffn {
    pre: RmsNorm,
    gate: Q4Matrix,
    up: Q4Matrix,
    down: Q4Matrix,
    post: RmsNorm,
    layer_scale: f32,
}
impl Ffn {
    pub fn new(
        pre: RmsNorm,
        gate: Q4Matrix,
        up: Q4Matrix,
        down: Q4Matrix,
        post: RmsNorm,
        layer_scale: f32,
    ) -> Result<Self, String> {
        let hidden = gate.columns();
        let intermediate = gate.rows();
        if up.columns() != hidden
            || up.rows() != intermediate
            || down.columns() != intermediate
            || down.rows() != hidden
            || pre.weights().len() != hidden as usize
            || post.weights().len() != hidden as usize
            || !layer_scale.is_finite()
        {
            return Err("Dimensions ou échelle FFN incompatibles".into());
        }
        Ok(Self {
            pre,
            gate,
            up,
            down,
            post,
            layer_scale,
        })
    }
    pub fn pre(&self) -> &RmsNorm {
        &self.pre
    }
    pub fn post(&self) -> &RmsNorm {
        &self.post
    }
    pub fn matrices(&self) -> [&Q4Matrix; 3] {
        [&self.gate, &self.up, &self.down]
    }
    pub fn width(&self) -> u32 {
        self.gate.columns()
    }
    pub fn intermediate(&self) -> u32 {
        self.gate.rows()
    }
    pub fn layer_scale(&self) -> f32 {
        self.layer_scale
    }
    pub fn reference(&self, input: &[f32]) -> Result<Vec<f64>, String> {
        if input.len() != self.width() as usize {
            return Err("Taille entrée FFN invalide".into());
        }
        let normalized: Vec<f32> = self
            .pre
            .reference(input)?
            .iter()
            .map(|v| *v as f32)
            .collect();
        let gate = self.gate.reference_matvec(&normalized)?;
        let up = self.up.reference_matvec(&normalized)?;
        let active: Vec<f32> = gate
            .iter()
            .zip(up)
            .map(|(&g, u)| {
                let g = f64::from(g as f32);
                let gelu = gelu_tanh(g) as f32;
                gelu * (u as f32)
            })
            .collect();
        let down: Vec<f32> = self
            .down
            .reference_matvec(&active)?
            .iter()
            .map(|v| *v as f32)
            .collect();
        let post = self.post.reference(&down)?;
        Ok(post
            .iter()
            .zip(input)
            .map(|(&p, &x)| f64::from(((p as f32) + x) * self.layer_scale))
            .collect())
    }
}
fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn activation_has_known_values_and_is_not_silu() {
        assert_eq!(gelu_tanh(0.0), 0.0);
        assert!((gelu_tanh(1.0) - 0.8411919906082768).abs() < 1e-12);
        assert!((gelu_tanh(-1.0) + 0.15880800939172324).abs() < 1e-12);
    }
    #[test]
    fn zero_branch_preserves_residual_and_applies_final_scale() {
        let f = Ffn::new(
            RmsNorm::new(vec![1.0; 32], 1e-6).unwrap(),
            Q4Matrix::new(32, 32, vec![0; 576]).unwrap(),
            Q4Matrix::new(32, 32, vec![0; 576]).unwrap(),
            Q4Matrix::new(32, 32, vec![0; 576]).unwrap(),
            RmsNorm::new(vec![1.0; 32], 1e-6).unwrap(),
            0.5,
        )
        .unwrap();
        assert_eq!(f.reference(&[2.0; 32]).unwrap(), vec![1.0; 32]);
        assert!(f.reference(&[0.0]).is_err());
    }
}
