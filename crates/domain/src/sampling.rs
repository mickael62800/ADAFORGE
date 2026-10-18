//! Sélection déterministe : score maximal, puis plus petit ID en cas d’égalité.
pub fn greedy(logits: &[f32]) -> Result<u32, String> {
    if logits.is_empty()
        || logits.len() > u32::MAX as usize
        || logits.iter().any(|v| !v.is_finite())
    {
        return Err("Logits vides ou non finis".into());
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maximum_ties_and_invalid_scores() {
        assert_eq!(greedy(&[-3.0, 2.0, 2.0, 1.0]).unwrap(), 1);
        assert_eq!(greedy(&[-3.0, -1.0]).unwrap(), 1);
        assert!(greedy(&[]).is_err());
        assert!(greedy(&[0.0, f32::NAN]).is_err());
    }
}
