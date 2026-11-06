//! Dot product, norm, cosine similarity and Euclidean distance over
//! embeddings. Each pass runs eight independent accumulators (a serial float
//! chain costs several times more on wide vectors); dot, norm and distance
//! are one pass each, cosine is three (a dot and two norms).

#[inline]
fn lanes(a: &[f32], b: &[f32], mut f: impl FnMut(f32, f32) -> f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..8 {
            acc[i] += f(x[i], y[i]);
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += f(*x, *y);
    }
    sum
}

/// `a · b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    lanes(a, b, |x, y| x * y)
}

/// `‖a‖₂`.
#[inline]
pub fn norm(a: &[f32]) -> f32 {
    lanes(a, a, |x, y| x * y).sqrt()
}

/// Cosine similarity, 0 if either vector is zero.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = norm(a) * norm(b);
    if d > 0.0 { dot(a, b) / d } else { 0.0 }
}

/// `‖a − b‖₂`.
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    lanes(a, b, |x, y| (x - y) * (x - y)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..19).map(|i| (19 - i) as f32).collect();
        let naive_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive_dot).abs() < 1e-3);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0; 4], &[1.0; 4]), 0.0);
        assert!((euclidean(&[0.0, 3.0], &[4.0, 0.0]) - 5.0).abs() < 1e-6);
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }
}
