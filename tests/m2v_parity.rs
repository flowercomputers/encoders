//! Compares ese output against the model2vec-rs reference implementation.
//! Requires network access on first run (downloads the model via hf-hub).
//!
//! Inputs are encoded one at a time on the reference side: the reference
//! implementations pool [PAD] tokens into shorter sequences when batching
//! (see readme), so single-input encoding is the canonical behavior.
#![cfg(feature = "model-potion-code-16m-v2")]

use model2vec_rs::model::StaticModel;

/// Quantized builds compare lossy runtime output against the f32 reference.
fn threshold() -> f32 {
    if cfg!(feature = "quant-8") {
        0.9995
    } else if cfg!(feature = "quant-16") {
        0.99999
    } else {
        0.999999
    }
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    d / (na * nb)
}

#[test]
fn matches_model2vec_reference() {
    let m = StaticModel::from_pretrained("minishlab/potion-code-16M-v2", None, None, None)
        .expect("failed to load reference model");
    let inputs: Vec<String> = vec![
        "def sort_numbers(nums):\n    return sorted(nums)".into(),
        "function to sort a list of numbers".into(),
        "".into(),
        "   ".into(),
        "☃🦀 unknown pieces mixed with getParameterTypes()".into(),
        "SELECT * FROM users WHERE id = ?;".into(),
        "你好世界 mixed CJK and English".into(),
        "long input ".repeat(400),
        "!@#$%^&*()".into(),
        "café résumé naïve".into(),
        "impl Iterator for SmallCharIter { type Item = char; }".into(),
        "// TODO(sam): handle the empty-vocab edge case".into(),
    ];
    let ours = ese::encode(&inputs);
    let mut min_sim = f32::INFINITY;
    for (i, input) in inputs.iter().enumerate() {
        let theirs = &m.encode(&[input.clone()])[0];
        let s = cos(&ours[i], theirs);
        min_sim = min_sim.min(s);
        assert!(
            s > threshold(),
            "input {i} below threshold {}: cos={s}",
            threshold()
        );
    }
    println!("min cosine vs model2vec-rs: {min_sim:.8}");
}
