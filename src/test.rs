#[test]
fn fuzz_never_panics() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let seeds: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![0xFF],
        vec![0, 0, 0, 0],
        vec![0xFF; 256],
        b"\x00\x01\x02\x03".to_vec(),
        b"\xc0\xaf".to_vec(),
        b"\xed\xa0\x80".to_vec(),
        b"\xf4\x90\x80\x80".to_vec(),
        b"hello".to_vec(),
        " ".repeat(1000).into_bytes(),
        "\u{FFFD}\u{FFFD}".as_bytes().to_vec(),
        "\0\0\0".as_bytes().to_vec(),
        "##".as_bytes().to_vec(),
        "a\u{0300}\u{0301}b".as_bytes().to_vec(),
        "\u{4E00}\u{9FFF}".as_bytes().to_vec(),
        "!@#$%^&*()".as_bytes().to_vec(),
        "x".repeat(200).into_bytes(),
    ];
    for seed in &seeds {
        if let Ok(s) = std::str::from_utf8(seed) {
            let _ = crate::encode_single(s);
        }
    }

    for i in 0u64..10_000 {
        let mut h = DefaultHasher::new();
        i.hash(&mut h);
        let mut state = h.finish();
        let len = (state % 512) as usize;
        let mut bytes = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.push((state & 0xFF) as u8);
        }
        if let Ok(s) = String::from_utf8(bytes) {
            let _ = crate::encode_single(&s);
        }
    }

    for cp in (0u32..0x10000).step_by(37) {
        if let Some(ch) = char::from_u32(cp) {
            let s: String = std::iter::repeat_n(ch, 5).collect();
            let _ = crate::encode_single(&s);
        }
    }
}

#[test]
fn known_token_resolves_and_unknown_does_not() {
    assert!(crate::model::lookup("hello").is_some());
    assert!(crate::model::lookup(crate::model::UNK_TOKEN).is_some());
    assert!(
        crate::model::TRUNCATION_POINTS
            .iter()
            .all(|&d| d <= crate::model::NATIVE_DIMENSIONS)
    );
    const _: () = assert!(crate::DIMENSIONS <= crate::model::NATIVE_DIMENSIONS);
    assert!(crate::model::lookup("definitely-not-a-token-\u{1F980}").is_none());
}

#[test]
fn metadata_and_comparison_helpers() {
    assert!(crate::model::LICENSE.is_some());
    let _: bool = crate::model::MULTILINGUAL;
    assert_eq!(crate::PARALLEL_MIN, 128);
    let a = crate::encode_single("the cat sat on the mat");
    let b = crate::encode_single("a cat is sitting on a mat");
    let c = crate::encode_single("quarterly revenue guidance");
    assert!((crate::similarity(&a, &a) - 1.0).abs() < 1e-4);
    assert!((crate::similarity(&a, &b) - crate::similarity(&b, &a)).abs() < 1e-6);
    assert!(crate::similarity(&a, &b) > crate::similarity(&a, &c));
    assert!(crate::distance(&a, &b) < crate::distance(&a, &c));
    assert!(
        (crate::compare("the cat sat on the mat", "the cat sat on the mat") - 1.0).abs() < 1e-4
    );
    let s = crate::compare_each(
        "the cat sat on the mat",
        ["a cat is sitting on a mat", "quarterly revenue guidance"],
    );
    assert_eq!(s.len(), 2);
    assert!(s[0] > s[1]);
    assert!((s[0] - crate::similarity(&a, &b)).abs() < 1e-6);
}

#[test]
fn empty_input_is_zero_vector() {
    assert!(crate::encode_single("").iter().all(|&v| v == 0.0));
    assert!(crate::encode_single("   \t\n").iter().all(|&v| v == 0.0));
}

#[test]
fn normalization_policy_is_applied() {
    let v = crate::encode_single("hello world");
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if crate::model::NORMALIZE {
        assert!((norm - 1.0).abs() < 1e-4, "norm = {norm}");
    } else {
        assert!(norm > 0.0);
    }
}

/// Compares against reference embeddings computed at build time with the
/// `tokenizers` crate (`cargo test --features tests`).
#[cfg(feature = "tests")]
mod reference {
    include!(concat!(env!("OUT_DIR"), "/optional_testdata.rs"));

    fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 && nb == 0.0 {
            return None;
        }
        Some(dot / (na * nb))
    }

    #[test]
    fn embeddings_match_tokenizers_reference() {
        let got = crate::encode(TEST_INPUTS);
        assert_eq!(got.len(), TEST_OUTPUTS.len());
        for (i, (want, got)) in TEST_OUTPUTS.iter().zip(&got).enumerate() {
            match cosine(want, got) {
                None => {
                    // Reference is the zero vector: so must we be.
                    assert!(
                        got.iter().all(|&v| v == 0.0),
                        "input {i} {:?}: expected zero vector",
                        TEST_INPUTS[i]
                    );
                }
                Some(sim) => {
                    // A mean within a few quantization-noise norms (an
                    // all-unknown text on a model with a tiny UNK row) has no
                    // meaningful direction even in f32; skip it.
                    if TEST_REF_NORMS[i] <= 4.0 * TEST_REF_NOISE[i] {
                        eprintln!(
                            "skipping input {i} {:?}: reference norm {} within 4 x quantization noise {}",
                            TEST_INPUTS[i], TEST_REF_NORMS[i], TEST_REF_NOISE[i]
                        );
                        continue;
                    }
                    assert!(
                        sim > TEST_COSINE_MIN,
                        "input {i} {:?}: cosine {sim} below {TEST_COSINE_MIN}",
                        TEST_INPUTS[i]
                    );
                }
            }
            if crate::model::NORMALIZE && !<crate::model::Elem as ese::Elem>::QUANTIZED {
                for (j, (w, g)) in want.iter().zip(got).enumerate() {
                    assert!(
                        (w - g).abs() < 1e-5,
                        "input {i} {:?}: dim {j} want {w} got {g}",
                        TEST_INPUTS[i]
                    );
                }
            }
        }
    }
}
