#[test]
fn test_fuzz() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let seed_strings: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![0xFF],
        vec![0, 0, 0, 0],
        vec![0xFF; 256],
        b"\x00\x01\x02\x03".to_vec(),
        b"\xc0\xaf".to_vec(),         // overlong UTF-8
        b"\xed\xa0\x80".to_vec(),     // surrogate half
        b"\xf4\x90\x80\x80".to_vec(), // above U+10FFFF
        "hello".as_bytes().to_vec(),
        " ".repeat(1000).into_bytes(),
        "\u{FFFD}\u{FFFD}".as_bytes().to_vec(),
        "\0\0\0".as_bytes().to_vec(),
        "##".as_bytes().to_vec(),
        "a\u{0300}\u{0301}b".as_bytes().to_vec(), // combining marks
        "\u{4E00}\u{9FFF}".as_bytes().to_vec(),   // CJK
        "!@#$%^&*()".as_bytes().to_vec(),
    ];

    // Test raw byte sequences that happen to be valid UTF-8
    for seed in &seed_strings {
        if let Ok(s) = std::str::from_utf8(seed) {
            let _ = crate::encode_single(s);
        }
    }

    // Pseudo-random generation from seeds
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

    // Also fuzz with valid unicode strings built from char ranges
    for cp in (0u32..0x10000).step_by(37) {
        if let Some(ch) = char::from_u32(cp) {
            let s: String = std::iter::repeat(ch).take(5).collect();
            let _ = crate::encode_single(&s);
        }
    }
}

#[cfg(feature = "tests")]
mod optional {
    include!(concat!(env!("OUT_DIR"), "/optional_testdata.rs"));

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[test]
    fn test_embeddings_match() {
        // The reference outputs (TEST_OUTPUTS) are computed at build time from the same
        // (possibly quantized) weight table the artifact contains, by applying the same
        // quantize→dequantize round-trip as the runtime. This test verifies that the
        // runtime faithfully reproduces the compiled model — it does NOT measure
        // quantization quality vs f32 (tracked separately).
        //
        // Quantized builds use a slightly relaxed threshold (0.9999) to tolerate
        // floating-point accumulation-order noise introduced when the oracle and the
        // runtime accumulate many dequantized values; observed minimums are ≥ 0.9999977
        // (quant-8) and ≥ 0.9999986 (quant-16). The strict 0.999999 threshold applies
        // to f32 builds, where no such noise exists.
        let threshold = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
            0.9999_f32
        } else {
            0.999999_f32
        };
        let embeddings = crate::encode(TEST_INPUTS);
        for (i, (want, got)) in TEST_OUTPUTS.iter().zip(embeddings.iter()).enumerate() {
            // When the oracle produces an all-zero vector (e.g. empty input with
        // ADD_SPECIAL_TOKENS=false), cosine similarity is undefined (NaN).
        // In that case verify that the runtime also produces all-zeros.
        let want_norm: f32 = want.iter().map(|x| x * x).sum::<f32>().sqrt();
        let got_norm: f32 = got.iter().map(|x| x * x).sum::<f32>().sqrt();
        if want_norm < f32::EPSILON {
            assert!(
                got_norm < f32::EPSILON,
                "oracle produced zero vector for {i} but runtime produced non-zero (norm={got_norm})"
            );
            continue;
        }
        let sim = cosine_similarity(want, got);
            assert!(
                sim > threshold,
                "cosine similarity for {i} below threshold: {sim}"
            );
            #[cfg(not(any(feature = "quant-8", feature = "quant-16")))]
            {
                let rel = |w: &[f32], g: &[f32]| -> f32 {
                    let d: f32 = w.iter().zip(g).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                    let n: f32 = w.iter().map(|a| a * a).sum::<f32>().sqrt();
                    d / n.max(f32::EPSILON)
                };
                assert!(rel(want, got) < 1e-4, "relative L2 error for {i} too high");
            }
        }
    }
}
