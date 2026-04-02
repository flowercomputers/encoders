use std::hint::black_box;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn dimensions() -> usize {
    ese_core::DIMENSIONS
}

#[wasm_bindgen]
pub fn encode(text: &str) -> Box<[f32]> {
    Vec::from(ese_core::encode_single(text)).into_boxed_slice()
}

#[wasm_bindgen]
pub fn encode_repeated(text: &str, iterations: u32) -> f32 {
    let iterations = iterations.max(1) as usize;
    let mut checksum = 0.0f32;

    for i in 0..iterations {
        let embedding = ese_core::encode_single(black_box(text));
        checksum += black_box(embedding[i % ese_core::DIMENSIONS]);
    }

    black_box(checksum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_exports_match_core_dimensions() {
        assert_eq!(dimensions(), ese_core::DIMENSIONS);
    }

    #[test]
    fn wasm_encode_returns_full_vector() {
        let embedding = encode("browser wasm smoke test");
        assert_eq!(embedding.len(), ese_core::DIMENSIONS);
    }

    #[test]
    fn repeated_encode_is_deterministic() {
        let left = encode_repeated("deterministic benchmark", 32);
        let right = encode_repeated("deterministic benchmark", 32);
        assert_eq!(left, right);
    }
}
