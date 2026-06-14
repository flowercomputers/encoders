//! WebAssembly bindings for ese (extremely static embeddings).
//!
//! Exposes `encode_single`, `encode` (returns flat Float32Array), `dimensions`, and
//! `model_name` via wasm-bindgen. The compiled model weights are baked in at build time —
//! no model download at runtime, no server required.
//!
//! Build:
//!   wasm-pack build api-wasm --target web          # browser (ES module)
//!   wasm-pack build api-wasm --target nodejs       # Node.js crosscheck
//!
//! rayon is intentionally NOT enabled — wasm32 is single-threaded.

use wasm_bindgen::prelude::*;

/// Return the number of dimensions per embedding vector.
#[wasm_bindgen]
pub fn dimensions() -> usize {
    ese_core::DIMENSIONS
}

/// Return the model name baked into this build.
#[wasm_bindgen]
pub fn model_name() -> String {
    ese_core::MODEL_NAME.to_owned()
}

/// Encode a single text and return its embedding as a Float32Array.
///
/// ```js
/// import init, { encode_single } from './pkg/ese_wasm.js';
/// await init();
/// const vec = encode_single("hello world");  // Float32Array of length 512
/// ```
#[wasm_bindgen]
pub fn encode_single(text: &str) -> Vec<f32> {
    ese_core::encode_single(text).to_vec()
}

/// Encode a batch of texts and return embeddings as a flat Float32Array.
///
/// The returned array has `texts.length * dimensions()` elements.
/// Row `i` occupies indices `[i * dimensions(), (i+1) * dimensions())`.
///
/// ```js
/// import init, { encode, dimensions } from './pkg/ese_wasm.js';
/// await init();
/// const dim = dimensions();
/// const flat = encode(["hello", "world"]);   // Float32Array length = 2 * 512
/// const v0 = flat.slice(0, dim);
/// const v1 = flat.slice(dim, 2 * dim);
/// ```
#[wasm_bindgen]
pub fn encode(texts: Vec<String>) -> Vec<f32> {
    let embeddings = ese_core::encode(texts);
    let mut flat = Vec::with_capacity(embeddings.len() * ese_core::DIMENSIONS);
    for emb in embeddings {
        flat.extend_from_slice(&emb);
    }
    flat
}
