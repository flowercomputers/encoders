//! A static embedding model compiled into the binary.
//!
//! Which model is chosen with a cargo feature named after it (default:
//! `flower-computer/bekko-static-v1-slim-75k` at its native 1024 dimensions,
//! stored as 16-bit integers). The model is fetched and turned into lookup
//! tables by `build.rs` using `ese-build`; this file only asks `ese` to
//! include the result. See the crate's `Cargo.toml` for the catalog, the
//! `mrl-*` widths and the `float32` / `int16` / `int8` storage features.
//!
//! ```
//! let v = encoders::encode_single("hello world");
//! assert_eq!(v.len(), encoders::DIMENSIONS);
//!
//! // The tokenizer lowercases, so these agree.
//! assert_eq!(encoders::encode_single("rustacean"), encoders::encode_single("Rustacean"));
//!
//! let batch = encoders::encode(["potato", "root vegetable"]);
//! assert_eq!(batch.len(), 2);
//!
//! // Models that ship a Normalize module (the potion family) return unit vectors.
//! let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
//! if encoders::model::NORMALIZE {
//!     assert!((norm - 1.0).abs() < 1e-3);
//! } else {
//!     assert!(norm > 0.0);
//! }
//! ```

ese::include_model!();

#[cfg(test)]
mod test;
