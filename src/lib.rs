//! Text embeddings compiled into the binary.
//!
//! One static embedding model, chosen by cargo feature, is fetched at build
//! time, quantized, and embedded as lookup tables. Encoding a text is a
//! tokenizer pass and a mean over table rows: a few hundred nanoseconds,
//! no model files, no startup, no runtime dependencies.
//!
//! ```
//! let query = encoders::encode_single("What is the capital of France?");
//! let docs = encoders::encode(["Paris is the capital of France.", "Tokyo is the capital of Japan."]);
//! assert_eq!(query.len(), encoders::DIMENSIONS);
//! assert!(encoders::similarity(&query, &docs[0]) > encoders::similarity(&query, &docs[1]));
//!
//! // The tokenizer lowercases, so these agree.
//! assert_eq!(encoders::encode_single("rustacean"), encoders::encode_single("Rustacean"));
//!
//! // Some models return unit vectors; `model::NORMALIZE` says which.
//! let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
//! if encoders::model::NORMALIZE {
//!     assert!((norm - 1.0).abs() < 1e-3);
//! } else {
//!     assert!(norm > 0.0);
//! }
//! ```
//!
//! # Features
//!
//! - One model feature, named after the model (`bekko-static-v1-slim-32k`,
//!   `potion-base-8m`, ...). With none, the current default is
//!   `bekko-static-v1-slim-75k`; the default may change between major
//!   versions, so name a model if you store embeddings.
//! - `mrl-1024` ... `mrl-32`: truncate a Matryoshka-trained model to a width
//!   it was trained at. Omit for the native width.
//! - `int16` (default), `int8` or `float32`: how the table is stored. An
//!   explicit `int8` or `float32` overrides the default.
//! - `rayon`: [`encode`] goes parallel from [`PARALLEL_MIN`] texts.
//!
//! The crate's readme has the catalog with notes on each family.
//!
//! # What is exported
//!
//! [`encode_single`], [`encode`] and [`encode_into`] produce embeddings;
//! [`similarity`] (cosine) and [`distance`] (Euclidean) compare them;
//! [`DIMENSIONS`] and [`PARALLEL_MIN`] are the width and the parallel
//! threshold. The [`model`] module holds everything known about the compiled
//! model: `MODEL_ID` and `MODEL_REVISION`, `LICENSE` (SPDX) and
//! `MULTILINGUAL`, `NATIVE_DIMENSIONS`, `TRUNCATION_POINTS` and `MATRYOSHKA`,
//! `NORMALIZE`, `VOCAB_SIZE`, and a `lookup` into the vocabulary.
//!
//! # Trade-offs
//!
//! Static embeddings have no attention: a text is the mean of its tokens'
//! fixed vectors, so word order and context are invisible and quality sits
//! below transformer encoders. The table lives in your binary, from a few MB
//! to a few hundred, and the first build downloads the model. Each model has
//! its own license, exposed as `model::LICENSE`.

ese::include_model!();

#[cfg(test)]
mod test;
