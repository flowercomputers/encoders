//! Compiled text embedding models.
//!
//! One static embedding model, chosen by cargo feature, is fetched at build
//! time, quantized, and embedded as lookup tables. Encoding a text is a
//! tokenizer pass and a mean over table rows: a few hundred nanoseconds,
//! with no model files, no startup, no runtime dependencies.
//!
//! ```
//! let query = encoders::encode_single("What is the capital of France?");
//! let docs = encoders::encode(["Paris is the capital of France.", "Tokyo is the capital of Japan."]);
//! assert_eq!(query.len(), encoders::DIMENSIONS);
//! assert!(encoders::similarity(&query, &docs[0]) > encoders::similarity(&query, &docs[1]));
//! ```
//!
//! # Features
//!
//! - One model feature, named after the model (`bekko-static-v1-slim-32k`,
//!   `potion-base-8m`, ...). With none, the current default is
//!   `bekko-static-v1-slim-100k` truncated to 512 dimensions; the default may
//!   change between major versions, so name a model if you store embeddings.
//! - `mrl-1024` ... `mrl-32`: truncate a Matryoshka-trained model to a width
//!   it was trained at. Omit for the model's native width (or 512 for the
//!   unnamed default).
//! - `int16` or `float32`: how the table is stored. With neither, it is
//!   `int8`.
//! - `rayon`: [`encode`] goes parallel from [`PARALLEL_MIN`] texts.
//!
//! [`encode_single`], [`encode`] and [`encode_into`] produce embeddings;
//! [`similarity`] (cosine) and [`distance`] (Euclidean) compare them;
//! [`DIMENSIONS`] gives you the size of output vectors per input.

ese::include_model!();

#[cfg(test)]
mod test;
