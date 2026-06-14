mod bpe;
mod lookup;
mod pretokenizer;
#[allow(dead_code, unused_imports)]
mod unigram;
mod wordpiece;

/// DIMENSIONS is the number of dimensions per encoding.
/// This can be changed with the `dim-*` crate features.
///
/// # Examples
///
/// ```
/// let encoding = ese::encode_single("hello world");
/// assert_eq!(encoding.len(), ese::DIMENSIONS);
/// ```
pub const DIMENSIONS: usize = lookup::DIMENSIONS;

/// MODEL_NAME is the static embedding model compiled into this build.
///
/// # Examples
///
/// ```
/// assert!(!ese::MODEL_NAME.is_empty());
/// ```
pub const MODEL_NAME: &str = lookup::MODEL_NAME;

/// IS_UNIGRAM is true when the compiled model uses the Unigram/SentencePiece tokenizer,
/// and false for WordPiece or BPE models. Unigram models are typically case-sensitive.
#[allow(dead_code)]
pub const IS_UNIGRAM: bool = lookup::IS_UNIGRAM;

/// IS_BPE is true when the compiled model uses the BPE (Byte-Pair Encoding) tokenizer
/// with the ByteLevel pre-tokenizer. This covers RoBERTa-family, GPT-2-derived, and
/// most LLM-derived static embedding models.
#[allow(dead_code)]
pub const IS_BPE: bool = lookup::IS_BPE;

/// Encodes a batch of texts into embedding vectors.
///
/// Each input string is independently normalized, tokenized, and encoded.
/// Enable the `rayon` feature for higher throughput on large batches (>16 inputs).
///
/// # Examples
///
/// ```
/// let texts = vec![
///     "potato".to_string(),
///     "root vegetable".to_string(),
/// ];
///
/// let embeddings = ese::encode(texts);
/// assert_eq!(embeddings.len(), 2);
/// assert_eq!(embeddings[0].len(), ese::DIMENSIONS);
/// ```
#[cfg(not(feature = "rayon"))]
pub fn encode(inputs: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<[f32; DIMENSIONS]> {
    inputs
        .into_iter()
        .map(|input| encode_single(input))
        .collect()
}

/// Encodes a batch of texts into embedding vectors.
///
/// Each input string is independently normalized, tokenized, and encoded.
/// Batches of 16 or more inputs are processed in parallel.
///
/// # Examples
///
/// ```
/// let texts = vec![
///     "potato".to_string(),
///     "root vegetable".to_string(),
/// ];
///
/// let embeddings = ese::encode(texts);
/// assert_eq!(embeddings.len(), 2);
/// assert_eq!(embeddings[0].len(), ese::DIMENSIONS);
/// ```
#[cfg(feature = "rayon")]
pub fn encode(
    inputs: impl IntoIterator<Item = impl AsRef<str> + Send + Sync>,
) -> Vec<[f32; DIMENSIONS]> {
    use rayon::prelude::*;

    let inputs: Vec<_> = inputs.into_iter().collect();
    let map_fn = |input: &_| encode_single(input);

    // 16 is an empirically chosen batch-size cutoff; below it sequential wins.
    if inputs.len() >= 16 {
        return inputs.par_iter().map(map_fn).collect();
    }

    inputs.iter().map(map_fn).collect()
}

/// Encodes a single text into an embedding vector.
///
/// For uncased models (the default WordPiece model), input is lowercased before
/// tokenization. For Unigram/SentencePiece models, case handling depends on the
/// normalizer compiled into the build.
/// Output is L2-normalized when the crate is built with `ESE_NORMALIZE_OUTPUT=1`.
/// When built with `ESE_SKIP_UNK=1`, UNK-id tokens are excluded from mean pooling,
/// matching the behavior of `model2vec.StaticModel`; use this for distilled model2vec artifacts.
///
/// Special tokens (`[CLS]`, `[SEP]`) are **not** pooled by default: the static-embedding
/// references (sentence-transformers `StaticEmbedding` and model2vec) do not include them
/// in the mean. Set `ESE_ADD_SPECIAL_TOKENS=1` at build time to opt in (rarely needed).
///
/// # Examples
///
/// ```
/// // Only holds for uncased (WordPiece) models; Unigram and BPE models are case-sensitive.
/// if !ese::IS_UNIGRAM && !ese::IS_BPE {
///     let a = ese::encode_single("rustacean");
///     let b = ese::encode_single("Rustacean");
///     assert_eq!(a, b, "encoding is case-insensitive for the default (uncased) model");
/// }
/// ```
///
/// ```
/// let v = ese::encode_single("hello world");
/// assert_eq!(v.len(), ese::DIMENSIONS);
/// ```
#[inline]
pub fn encode_single(input: impl AsRef<str>) -> [f32; DIMENSIONS] {
    let text = input.as_ref();

    // Const dispatch: the compiler eliminates dead arms at compile time.
    // IS_BPE=true    → ByteLevel BPE path (src/bpe.rs).
    // IS_UNIGRAM=true → Unigram/Viterbi path (src/unigram.rs).
    // otherwise      → WordPiece path (unchanged from before).
    if lookup::IS_BPE {
        return bpe::encode_single(text);
    }
    if lookup::IS_UNIGRAM {
        return unigram::encode_single(text);
    }

    // ── WordPiece path (byte-for-byte identical to the original) ─────────────
    let mut vector = [0.0f32; DIMENSIONS];
    if lookup::ADD_SPECIAL_TOKENS {
        // CLS special token: use its per-row params.
        // quant builds: CLS_MIN / CLS_SCALE from model_constants.rs.
        // f32 build:    CLS_MIN=0.0, CLS_SCALE=1.0 → identity (same as before).
        wordpiece::accumulate(&mut vector, &lookup::CLS, lookup::CLS_MIN, lookup::CLS_SCALE);
    }
    // Pre-count CLS + SEP; the SEP embedding is added after the loop.
    let mut token_count: u32 = if lookup::ADD_SPECIAL_TOKENS { 2 } else { 0 };
    let mut normalized = String::with_capacity(text.len());
    pretokenizer::normalize_into(text, &mut normalized);
    let mut wp_buf = String::with_capacity(64);
    pretokenizer::for_each_pre_token(&normalized, |token| {
        wordpiece::wordpiece_accumulate(token, &mut vector, &mut token_count, &mut wp_buf);
    });
    if lookup::ADD_SPECIAL_TOKENS {
        let scale = 1.0 / token_count as f32;
        let mut sep = [0.0f32; DIMENSIONS];
        // SEP special token: use its per-row params.
        wordpiece::accumulate(&mut sep, &lookup::SEP, lookup::SEP_MIN, lookup::SEP_SCALE);
        for (v, s) in vector.iter_mut().zip(sep.iter()) {
            *v = (*v + *s) * scale;
        }
    } else {
        let scale = if token_count != 0 { 1.0 / token_count as f32 } else { 1.0 };
        for v in &mut vector {
            *v *= scale;
        }
    }
    if lookup::NORMALIZE_OUTPUT {
        normalize(&mut vector);
    }
    vector
}

#[inline]
fn normalize(vector: &mut [f32; DIMENSIONS]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vector {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod test;
