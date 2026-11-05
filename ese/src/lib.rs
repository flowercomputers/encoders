//! Extremely Static Embeddings: the model-agnostic runtime.
//!
//! This crate holds the tokenizer, pooling and perfect-hash lookup code. It
//! contains no model. A model is compiled in by a crate that
//!
//! 1. calls [`ese_build::Builder::generate`](https://github.com/flowercomputers/ese)
//!    from its `build.rs`, and
//! 2. invokes [`include_model!`] at the top level of its `lib.rs`.
//!
//! The `encoders` crate at the root of this workspace does exactly that for
//! a catalog of models selected by cargo feature, and is the one to depend
//! on if you just want embeddings. Copy its `build.rs` to compile in a
//! different model.
//!
//! The runtime reproduces the `BertNormalizer` + `BertPreTokenizer` +
//! `WordPiece` pipeline of the `tokenizers` crate, including its literal
//! matching of special added tokens typed into the input, followed by mean
//! pooling with `add_special_tokens=false`, which is what both
//! sentence-transformers' `StaticEmbedding` and model2vec do.

use core::mem::MaybeUninit;

mod elem;

pub mod accum;
pub mod phf;
pub mod pretokenizer;
pub mod wordpiece;

pub use elem::Elem;

/// The `BertNormalizer` / `WordPiece` parameters read from `tokenizer.json`.
#[derive(Clone, Copy, Debug)]
pub struct TokenizerConfig {
    pub lowercase: bool,
    pub strip_accents: bool,
    pub handle_chinese_chars: bool,
    pub clean_text: bool,
    pub subword_prefix: &'static str,
    pub max_word_len: usize,
    /// Length in bytes of the longest vocabulary entry (without the
    /// subword prefix). WordPiece never tries a longer piece.
    pub max_token_bytes: usize,
    /// Special added tokens (`[CLS]`, `[SEP]`, ...). Like the `tokenizers`
    /// crate, these are matched literally on the raw input, before
    /// normalization, and looked up as single tokens.
    pub special_tokens: &'static [&'static str],
    /// The unknown token's text; a literal occurrence follows the UNK policy.
    pub unk_token: &'static str,
    /// Byte classes for the pre-tokenizer's ASCII fast path; must be built
    /// from the same flags and special tokens as the fields above.
    pub ascii: &'static pretokenizer::AsciiTable,
}

/// Everything `encode_single` needs besides the lookup table itself.
#[derive(Debug)]
pub struct ModelParams<E: 'static, const D: usize> {
    pub tokenizer: TokenizerConfig,
    /// model2vec semantics: unknown tokens are dropped before pooling.
    /// sentence-transformers semantics: the `[UNK]` row is pooled like any other.
    pub drop_unk: bool,
    /// L2-normalize the pooled vector.
    pub normalize: bool,
    /// Truncate to this many tokens (`tokenizer.json`'s `truncation`), counting
    /// unknown tokens even when they are dropped; 0 for none.
    pub max_tokens: usize,
    /// Per-column dequantization: `value = quant_min[i] + stored * quant_scale[i]`.
    /// Ignored for f32 rows.
    pub quant_min: &'static [f32; D],
    pub quant_scale: &'static [f32; D],
    /// Stored row for the unknown token and its row exponent.
    pub unk: &'static [E; D],
    pub unk_exp: u8,
}

/// Encode one text into `out`. `lookup(hash)` resolves a vocabulary key's
/// hash (see [`phf`]) to its stored row and row exponent.
#[inline]
pub fn encode_into<E, const D: usize, L>(
    params: &ModelParams<E, D>,
    lookup: &L,
    text: &str,
    out: &mut [f32; D],
) where
    E: Elem,
    L: Fn(u64) -> Option<(&'static [E; D], u8)>,
{
    // SAFETY: same layout; only initialized values are written.
    let slot = unsafe { &mut *(out as *mut [f32; D] as *mut MaybeUninit<[f32; D]>) };
    encode_uninit(params, lookup, text, slot);
}

/// [`encode_into`] for uninitialized output storage. Every lane is written.
#[inline]
pub fn encode_uninit<E, const D: usize, L>(
    params: &ModelParams<E, D>,
    lookup: &L,
    text: &str,
    out: &mut MaybeUninit<[f32; D]>,
) where
    E: Elem,
    L: Fn(u64) -> Option<(&'static [E; D], u8)>,
{
    let mut slot = MaybeUninit::uninit();
    let pool = accum::Accumulator::<E, D>::init(&mut slot);
    pool.set_limit(params.max_tokens);
    let unk = if params.drop_unk {
        None
    } else {
        Some((params.unk, params.unk_exp))
    };

    pretokenizer::for_each_word(&params.tokenizer, text, |word, kind, hashes| {
        if !pool.full() {
            wordpiece::wordpiece_accumulate(&params.tokenizer, word, kind, hashes, pool, lookup, unk);
        }
    });

    pool.finish_into(out, params.quant_min, params.quant_scale, params.normalize);
}

/// Encode one text. See [`encode_into`].
#[inline]
pub fn encode_single<E, const D: usize, L>(
    params: &ModelParams<E, D>,
    lookup: &L,
    text: &str,
) -> [f32; D]
where
    E: Elem,
    L: Fn(u64) -> Option<(&'static [E; D], u8)>,
{
    let mut out = MaybeUninit::uninit();
    encode_uninit(params, lookup, text, &mut out);
    // SAFETY: `encode_uninit` writes every lane.
    unsafe { out.assume_init() }
}

/// Batches of at least this many texts are encoded in parallel when the
/// `rayon` feature is on.
pub const PARALLEL_MIN: usize = 128;

/// Uninitialized storage for `n` embeddings, and the final cast once every
/// slot has been written.
fn uninit_vec<const D: usize>(n: usize) -> Vec<MaybeUninit<[f32; D]>> {
    let mut v = Vec::with_capacity(n);
    // SAFETY: `MaybeUninit` needs no initialization and capacity is `n`.
    unsafe { v.set_len(n) };
    v
}

/// # Safety
/// Every element of `v` must have been written.
unsafe fn assume_init_vec<const D: usize>(v: Vec<MaybeUninit<[f32; D]>>) -> Vec<[f32; D]> {
    let mut v = core::mem::ManuallyDrop::new(v);
    // SAFETY: same layout and allocation; the caller guarantees initialization.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut [f32; D], v.len(), v.capacity()) }
}

/// Encode a batch of texts, in parallel when the `rayon` feature is on and
/// the batch has at least [`PARALLEL_MIN`] items. Each embedding is written
/// once, straight into the returned vector.
#[cfg(feature = "rayon")]
pub fn encode<E, const D: usize, L, I>(
    params: &ModelParams<E, D>,
    lookup: &L,
    inputs: I,
) -> Vec<[f32; D]>
where
    E: Elem,
    L: Fn(u64) -> Option<(&'static [E; D], u8)> + Sync,
    I: IntoIterator,
    I::Item: AsRef<str> + Send + Sync,
{
    use rayon::prelude::*;
    let inputs: Vec<I::Item> = inputs.into_iter().collect();
    let mut out = uninit_vec::<D>(inputs.len());
    if inputs.len() >= PARALLEL_MIN {
        out.par_iter_mut()
            .zip(inputs.par_iter())
            .with_min_len(16)
            .for_each(|(o, s)| encode_uninit(params, lookup, s.as_ref(), o));
    } else {
        for (o, s) in out.iter_mut().zip(&inputs) {
            encode_uninit(params, lookup, s.as_ref(), o);
        }
    }
    // SAFETY: every slot was written by `encode_uninit`.
    unsafe { assume_init_vec(out) }
}

/// Encode a batch of texts. Each embedding is written once, straight into
/// the returned vector.
#[cfg(not(feature = "rayon"))]
pub fn encode<E, const D: usize, L, I>(
    params: &ModelParams<E, D>,
    lookup: &L,
    inputs: I,
) -> Vec<[f32; D]>
where
    E: Elem,
    L: Fn(u64) -> Option<(&'static [E; D], u8)>,
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let inputs: Vec<I::Item> = inputs.into_iter().collect();
    let mut out = uninit_vec::<D>(inputs.len());
    for (o, s) in out.iter_mut().zip(&inputs) {
        encode_uninit(params, lookup, s.as_ref(), o);
    }
    // SAFETY: every slot was written by `encode_uninit`.
    unsafe { assume_init_vec(out) }
}

/// Compile the model generated into this crate's `OUT_DIR` by `ese-build`.
///
/// Expands, at the invocation site, to:
///
/// - `pub mod model` with the generated constants (including
///   `NATIVE_DIMENSIONS`, `TRUNCATION_POINTS` and `MATRYOSHKA`), the embedded
///   tables, `lookup(&str)` / `lookup_hash(u64)` and `PARAMS`;
/// - `pub const DIMENSIONS: usize`;
/// - `pub fn encode_single(impl AsRef<str>) -> [f32; DIMENSIONS]`;
/// - `pub fn encode_into(impl AsRef<str>, &mut [f32; DIMENSIONS])`;
/// - `pub fn encode(impl IntoIterator<Item: AsRef<str> + Send + Sync>) -> Vec<[f32; DIMENSIONS]>`.
#[macro_export]
macro_rules! include_model {
    () => {
        /// Generated model tables and parameters (see `ese-build`).
        pub mod model {
            include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

            const _: () = assert!(
                MAX_WORD_LEN * 4 <= $crate::pretokenizer::WORD_BUF,
                "max_input_chars_per_word is too large for the stack word buffer"
            );
            const _: () = assert!(
                MAX_WORD_LEN <= $crate::wordpiece::MAX_PIECES,
                "max_input_chars_per_word is too large for the stack piece buffer"
            );

            /// One stored row, padded to whole cache lines.
            #[repr(C, align(64))]
            pub struct Slot {
                pub weight: Param,
            }
            const _: () = assert!(
                ::core::mem::size_of::<Slot>() == ROW_STRIDE,
                "generated ROW_STRIDE does not match the Slot layout"
            );

            #[repr(C, align(64))]
            struct AlignedRows([u8; TABLE_SIZE * ROW_STRIDE]);
            static ALIGNED_ROWS: AlignedRows =
                AlignedRows(*include_bytes!(concat!(env!("OUT_DIR"), "/weights.bin")));
            static SLOTS: &[Slot; TABLE_SIZE] = unsafe { ::core::mem::transmute(&ALIGNED_ROWS.0) };

            /// Per-slot verifier: the seeded hash of the slot's key with
            /// the row exponent packed into its low byte (see `ese::phf`).
            #[repr(C, align(8))]
            struct AlignedVerify([u8; TABLE_SIZE * 8]);
            static ALIGNED_VERIFY: AlignedVerify =
                AlignedVerify(*include_bytes!(concat!(env!("OUT_DIR"), "/verify.bin")));
            static VERIFY: &[u64; TABLE_SIZE] =
                unsafe { ::core::mem::transmute(&ALIGNED_VERIFY.0) };
            const _: () = assert!(
                ROW_EXP_MAX as usize <= $crate::accum::MAX_ROW_EXP,
                "generated row exponents exceed the runtime's MAX_ROW_EXP"
            );

            #[repr(C, align(2))]
            struct AlignedSeeds([u8; NUM_BUCKETS * 2]);
            static ALIGNED_SEEDS: AlignedSeeds =
                AlignedSeeds(*include_bytes!(concat!(env!("OUT_DIR"), "/seeds.bin")));
            static SEEDS: &[u16; NUM_BUCKETS] = unsafe { ::core::mem::transmute(&ALIGNED_SEEDS.0) };

            /// Resolve a key hash (`ese::phf::hash_key`) to its stored row
            /// and row exponent. The hash picks the bucket, the bucket's
            /// seed gives a second hash whose low bits pick the slot and
            /// whose full value is the verifier.
            #[inline(always)]
            pub fn lookup_hash(h: u64) -> Option<(&'static Param, u8)> {
                let bucket = $crate::phf::bucket(h, NUM_BUCKETS);
                // SAFETY: bucket < NUM_BUCKETS by construction and the slot is
                // masked into TABLE_SIZE, which is a power of two.
                unsafe {
                    let seed = *SEEDS.get_unchecked(bucket);
                    let x = $crate::phf::seeded(h, seed);
                    let slot = (x as usize) & TABLE_MASK;
                    let v = *VERIFY.get_unchecked(slot);
                    if $crate::phf::verifies(v, x) {
                        Some((&SLOTS.get_unchecked(slot).weight, v as u8))
                    } else {
                        None
                    }
                }
            }

            /// Resolve a vocabulary token to its stored row and row exponent.
            #[inline(always)]
            pub fn lookup(key: &str) -> Option<(&'static Param, u8)> {
                lookup_hash($crate::phf::hash_key(key.as_bytes()))
            }

            const ASCII: $crate::pretokenizer::AsciiTable =
                $crate::pretokenizer::AsciiTable::new(CLEAN_TEXT, LOWERCASE, SPECIAL_TOKENS);

            pub static PARAMS: $crate::ModelParams<Elem, DIMENSIONS> = $crate::ModelParams {
                tokenizer: $crate::TokenizerConfig {
                    lowercase: LOWERCASE,
                    strip_accents: STRIP_ACCENTS,
                    handle_chinese_chars: HANDLE_CHINESE_CHARS,
                    clean_text: CLEAN_TEXT,
                    subword_prefix: SUBWORD_PREFIX,
                    max_word_len: MAX_WORD_LEN,
                    max_token_bytes: MAX_TOKEN_BYTES,
                    special_tokens: SPECIAL_TOKENS,
                    unk_token: UNK_TOKEN,
                    ascii: &ASCII,
                },
                drop_unk: DROP_UNK,
                normalize: NORMALIZE,
                max_tokens: MAX_TOKENS,
                quant_min: &QUANT_MIN,
                quant_scale: &QUANT_SCALE,
                unk: &UNK,
                unk_exp: UNK_EXP,
            };
        }

        /// Number of dimensions per embedding.
        pub const DIMENSIONS: usize = model::DIMENSIONS;

        /// Encode a single text into an embedding vector.
        #[inline]
        pub fn encode_single(input: impl AsRef<str>) -> [f32; DIMENSIONS] {
            $crate::encode_single(&model::PARAMS, &model::lookup_hash, input.as_ref())
        }

        /// Encode a single text into caller-provided storage.
        #[inline]
        pub fn encode_into(input: impl AsRef<str>, out: &mut [f32; DIMENSIONS]) {
            $crate::encode_into(&model::PARAMS, &model::lookup_hash, input.as_ref(), out)
        }

        /// Encode a batch of texts. With the `rayon` feature of `ese` enabled,
        /// batches of `ese::PARALLEL_MIN` (64) or more are processed in parallel.
        pub fn encode(
            inputs: impl IntoIterator<Item = impl AsRef<str> + Send + Sync>,
        ) -> Vec<[f32; DIMENSIONS]> {
            $crate::encode(&model::PARAMS, &model::lookup_hash, inputs)
        }
    };
}
