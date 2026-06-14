include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

/// Slot layout:
///
/// f32 build (no quant features):
///   { verify: u64, weight: [f32; DIMS] }
///   SLOT_SIZE = 8 + DIMENSIONS * DTYPE_SIZE
///
/// quant builds (quant-8 / quant-16):
///   { verify: u64, row_min: f32, row_scale: f32, weight: [uN; DIMS] }
///   SLOT_SIZE = 8 + 8 + DIMENSIONS * DTYPE_SIZE
///
/// The per-row (min, scale) fields are gated on the quant features so
/// the f32 build's layout is unchanged (zero cost, no struct padding change).
#[repr(C, align(8))]
pub(crate) struct Slot {
    pub(crate) verify: u64,
    #[cfg(any(feature = "quant-8", feature = "quant-16"))]
    pub(crate) row_min: f32,
    #[cfg(any(feature = "quant-8", feature = "quant-16"))]
    pub(crate) row_scale: f32,
    pub(crate) weight: Param,
}

#[repr(C, align(8))]
struct AlignedSlots([u8; TABLE_SIZE * SLOT_SIZE]);
static ALIGNED_SLOTS: AlignedSlots =
    AlignedSlots(*include_bytes!(concat!(env!("OUT_DIR"), "/weights.bin")));
static SLOTS: &'static [Slot; TABLE_SIZE] = unsafe { std::mem::transmute(&ALIGNED_SLOTS.0) };

#[repr(C, align(4))]
struct AlignedSeeds([u8; NUM_BUCKETS * 4]);
static ALIGNED_S: AlignedSeeds =
    AlignedSeeds(*include_bytes!(concat!(env!("OUT_DIR"), "/seeds.bin")));
static SEEDS: &'static [u32; NUM_BUCKETS] = unsafe { std::mem::transmute(&ALIGNED_S.0) };

// scores.bin: [f32; TABLE_SIZE] parallel to PHF slots, with the log-prob score
// for each Unigram vocab token. Empty slots contain 0.0. Only meaningful when
// IS_UNIGRAM=true; for WordPiece builds this is a 4-byte dummy and these
// functions are dead code (eliminated by the const branch in encode_single).
#[allow(dead_code)]
static SCORES_RAW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scores.bin"));

// slot_ids.bin: [u32; TABLE_SIZE] — slot_ids[slot_idx] = vocab_id (original token index)
// for occupied slots, u32::MAX for empty slots.
// Only meaningful for BPE builds (IS_BPE=true). For WordPiece/Unigram builds this is a
// 4-byte dummy file, and bpe.rs is dead code (IS_BPE=false elides it via const dispatch).
#[allow(dead_code)]
static SLOT_IDS_RAW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/slot_ids.bin"));

/// Look up a token and return a reference to the Slot (weight + per-row params).
/// Returns None if the token is not in the vocabulary.
#[inline(always)]
pub fn lookup_slot(key: &str) -> Option<&'static Slot> {
    let bytes = key.as_bytes();
    // Use u64 arithmetic throughout before casting to usize.
    // On 64-bit hosts usize == u64 so this is identical to the original.
    // On wasm32 (usize == u32) a direct `as usize` would truncate the hash
    // before the modulo, producing wrong bucket indices — the PHF was built
    // with 64-bit hashes, so the runtime must match.
    let bucket = (hash(bytes, 0) % NUM_BUCKETS as u64) as usize;
    unsafe {
        let slot = (hash(bytes, *SEEDS.get_unchecked(bucket) as u64) & TABLE_MASK as u64) as usize;
        let entry = SLOTS.get_unchecked(slot);
        if entry.verify == hash(bytes, u64::MAX) {
            Some(entry)
        } else {
            None
        }
    }
}

/// Look up a token and return its PHF slot index.
/// Returns None if the token is not in the vocabulary.
/// Used by src/bpe.rs to retrieve the vocab_id from SLOT_IDS_RAW.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn lookup_slot_index(key: &str) -> Option<usize> {
    let bytes = key.as_bytes();
    // u64 arithmetic throughout before casting to usize — see lookup_slot:
    // a direct `as usize` truncates the hash on wasm32 (usize == u32).
    let bucket = (hash(bytes, 0) % NUM_BUCKETS as u64) as usize;
    unsafe {
        let slot = (hash(bytes, *SEEDS.get_unchecked(bucket) as u64) & TABLE_MASK as u64) as usize;
        let entry = SLOTS.get_unchecked(slot);
        if entry.verify == hash(bytes, u64::MAX) {
            Some(slot)
        } else {
            None
        }
    }
}

/// Look up a token and return its vocab id (original token index in the model vocabulary).
/// Returns None if the token is not in the vocabulary.
/// Only meaningful for BPE builds (IS_BPE=true); for WordPiece/Unigram the slot_ids.bin is
/// a dummy. Used by src/bpe.rs to map token strings to numeric ids for merge pair keying.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn lookup_vocab_id(key: &str) -> Option<u32> {
    let slot = lookup_slot_index(key)?;
    unsafe {
        let id_offset = slot * 4;
        let id = u32::from_le_bytes(
            SLOT_IDS_RAW[id_offset..id_offset + 4]
                .try_into()
                .unwrap_unchecked(),
        );
        Some(id)
    }
}

/// Convenience wrapper that returns only the weight Param (for callers that do not
/// need per-row params — currently only the UNK/CLS/SEP special-token consts path
/// and the Unigram lookup_with_score helper).
#[allow(dead_code)]
#[inline(always)]
pub fn lookup(key: &str) -> Option<&'static Param> {
    lookup_slot(key).map(|s| &s.weight)
}

/// Look up a Unigram vocab token, returning both its Slot and log-prob score.
/// Returns None if the token is not in the vocabulary.
/// Only meaningful when IS_UNIGRAM=true; the score array is a dummy for WordPiece.
#[allow(dead_code)]
#[inline(always)]
pub fn lookup_slot_with_score(key: &str) -> Option<(&'static Slot, f32)> {
    let bytes = key.as_bytes();
    let bucket = (hash(bytes, 0) % NUM_BUCKETS as u64) as usize;
    unsafe {
        let seed = *SEEDS.get_unchecked(bucket) as u64;
        let slot_idx = (hash(bytes, seed) & TABLE_MASK as u64) as usize;
        let entry = SLOTS.get_unchecked(slot_idx);
        if entry.verify == hash(bytes, u64::MAX) {
            let score_offset = slot_idx * 4;
            let score_bytes = SCORES_RAW.get_unchecked(score_offset..score_offset + 4);
            let score = f32::from_le_bytes([score_bytes[0], score_bytes[1], score_bytes[2], score_bytes[3]]);
            Some((entry, score))
        } else {
            None
        }
    }
}

/// Look up a Unigram vocab token, returning both its weight row and log-prob score.
/// Returns None if the token is not in the vocabulary.
/// Only meaningful when IS_UNIGRAM=true; the score array is a dummy for WordPiece.
/// Prefer lookup_slot_with_score when per-row quant params are needed.
#[allow(dead_code)]
#[inline(always)]
pub fn lookup_with_score(key: &str) -> Option<(&'static Param, f32)> {
    let bytes = key.as_bytes();
    let bucket = (hash(bytes, 0) % NUM_BUCKETS as u64) as usize;
    unsafe {
        let seed = *SEEDS.get_unchecked(bucket) as u64;
        let slot = (hash(bytes, seed) & TABLE_MASK as u64) as usize;
        let entry = SLOTS.get_unchecked(slot);
        if entry.verify == hash(bytes, u64::MAX) {
            // Read the f32 score from the parallel scores array at this slot
            let score_offset = slot * 4;
            let score_bytes = SCORES_RAW.get_unchecked(score_offset..score_offset + 4);
            let score = f32::from_le_bytes([score_bytes[0], score_bytes[1], score_bytes[2], score_bytes[3]]);
            Some((&entry.weight, score))
        } else {
            None
        }
    }
}

#[inline(always)]
fn hash(key: &[u8], seed: u64) -> u64 {
    let mut h = seed ^ 0x517cc1b727220a95;
    for &b in key {
        h = (h ^ b as u64).wrapping_mul(0x2127599bf4325c37);
    }
    h ^ (h >> 32)
}
