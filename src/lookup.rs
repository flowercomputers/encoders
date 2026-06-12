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

/// Look up a token and return a reference to the Slot (weight + per-row params).
/// Returns None if the token is not in the vocabulary.
#[inline(always)]
pub fn lookup_slot(key: &str) -> Option<&'static Slot> {
    let bytes = key.as_bytes();
    let bucket = (hash(bytes, 0) as usize) % NUM_BUCKETS;
    unsafe {
        let slot = (hash(bytes, *SEEDS.get_unchecked(bucket) as u64) as usize) & TABLE_MASK;
        let entry = SLOTS.get_unchecked(slot);
        if entry.verify == hash(bytes, u64::MAX) {
            Some(entry)
        } else {
            None
        }
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
    let bucket = (hash(bytes, 0) as usize) % NUM_BUCKETS;
    unsafe {
        let seed = *SEEDS.get_unchecked(bucket) as u64;
        let slot_idx = (hash(bytes, seed) as usize) & TABLE_MASK;
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
    let bucket = (hash(bytes, 0) as usize) % NUM_BUCKETS;
    unsafe {
        let seed = *SEEDS.get_unchecked(bucket) as u64;
        let slot = (hash(bytes, seed) as usize) & TABLE_MASK;
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
