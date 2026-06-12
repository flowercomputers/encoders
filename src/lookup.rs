include!(concat!(env!("OUT_DIR"), "/model_constants.rs"));

#[repr(C, align(8))]
struct Slot {
    verify: u64,
    weight: Param,
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

#[inline(always)]
pub fn lookup(key: &str) -> Option<&'static Param> {
    let bytes = key.as_bytes();
    let bucket = (hash(bytes, 0) as usize) % NUM_BUCKETS;
    unsafe {
        let slot = (hash(bytes, *SEEDS.get_unchecked(bucket) as u64) as usize) & TABLE_MASK;
        let entry = SLOTS.get_unchecked(slot);
        if entry.verify == hash(bytes, u64::MAX) {
            Some(&entry.weight)
        } else {
            None
        }
    }
}

/// Look up a Unigram vocab token, returning both its weight row and log-prob score.
/// Returns None if the token is not in the vocabulary.
/// Only meaningful when IS_UNIGRAM=true; the score array is a dummy for WordPiece.
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

/// Scan SCORES_RAW for the minimum (most-negative) f32 value.
/// Called from unigram.rs to compute unk_score = min_score - K_UNK_PENALTY.
/// Empty PHF slots contain 0.0; real Unigram scores are negative log-probs.
/// For WordPiece builds IS_UNIGRAM=false so this is dead code.
#[allow(dead_code)]
pub fn scan_min_score_from_raw() -> f32 {
    let mut min_score = 0.0f32;
    let mut i = 0usize;
    while i + 4 <= SCORES_RAW.len() {
        let bytes = [SCORES_RAW[i], SCORES_RAW[i + 1], SCORES_RAW[i + 2], SCORES_RAW[i + 3]];
        let v = f32::from_le_bytes(bytes);
        if v < min_score {
            min_score = v;
        }
        i += 4;
    }
    min_score
}

#[inline(always)]
fn hash(key: &[u8], seed: u64) -> u64 {
    let mut h = seed ^ 0x517cc1b727220a95;
    for &b in key {
        h = (h ^ b as u64).wrapping_mul(0x2127599bf4325c37);
    }
    h ^ (h >> 32)
}
