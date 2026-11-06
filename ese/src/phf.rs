//! Hash shared with `ese-build`'s perfect-hash construction. `ese-build`
//! uses these functions directly, so the two sides cannot drift.
//!
//! A key is folded left to right, one byte at a time, from a fixed initial
//! state; the fold state after each byte is the hash of that prefix. That
//! lets WordPiece hash every prefix of a word in one pass instead of
//! rehashing each candidate from scratch. The folded key then picks a bucket,
//! and the bucket's seed is mixed in *after* the fold to pick the slot. The
//! full seeded value is the slot's verifier.

const K: u64 = 0x2127599bf4325c37;

/// Fold state before any byte of the key.
pub const INIT: u64 = 0x517cc1b727220a95u64.wrapping_mul(K);

/// Fold one more byte into a key hash.
#[inline(always)]
pub const fn fold_byte(h: u64, b: u8) -> u64 {
    (h ^ b as u64).wrapping_mul(K)
}

/// Fold `bytes` into a key hash started at `h`.
#[inline(always)]
pub const fn fold(mut h: u64, bytes: &[u8]) -> u64 {
    let mut i = 0;
    while i < bytes.len() {
        h = fold_byte(h, bytes[i]);
        i += 1;
    }
    h
}

/// Hash of a complete key.
#[inline(always)]
pub const fn hash_key(key: &[u8]) -> u64 {
    fold(INIT, key)
}

/// Bucket of a folded key: the high bits of a 64x64 multiply, which spreads
/// uniform hashes as evenly as `% num_buckets` without the division.
#[inline(always)]
pub const fn bucket(h: u64, num_buckets: usize) -> usize {
    ((h as u128 * num_buckets as u128) >> 64) as usize
}

/// The seeded hash of a folded key. Its low bits pick the slot; the whole
/// value is the verifier. Seeds are 16 bits, so the seed table costs two
/// bytes per vocabulary entry (about 200 KB for a 100k vocabulary).
///
/// The seed is mixed into the fully folded key, and the mix is a bijection,
/// so two different keys under two different seeds collide only when their
/// 64-bit key hashes XOR to the XOR of the two scrambled seeds. Mixing the
/// seed in *before* the key let a different first byte cancel a different
/// seed and produced false lookups.
#[inline(always)]
pub const fn seeded(h: u64, seed: u16) -> u64 {
    let x = (h ^ (seed as u64).wrapping_mul(0x9E3779B97F4A7C15)).wrapping_mul(K);
    x ^ (x >> 32)
}

/// A slot's verifier packs the row exponent into the low byte of the
/// seeded hash, so one load answers both "is this the key" and "how is
/// the row scaled".
#[inline(always)]
pub const fn pack_verifier(seeded: u64, exp: u8) -> u64 {
    (seeded & !0xFF) | exp as u64
}

/// Whether `seeded` matches a packed verifier.
#[inline(always)]
pub const fn verifies(verifier: u64, seeded: u64) -> bool {
    (verifier ^ seeded) >> 8 == 0
}
