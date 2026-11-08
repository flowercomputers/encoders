//! Hash shared with `ese-build`'s perfect-hash construction. `ese-build`
//! uses these functions directly, so the two sides cannot drift.
//!
//! A key is folded eight bytes at a time (little-endian, the last chunk
//! zero-padded), one multiply per chunk, and its length is mixed in last, so
//! a word of up to eight bytes hashes in two multiplies. Hashes of every
//! prefix of a word, which WordPiece needs when the whole word is not in the
//! vocabulary, come from [`Hasher`] one byte at a time. The folded key then
//! picks a bucket, and the bucket's seed is mixed in *after* the fold to pick
//! the slot. The full seeded value is the slot's verifier.
//!
//! Each chunk step `h -> (h ^ chunk) * K` is a bijection for a fixed chunk,
//! so two different keys of the same length never fold to the same value.

const K: u64 = 0x2127599bf4325c37;

/// Fold state before any chunk of the key.
pub const INIT: u64 = 0x517cc1b727220a95u64.wrapping_mul(K);

/// Fold one eight-byte chunk into a key hash.
#[inline(always)]
pub const fn mix(h: u64, chunk: u64) -> u64 {
    (h ^ chunk).wrapping_mul(K)
}

/// Mix the key length into the folded chunks. The final shift moves the
/// high half, where a difference in a chunk's upper bytes ends up, into the
/// low bits that pick the slot.
#[inline(always)]
pub const fn finish(h: u64, len: usize) -> u64 {
    let x = (h ^ len as u64).wrapping_mul(K);
    x ^ (x >> 32)
}

/// The low `n` bytes of `x` (`n < 8`).
#[inline(always)]
const fn low_bytes(x: u64, n: usize) -> u64 {
    x & ((1u64 << (8 * n)) - 1)
}

/// Hash of a complete key.
#[inline]
pub const fn hash_key(key: &[u8]) -> u64 {
    let mut h = INIT;
    let mut i = 0;
    while i + 8 <= key.len() {
        let mut c = 0u64;
        let mut k = 0;
        while k < 8 {
            c |= (key[i + k] as u64) << (8 * k);
            k += 1;
        }
        h = mix(h, c);
        i += 8;
    }
    if i < key.len() {
        let mut c = 0u64;
        let mut k = 0;
        while i + k < key.len() {
            c |= (key[i + k] as u64) << (8 * k);
            k += 1;
        }
        h = mix(h, c);
    }
    finish(h, key.len())
}

/// [`hash_key`] of `buf[..len]` using whole eight-byte loads: `buf` must
/// extend to the next multiple of eight past `len` (the extra bytes are
/// masked off, whatever they hold).
#[inline(always)]
pub fn hash_padded(buf: &[u8], len: usize) -> u64 {
    let (chunks, _) = buf.as_chunks::<8>();
    let full = len / 8;
    let mut h = INIT;
    for c in &chunks[..full] {
        h = mix(h, u64::from_le_bytes(*c));
    }
    let rest = len % 8;
    if rest != 0 {
        h = mix(h, low_bytes(u64::from_le_bytes(chunks[full]), rest));
    }
    finish(h, len)
}

/// Hashes of a key's prefixes, one byte at a time: after pushing `k` bytes,
/// [`Hasher::hash`] is the [`hash_key`] of those `k` bytes.
#[derive(Clone, Copy, Debug)]
pub struct Hasher {
    h: u64,
    tail: u64,
    len: usize,
}

impl Hasher {
    pub const fn new() -> Self {
        Self {
            h: INIT,
            tail: 0,
            len: 0,
        }
    }

    /// A hasher that has already consumed `bytes`.
    pub const fn with(bytes: &[u8]) -> Self {
        let mut s = Self::new();
        let mut i = 0;
        while i < bytes.len() {
            s = s.push(bytes[i]);
            i += 1;
        }
        s
    }

    #[inline(always)]
    pub const fn push(mut self, b: u8) -> Self {
        self.tail |= (b as u64) << (8 * (self.len % 8));
        self.len += 1;
        if self.len.is_multiple_of(8) {
            self.h = mix(self.h, self.tail);
            self.tail = 0;
        }
        self
    }

    #[inline(always)]
    pub const fn hash(&self) -> u64 {
        let h = if !self.len.is_multiple_of(8) {
            mix(self.h, self.tail)
        } else {
            self.h
        };
        finish(h, self.len)
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Bucket of a folded key: the high bits of a 64x64 multiply, which spreads
/// uniform hashes as evenly as `% num_buckets` without the division.
#[inline(always)]
pub const fn bucket(h: u64, num_buckets: usize) -> usize {
    ((h as u128 * num_buckets as u128) >> 64) as usize
}

/// Slot of a seeded hash in a table of `table_size` slots: the high bits of
/// a 64x64 multiply, so the table can be any size and is sized to the
/// vocabulary rather than rounded up to a power of two.
#[inline(always)]
pub const fn slot(x: u64, table_size: usize) -> usize {
    ((x as u128 * table_size as u128) >> 64) as usize
}

/// The seeded hash of a folded key. [`slot`] maps it into the table; the
/// whole value is the verifier. Seeds are 16 bits, so the seed table costs two
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_three_ways_agree() {
        let text = b"the quick brown fox jumps over the lazy dog, ##pieces and \xc3\xa9t\xc3\xa9";
        let mut padded = [0xAAu8; 96];
        padded[..text.len()].copy_from_slice(text);
        for start in 0..text.len() {
            let mut hs = Hasher::new();
            for end in start + 1..=text.len() {
                hs = hs.push(text[end - 1]);
                let key = &text[start..end];
                let want = hash_key(key);
                assert_eq!(hs.hash(), want, "{start}..{end}");
                assert_eq!(hash_padded(&padded[start..], end - start), want);
            }
        }
        assert_eq!(Hasher::with(b"##ab").hash(), hash_key(b"##ab"));
    }

    #[test]
    fn length_is_part_of_the_key() {
        assert_ne!(hash_key(b"a"), hash_key(b"a\0"));
        assert_ne!(hash_key(b""), hash_key(b"\0"));
        assert_ne!(hash_key(b"abcdefgh"), hash_key(b"abcdefgh\0"));
    }
}
