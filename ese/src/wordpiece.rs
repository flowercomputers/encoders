//! Greedy longest-match-first WordPiece, feeding rows to the accumulator.

use core::mem::MaybeUninit;

use crate::{
    Elem, TokenizerConfig,
    accum::Accumulator,
    phf::{self, Hasher},
    pretokenizer::{WORD_BUF, WordKind},
};

#[inline(always)]
fn unknown<E: Elem, const D: usize>(
    pool: &mut Accumulator<E, D>,
    unk: Option<(&'static [E; D], u8)>,
) {
    if let Some((row, exp)) = unk {
        pool.add(row, exp);
    }
}

/// Split `word` into WordPiece tokens and add each token's row to `pool`.
/// `lookup(hash)` resolves a key hash (see [`phf`]) to its row and row
/// exponent. `hash` is the word's key hash as computed by the pre-tokenizer,
/// which settles the common case of a word that is one vocabulary entry in
/// a single probe. A word that cannot be fully segmented, or was flagged
/// overlong by the pre-tokenizer, becomes a single unknown token; `unk == None`
/// means unknown tokens are dropped (model2vec semantics). A literal special
/// token is one exact lookup, and a literal unknown token follows the same
/// UNK policy.
///
/// Otherwise the hash of every candidate piece starting at one position is
/// one pass over the bytes ([`Hasher`] yields each prefix's hash), and no
/// piece longer than the longest vocabulary entry is tried, so a long word
/// that is not in the vocabulary costs `O(len * max_token)` probes rather
/// than `O(len^3)` bytes hashed.
#[inline]
pub fn wordpiece_accumulate<E, const D: usize, L>(
    cfg: &TokenizerConfig,
    word: &str,
    kind: WordKind,
    hash: u64,
    pool: &mut Accumulator<E, D>,
    lookup: &L,
    unk: Option<(&'static [E; D], u8)>,
) where
    E: Elem,
    L: Fn(u64) -> Option<(&'static [E; D], u8)>,
{
    match kind {
        WordKind::Normal => {}
        WordKind::Overlong => {
            unknown(pool, unk);
            return;
        }
        WordKind::Literal => {
            if word == cfg.unk_token {
                unknown(pool, unk);
            } else if let Some((row, exp)) = lookup(phf::hash_key(word.as_bytes())) {
                pool.add(row, exp);
            } else {
                unknown(pool, unk);
            }
            return;
        }
    }
    // Fast path: most words are one vocabulary entry, and the greedy search
    // below would try the whole word first anyway. Its hash is already in
    // `hashes`, so a hit costs one probe and none of the search setup.
    let n = word.len();
    let found = if n <= cfg.max_token_bytes && n > 0 {
        lookup(hash)
    } else {
        None
    };
    accumulate_found(cfg, word, found, pool, lookup, unk);
}

/// [`wordpiece_accumulate`] for a normal word whose whole-word lookup has
/// already been done (`found`, ignored when the word cannot be one
/// vocabulary entry), so a caller can do several words' lookups together.
#[inline(always)]
pub fn accumulate_found<E, const D: usize, L>(
    cfg: &TokenizerConfig,
    word: &str,
    found: Option<(&'static [E; D], u8)>,
    pool: &mut Accumulator<E, D>,
    lookup: &L,
    unk: Option<(&'static [E; D], u8)>,
) where
    E: Elem,
    L: Fn(u64) -> Option<(&'static [E; D], u8)>,
{
    // Fast path: most words are one vocabulary entry, and the greedy search
    // would try the whole word first anyway.
    let n = word.len();
    if n <= cfg.max_token_bytes
        && n > 0
        && let Some((row, exp)) = found
    {
        pool.add(row, exp);
        return;
    }
    segment(cfg, word, pool, lookup, unk);
}

/// The full greedy longest-match search, for words that are not a single
/// vocabulary entry. Kept out of line so the common whole-word path above
/// stays small.
#[inline(never)]
fn segment<E, const D: usize, L>(
    cfg: &TokenizerConfig,
    word: &str,
    pool: &mut Accumulator<E, D>,
    lookup: &L,
    unk: Option<(&'static [E; D], u8)>,
) where
    E: Elem,
    L: Fn(u64) -> Option<(&'static [E; D], u8)>,
{
    let bytes = word.as_bytes();
    let n = bytes.len();
    // local[k] is the hash of the piece `word[start..start + k + 1]`.
    let mut local: [MaybeUninit<u64>; WORD_BUF] = [MaybeUninit::uninit(); WORD_BUF];
    // Pieces found so far. A word becomes one unknown token if any position
    // fails to match, so nothing reaches the pool before the whole word has
    // segmented. There are at most as many pieces as characters.
    let mut pieces: [MaybeUninit<(&'static [E; D], u8)>; MAX_PIECES] =
        [MaybeUninit::uninit(); MAX_PIECES];
    let mut npieces = 0;
    let mut start = 0;
    while start < n {
        let limit = n.min(start + cfg.max_token_bytes);
        // Hash of `word[start..start + k + 1]` at index k.
        let mut hs = if start == 0 {
            Hasher::new()
        } else {
            Hasher::with(cfg.subword_prefix.as_bytes())
        };
        for (slot, &b) in local.iter_mut().zip(&bytes[start..limit]) {
            hs = hs.push(b);
            slot.write(hs.hash());
        }
        // SAFETY: the first `limit - start` slots were written just above.
        let piece_hashes: &[u64] =
            unsafe { core::slice::from_raw_parts(local.as_ptr() as *const u64, limit - start) };
        let mut end = limit;
        while end > start && !word.is_char_boundary(end) {
            end -= 1;
        }
        loop {
            if end == start {
                unknown(pool, unk);
                return;
            }
            if let Some(found) = lookup(piece_hashes[end - start - 1]) {
                if npieces == MAX_PIECES {
                    // Unreachable for `max_word_len <= MAX_PIECES`, which
                    // `include_model!` asserts; stay safe regardless.
                    unknown(pool, unk);
                    return;
                }
                pieces[npieces].write(found);
                npieces += 1;
                start = end;
                break;
            }
            end -= 1;
            while end > start && !word.is_char_boundary(end) {
                end -= 1;
            }
        }
    }
    for p in &pieces[..npieces] {
        // SAFETY: `pieces[..npieces]` were written above.
        let (row, exp) = unsafe { p.assume_init() };
        pool.add(row, exp);
    }
}

/// Upper bound on pieces per word: one per character. `include_model!`
/// asserts `max_word_len <= MAX_PIECES`.
pub const MAX_PIECES: usize = WORD_BUF / 4;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenizer::AsciiTable;
    use std::collections::HashMap;

    const VOCAB: &[&str] = &[
        "[UNK]", "un", "##aff", "##able", "##s", "a", "b", "##a", "##b", "##c", "ab", "##ab",
        "abc", "x", "##y", "##z", "xy", "é", "##é", "##ß", "ñ", "##ñ", "##ñb", "##q", "straße",
        "##e", "##ße",
    ];
    static ROWS: [[u8; 2]; 27] = {
        let mut r = [[0u8; 2]; 27];
        let mut i = 0;
        while i < 27 {
            r[i] = [i as u8, 0];
            i += 1;
        }
        r
    };
    const ASCII: AsciiTable = AsciiTable::new(true, true, &["[UNK]"]);
    const CFG: TokenizerConfig = TokenizerConfig {
        lowercase: true,
        strip_accents: false,
        handle_chinese_chars: true,
        clean_text: true,
        subword_prefix: "##",
        max_word_len: 100,
        max_token_bytes: 7,
        special_tokens: &["[UNK]"],
        unk_token: "[UNK]",
        ascii: &ASCII,
    };

    /// Token ids the `tokenizers` algorithm produces, written the naive way.
    fn naive(word: &str) -> Vec<usize> {
        let id = |s: &str| VOCAB.iter().position(|t| *t == s);
        let mut out = Vec::new();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while end > start {
                if word.is_char_boundary(end) {
                    let piece = if start == 0 {
                        word[start..end].to_string()
                    } else {
                        format!("##{}", &word[start..end])
                    };
                    if let Some(i) = id(&piece) {
                        found = Some((i, end));
                        break;
                    }
                }
                end -= 1;
            }
            match found {
                Some((i, e)) => {
                    out.push(i);
                    start = e;
                }
                None => return vec![0],
            }
        }
        out
    }

    #[test]
    fn matches_naive_longest_match() {
        let table: HashMap<u64, usize> = VOCAB
            .iter()
            .enumerate()
            .map(|(i, t)| (phf::hash_key(t.as_bytes()), i))
            .collect();
        let lookup = |h: u64| table.get(&h).map(|&i| (&ROWS[i], 0u8));
        let long = "b".repeat(99) + "q";
        let words = [
            long.as_str(),
            "unaffable",
            "ab",
            "abc",
            "abab",
            "abcab",
            "xyz",
            "xyzzy",
            "q",
            "aq",
            "éñb",
            "straße",
            "straßes",
            "ñbñb",
            "abababababababababab",
            "unaffables",
            &("b".repeat(99) + "q"),
        ];
        for word in words {
            let mut slot = MaybeUninit::uninit();
            let pool = Accumulator::<u8, 2>::init(&mut slot);
            wordpiece_accumulate(
                &CFG,
                word,
                WordKind::Normal,
                phf::hash_key(word.as_bytes()),
                pool,
                &lookup,
                Some((&ROWS[0], 0)),
            );
            let n = pool.count() as usize;
            let out = pool.finish(&[0.0; 2], &[1.0; 2], false);
            // Each row's first byte is its vocabulary id, so the mean times
            // the count is the sum of ids.
            let want = naive(word);
            assert_eq!(n, want.len(), "{word}: token count");
            let sum: usize = want.iter().sum();
            assert!(
                ((out[0] * n as f32) - sum as f32).abs() < 1e-3,
                "{word}: ids sum {} vs {}",
                out[0] * n as f32,
                sum
            );
        }
    }
}
