//! `BertNormalizer` and `BertPreTokenizer`, as implemented by the
//! `tokenizers` crate, fused into one streaming pass that never allocates.
//!
//! Special added tokens are handled the way `tokenizers` does: matched
//! literally on the raw input before any normalization (so `[SEP]` matches
//! and `[sep]` does not), each becoming one token regardless of
//! `add_special_tokens`.
//!
//! ASCII is classified and lowercased 64 bytes at a time with comparisons
//! that LLVM vectorizes, and its words are found from bitmasks and handed
//! to the callback straight from the block, without per-byte branches.
//! Everything else (non-ASCII, dropped control bytes, a possible special
//! token) goes through a per-character path: character classes and
//! normalized forms come from `ese-unicode`'s precomputed table, one lookup
//! per character, and characters collect in a fixed stack buffer until
//! whitespace, punctuation or a CJK character ends the word. WordPiece turns
//! any word longer than `max_word_len` characters into a single unknown
//! token, so a word that overflows the limit is reported as `overlong` and
//! its remaining characters are not stored.

use core::mem::MaybeUninit;

use ese_unicode::{class, class_of, normalize};

use crate::{TokenizerConfig, phf};

/// How a word handed to the callback should be looked up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordKind {
    /// Normalized text: WordPiece applies.
    Normal,
    /// Longer than `max_word_len` characters: a single unknown token.
    Overlong,
    /// A special token matched literally on the raw input: one exact lookup.
    Literal,
}

/// A pre-token found inside one block: `len` bytes from `start` of the
/// block's normalized (ASCII) bytes, with its key hash.
#[derive(Clone, Copy, Debug)]
pub struct BlockToken {
    pub start: u8,
    pub len: u8,
    pub kind: WordKind,
    pub hash: u64,
}

/// Receives pre-tokens, in input order.
pub trait WordSink {
    /// One pre-token; `hash` is its key hash (`phf::hash_key`; 0 for a
    /// literal special token or an overlong word).
    fn word(&mut self, word: &str, kind: WordKind, hash: u64);

    /// Consecutive pre-tokens from one block: token `t` is
    /// `text[t.start..t.start + t.len]`, ASCII. The default hands them to
    /// [`WordSink::word`] one at a time; a sink can override it to look all
    /// of them up before using any, so the lookups' memory loads overlap.
    #[inline(always)]
    fn block_words(&mut self, text: &[u8], tokens: &[BlockToken]) {
        for t in tokens {
            let s = t.start as usize;
            // SAFETY: block tokens are ASCII.
            let word = unsafe { core::str::from_utf8_unchecked(&text[s..s + t.len as usize]) };
            self.word(word, t.kind, t.hash);
        }
    }
}

impl<F: FnMut(&str, WordKind, u64)> WordSink for F {
    #[inline(always)]
    fn word(&mut self, word: &str, kind: WordKind, hash: u64) {
        self(word, kind, hash)
    }
}

/// Bytes of stack reserved for one normalized word. Must hold
/// `max_word_len` characters of up to four bytes each; the `include_model!`
/// macro asserts this at compile time.
pub const WORD_BUF: usize = 512;

struct WordBuf {
    /// The word, plus eight bytes of slack so [`phf::hash_padded`] can read
    /// whole chunks past its end.
    buf: [u8; WORD_BUF + 8],
    len: usize,
    chars: usize,
    max_chars: usize,
    overlong: bool,
}

impl WordBuf {
    #[cfg(test)]
    fn new(cfg: &TokenizerConfig) -> Self {
        WordBuf {
            buf: [0; WORD_BUF + 8],
            len: 0,
            chars: 0,
            max_chars: cfg.max_word_len,
            overlong: false,
        }
    }

    #[inline(always)]
    fn push(&mut self, ch: char) {
        if self.overlong {
            return;
        }
        self.chars += 1;
        let n = ch.len_utf8();
        if self.chars > self.max_chars || self.len + n > WORD_BUF {
            self.overlong = true;
            return;
        }
        ch.encode_utf8(&mut self.buf[self.len..]);
        self.len += n;
    }

    /// Push one ASCII byte that is already normalized.
    #[inline(always)]
    fn push_ascii(&mut self, b: u8) {
        if self.overlong {
            return;
        }
        self.chars += 1;
        if self.chars > self.max_chars || self.len >= WORD_BUF {
            self.overlong = true;
            return;
        }
        self.buf[self.len] = b;
        self.len += 1;
    }

    /// Push the first `n` bytes of `src`, ASCII and already normalized. Same
    /// result as `push_ascii` on each byte: once the word passes `max_chars`
    /// (or the buffer) it is overlong and nothing more is stored. Copies
    /// whole eight-byte chunks (a variable-length copy becomes a `memcpy`
    /// call), so `src` must extend to the next multiple of eight past `n`.
    #[inline(always)]
    fn push_run(&mut self, src: &[u8], n: usize) {
        if self.overlong {
            return;
        }
        if self.chars + n > self.max_chars || self.len + n > WORD_BUF {
            self.overlong = true;
            return;
        }
        self.chars += n;
        // `len + n <= WORD_BUF`, so the rounded-up copy stays inside the
        // eight bytes of slack.
        for k in (0..n).step_by(8) {
            let chunk: [u8; 8] = src[k..k + 8].try_into().unwrap();
            self.buf[self.len + k..self.len + k + 8].copy_from_slice(&chunk);
        }
        self.len += n;
    }

    #[inline(always)]
    fn pending(&self) -> bool {
        self.len > 0 || self.overlong
    }

    #[inline(always)]
    fn flush<F: WordSink>(&mut self, f: &mut F) {
        if self.len > 0 || self.overlong {
            // SAFETY: only whole chars are ever encoded into `buf[..len]`.
            let word = unsafe { core::str::from_utf8_unchecked(&self.buf[..self.len]) };
            if self.overlong {
                f.word(word, WordKind::Overlong, 0);
            } else {
                f.word(
                    word,
                    WordKind::Normal,
                    phf::hash_padded(&self.buf, self.len),
                );
            }
        }
        self.len = 0;
        self.chars = 0;
        self.overlong = false;
    }
}

/// What the fast path does with one input byte. Built once per model by
/// [`AsciiTable::new`] from the normalizer flags and the special tokens.
#[derive(Debug)]
pub struct AsciiTable {
    /// Per input byte: one of the `CLASS_*` codes, or'd with
    /// `SPECIAL_START` if a special token begins with that byte.
    class: [u8; 256],
    /// Per ASCII byte: the normalized byte to store for a word character.
    lower: [u8; 128],
    /// The flags the table was built with; the block classifier in
    /// [`for_each_word`] recomputes the classes from them with comparisons
    /// (which vectorize) instead of loading from `class`.
    clean_text: bool,
    lowercase: bool,
    /// Distinct first bytes of the special tokens, padded with `0xFF` (never
    /// a fast-path byte). If there are more than fit, `special_first_overflow`
    /// is set and the classifier consults `class` instead.
    special_first: [u8; SPECIAL_FIRSTS],
    special_first_overflow: bool,
}

/// Distinct special-token first bytes the block classifier compares against.
const SPECIAL_FIRSTS: usize = 4;

const CLASS_WORD: u8 = 0;
const CLASS_SPACE: u8 = 1;
const CLASS_PUNCT: u8 = 2;
const CLASS_DROP: u8 = 3;
const CLASS_NON_ASCII: u8 = 4;
const CLASS_MASK: u8 = 0x7F;
const SPECIAL_START: u8 = 0x80;

impl AsciiTable {
    /// The table for a normalizer with these flags and special tokens. It
    /// reproduces, for ASCII, exactly what the general per-character path
    /// does: `clean_text` drops controls and splits on whitespace,
    /// punctuation is isolated, letters are lowercased if `lowercase`.
    pub const fn new(clean_text: bool, lowercase: bool, special_tokens: &[&str]) -> Self {
        let mut class = [CLASS_NON_ASCII; 256];
        let mut lower = [0u8; 128];
        let mut b = 0usize;
        while b < 128 {
            let c = b as u8;
            let is_punct = matches!(c, 33..=47 | 58..=64 | 91..=96 | 123..=126);
            let cls = if clean_text {
                if c == 0 || (c.is_ascii_control() && !matches!(c, b'\t' | b'\n' | b'\r')) {
                    CLASS_DROP
                } else if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                    CLASS_SPACE
                } else if is_punct {
                    CLASS_PUNCT
                } else {
                    CLASS_WORD
                }
            } else if matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r') {
                // `char::is_whitespace` for ASCII.
                CLASS_SPACE
            } else if is_punct {
                CLASS_PUNCT
            } else {
                CLASS_WORD
            };
            class[b] = cls;
            lower[b] = if lowercase { c.to_ascii_lowercase() } else { c };
            b += 1;
        }
        let mut special_first = [0xFF; SPECIAL_FIRSTS];
        let mut nfirst = 0;
        let mut special_first_overflow = false;
        let mut t = 0;
        while t < special_tokens.len() {
            let bytes = special_tokens[t].as_bytes();
            if !bytes.is_empty() {
                let b0 = bytes[0];
                class[b0 as usize] |= SPECIAL_START;
                let mut seen = false;
                let mut k = 0;
                while k < nfirst {
                    seen |= special_first[k] == b0;
                    k += 1;
                }
                if !seen {
                    if nfirst < SPECIAL_FIRSTS {
                        special_first[nfirst] = b0;
                        nfirst += 1;
                    } else {
                        special_first_overflow = true;
                    }
                }
            }
            t += 1;
        }
        Self {
            class,
            lower,
            clean_text,
            lowercase,
            special_first,
            special_first_overflow,
        }
    }
}

/// Bytes classified at a time by [`for_each_word`]'s fast path, and so the
/// most tokens one [`WordSink::block_words`] call can carry.
pub const BLOCK: usize = 64;

/// One block's classification: bit `j` describes byte `j`. `slow` bytes
/// (non-ASCII, dropped controls, possible special-token starts) go through
/// [`scalar_step`]; every other byte is a word byte, punctuation or space.
struct Masks {
    word: u64,
    punct: u64,
    slow: u64,
}

/// Bits `p..cut` set.
#[inline(always)]
fn range_mask(p: usize, cut: usize) -> u64 {
    let below_cut = if cut >= 64 { !0 } else { (1u64 << cut) - 1 };
    below_cut & (!0u64 << p)
}

/// Bit `k` of the result is the top bit of byte `k` of `x`: eight bytes of
/// all-ones or all-zeros flags packed into eight bits with one multiply.
/// Each flag bit lands on a distinct bit of the product, so nothing carries.
#[inline(always)]
fn pack8(x: u64) -> u64 {
    (((x >> 7) & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080)) >> 56
}

/// Pack 64 byte flags (0x00 or 0xFF) into a bitmask.
#[inline(always)]
fn pack64(flags: &[u8; BLOCK]) -> u64 {
    let mut m = 0;
    for (k, chunk) in flags.as_chunks::<8>().0.iter().enumerate() {
        m |= pack8(u64::from_le_bytes(*chunk)) << (8 * k);
    }
    m
}

/// Classify and normalize one block. The per-byte part is a plain map from
/// bytes to flag bytes using comparisons only (no table loads, no branches),
/// which LLVM vectorizes; the flags are then packed to bitmasks eight bytes
/// at a time. Agrees with [`AsciiTable::class`] and [`AsciiTable::lower`]
/// for every byte (tested).
#[inline(always)]
fn classify(tab: &AsciiTable, b: &[u8; BLOCK], low: &mut [u8; BLOCK + 8]) -> Masks {
    let clean = tab.clean_text;
    let lowercase = tab.lowercase;
    let firsts = tab.special_first;
    let mut word = [0u8; BLOCK];
    let mut punct = [0u8; BLOCK];
    let mut slow = [0u8; BLOCK];
    for j in 0..BLOCK {
        let c = b[j];
        let is_punct = (33..=47).contains(&c)
            | (58..=64).contains(&c)
            | (91..=96).contains(&c)
            | (123..=126).contains(&c);
        let tnr = (c == b'\t') | (c == b'\n') | (c == b'\r');
        let space = (c == b' ') | tnr | (!clean & ((c == 0x0B) | (c == 0x0C)));
        let drop = clean & ((c < 32) | (c == 127)) & !tnr;
        let special = (c == firsts[0]) | (c == firsts[1]) | (c == firsts[2]) | (c == firsts[3]);
        let is_slow = (c >= 0x80) | drop | special;
        word[j] = 0u8.wrapping_sub(!(is_punct | space | is_slow) as u8);
        punct[j] = 0u8.wrapping_sub((is_punct & !special) as u8);
        slow[j] = 0u8.wrapping_sub(is_slow as u8);
        let upper = lowercase & c.is_ascii_uppercase();
        low[j] = c | ((upper as u8) << 5);
    }
    let mut m = Masks {
        word: pack64(&word),
        punct: pack64(&punct),
        slow: pack64(&slow),
    };
    if tab.special_first_overflow {
        for (j, &c) in b.iter().enumerate() {
            if tab.class[c as usize] & SPECIAL_START != 0 {
                m.slow |= 1 << j;
                m.word &= !(1 << j);
                m.punct &= !(1 << j);
            }
        }
    }
    m
}

/// Hand `tokens[..*ntok]` to `f` as one batch and empty it.
#[inline(always)]
fn hand_over<F: WordSink>(
    f: &mut F,
    low: &[u8],
    tokens: &[MaybeUninit<BlockToken>; BLOCK],
    ntok: &mut usize,
) {
    if *ntok > 0 {
        // SAFETY: the caller wrote `tokens[..ntok]`.
        let ts =
            unsafe { core::slice::from_raw_parts(tokens.as_ptr() as *const BlockToken, *ntok) };
        f.block_words(low, ts);
        *ntok = 0;
    }
}

/// Emit the tokens of block bytes `p..cut`, all word bytes, punctuation or
/// space. A token that starts and ends inside
/// the range goes straight to `f` from the block's own buffers. Only a word
/// that continues a pending one (from an earlier block or a slow byte) or
/// may continue past `cut` goes through `w`. `at_end` means the input ends
/// at `cut`, so nothing continues past it.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fast_segment<F: WordSink>(
    w: &mut WordBuf,
    f: &mut F,
    low: &[u8; BLOCK + 8],
    m: &Masks,
    p: usize,
    cut: usize,
    at_end: bool,
) {
    let range = range_mask(p, cut);
    let word = m.word & range;
    let punct = m.punct & range;
    // Word bytes whose previous byte is also a word byte in this range.
    let cont = word & (word << 1);
    let mut start = (word | punct) & !cont;
    let end = punct | (word & !(word >> 1));
    // A pending word continues only into a word byte.
    if w.pending() && (word >> p) & 1 == 0 {
        w.flush(f);
    }
    // Tokens wholly inside the range are collected and handed over together;
    // the batch is flushed before anything goes through `w`, to keep order.
    let mut tokens = [MaybeUninit::<BlockToken>::uninit(); BLOCK];
    let mut ntok = 0;
    let last = cut - 1;
    while start != 0 {
        let s = start.trailing_zeros() as usize;
        start &= start - 1;
        let e = s + (end >> s).trailing_zeros() as usize;
        let continues_in = s == p && w.pending();
        let continues_out = e == last && !at_end && (word >> e) & 1 != 0;
        if continues_in || continues_out {
            hand_over(f, low, &tokens, &mut ntok);
            w.push_run(&low[s..], e - s + 1);
            if !continues_out {
                w.flush(f);
            }
        } else {
            let len = e - s + 1;
            let kind = if len > w.max_chars {
                WordKind::Overlong
            } else {
                WordKind::Normal
            };
            tokens[ntok].write(BlockToken {
                start: s as u8,
                len: len as u8,
                kind,
                hash: phf::hash_padded(&low[s..], len),
            });
            ntok += 1;
        }
    }
    hand_over(f, low, &tokens, &mut ntok);
}

/// Normalize `input` and hand every pre-token to `f`, in order: runs of
/// tokens found inside one block through [`WordSink::block_words`], every
/// other token through [`WordSink::word`]. Normalization is
/// BertNormalizer's: `clean_text`, CJK padding, accent stripping (NFD, drop
/// non-spacing marks), lowercasing, each gated by its flag. Splitting is
/// BertPreTokenizer's: whitespace separates, punctuation is isolated.
/// Special tokens are matched literally first.
///
/// Input is classified 64 bytes at a time into bitmasks (see `classify`);
/// runs of plain ASCII are tokenized from the masks without per-byte
/// branches, and each remaining byte (non-ASCII, dropped controls, a possible
/// special token) takes the per-character `scalar_step`, after which the
/// block's masks are reused from the next position.
#[inline]
pub fn for_each_word<F: WordSink>(cfg: &TokenizerConfig, input: &str, mut f: F) {
    let mut word_buf = WordBuf {
        buf: [0; WORD_BUF + 8],
        len: 0,
        chars: 0,
        max_chars: cfg.max_word_len,
        overlong: false,
    };
    let w = &mut word_buf;
    let tab = cfg.ascii;
    let bytes = input.as_bytes();
    let len = bytes.len();
    // Eight bytes of slack for `phf::hash_padded`.
    let mut low = [0u8; BLOCK + 8];
    let mut i = 0;
    while i < len {
        let n = (len - i).min(BLOCK);
        let block: [u8; BLOCK] = if n == BLOCK {
            bytes[i..i + BLOCK].try_into().unwrap()
        } else {
            // Padding is outside every range used below.
            let mut b = [b' '; BLOCK];
            b[..n].copy_from_slice(&bytes[i..]);
            b
        };
        let m = classify(tab, &block, &mut low);
        let mut p = 0;
        loop {
            let slow = m.slow & range_mask(p, n);
            let cut = if slow == 0 {
                n
            } else {
                slow.trailing_zeros() as usize
            };
            if cut > p {
                let at_end = cut == n && i + n == len;
                fast_segment(w, &mut f, &low, &m, p, cut, at_end);
            }
            if cut == n {
                i += n;
                break;
            }
            let next = scalar_step(cfg, input, i + cut, w, &mut f);
            if next >= i + n {
                i = next;
                break;
            }
            p = next - i;
        }
    }
    w.flush(&mut f);
}

/// The per-character pre-tokenizer: handle the character (or special token)
/// at byte `i` and return the index after it. Used for the bytes
/// [`classify`] marks slow; on its own, in a loop, it is the whole
/// pre-tokenizer (the reference the block path is tested against).
#[inline(never)]
fn scalar_step<F: WordSink>(
    cfg: &TokenizerConfig,
    input: &str,
    i: usize,
    w: &mut WordBuf,
    f: &mut F,
) -> usize {
    let tab = cfg.ascii;
    let bytes = input.as_bytes();
    let b = bytes[i];
    let class = tab.class[b as usize];
    if class & SPECIAL_START != 0
        && let Some(t) = cfg
            .special_tokens
            .iter()
            .filter(|t| bytes[i..].starts_with(t.as_bytes()))
            .max_by_key(|t| t.len())
    {
        w.flush(f);
        f.word(t, WordKind::Literal, 0);
        return i + t.len();
    }
    match class & CLASS_MASK {
        CLASS_WORD => {
            w.push_ascii(tab.lower[b as usize]);
            i + 1
        }
        CLASS_SPACE => {
            w.flush(f);
            i + 1
        }
        CLASS_PUNCT => {
            w.flush(f);
            // SAFETY: `b` is ASCII.
            let piece = unsafe { core::str::from_utf8_unchecked(&bytes[i..i + 1]) };
            f.word(piece, WordKind::Normal, phf::hash_key(&[b]));
            i + 1
        }
        CLASS_DROP => i + 1,
        _ => {
            // SAFETY: `i` is on a char boundary: it only ever advances by
            // whole chars or whole special tokens.
            let ch = unsafe { input.get_unchecked(i..) }.chars().next().unwrap();
            let next = i + ch.len_utf8();
            let k = class_of(ch);
            if cfg.clean_text {
                if ch == '\u{FFFD}' || k & class::CONTROL != 0 {
                    return next;
                }
                if k & class::WHITESPACE != 0 {
                    w.flush(f);
                    return next;
                }
            }
            if cfg.handle_chinese_chars && k & class::CJK != 0 {
                w.flush(f);
                emit_normalized(cfg, ch, w, f);
                w.flush(f);
            } else {
                emit_normalized(cfg, ch, w, f);
            }
            next
        }
    }
}

/// Accent stripping and lowercasing of one non-ASCII character, each gated
/// by its flag, then BertPreTokenizer on every resulting character.
#[inline(always)]
fn emit_normalized<F: WordSink>(cfg: &TokenizerConfig, ch: char, w: &mut WordBuf, f: &mut F) {
    if !cfg.strip_accents && !cfg.lowercase {
        emit(ch, w, f);
    } else {
        for c in normalize(ch, cfg.strip_accents, cfg.lowercase).chars() {
            emit(c, w, f);
        }
    }
}

/// BertPreTokenizer applied to one normalized character: whitespace ends
/// the word, punctuation is a word of its own.
#[inline(always)]
fn emit<F: WordSink>(ch: char, w: &mut WordBuf, f: &mut F) {
    let (space, punct) = if ch.is_ascii() {
        // Normalization can yield ASCII (U+1FEF decomposes to '`'), and
        // BERT's ASCII punctuation is by range, not category.
        let b = ch as u8;
        (
            matches!(b, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r'),
            matches!(b, 33..=47 | 58..=64 | 91..=96 | 123..=126),
        )
    } else {
        let k = class_of(ch);
        (k & class::WHITESPACE != 0, k & class::PUNCTUATION != 0)
    };
    if space {
        w.flush(f);
    } else if punct {
        w.flush(f);
        let mut tmp = [0u8; 4];
        let piece = ch.encode_utf8(&mut tmp);
        f.word(piece, WordKind::Normal, phf::hash_key(piece.as_bytes()));
    } else {
        w.push(ch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECIALS: &[&str] = &["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
    const BERT: TokenizerConfig = TokenizerConfig {
        lowercase: true,
        strip_accents: true,
        handle_chinese_chars: true,
        clean_text: true,
        subword_prefix: "##",
        max_word_len: 100,
        max_token_bytes: 100,
        special_tokens: SPECIALS,
        unk_token: "[UNK]",
        ascii: &AsciiTable::new(true, true, SPECIALS),
    };

    fn words(cfg: &TokenizerConfig, s: &str) -> Vec<(String, WordKind)> {
        let mut v = Vec::new();
        for_each_word(cfg, s, |w: &str, k: WordKind, hash: u64| {
            if k == WordKind::Normal {
                // The hash must be exactly what hashing the bytes gives.
                assert_eq!(hash, phf::hash_key(w.as_bytes()), "{w:?}");
            }
            v.push((w.to_string(), k))
        });
        v
    }

    fn plain(cfg: &TokenizerConfig, s: &str) -> Vec<String> {
        words(cfg, s).into_iter().map(|(w, _)| w).collect()
    }

    #[test]
    fn bert_defaults() {
        assert_eq!(plain(&BERT, "Café\tRÉSUMÉ\0x"), ["cafe", "resumex"]);
        assert_eq!(plain(&BERT, "a你好b"), ["a", "你", "好", "b"]);
        assert_eq!(plain(&BERT, "straße"), ["straße"]);
        assert_eq!(
            plain(&BERT, "a]b[c  d.e!"),
            ["a", "]", "b", "[", "c", "d", ".", "e", "!"]
        );
        assert!(plain(&BERT, "   ").is_empty());
        assert!(plain(&BERT, "").is_empty());
    }

    #[test]
    fn format_and_private_use_chars_are_dropped() {
        // soft hyphen (Cf), zero-width space (Cf), BOM (Cf), private use (Co)
        assert_eq!(
            plain(&BERT, "co\u{AD}operate a\u{200B}b \u{FEFF}x \u{E000}y"),
            ["cooperate", "ab", "x", "y"]
        );
        // but tab, newline and carriage return are whitespace
        assert_eq!(plain(&BERT, "a\tb\nc\rd"), ["a", "b", "c", "d"]);
    }

    #[test]
    fn flags_off_is_identity() {
        let id = TokenizerConfig {
            lowercase: false,
            strip_accents: false,
            handle_chinese_chars: false,
            clean_text: false,
            ascii: {
                const T: AsciiTable = AsciiTable::new(false, false, SPECIALS);
                &T
            },
            ..BERT
        };
        assert_eq!(plain(&id, "Café 你好"), ["Café", "你好"]);
    }

    #[test]
    fn spacing_marks_survive_accent_stripping() {
        // Devanagari vowel sign (Mc) must be kept; only Mn is stripped.
        assert_eq!(plain(&BERT, "की"), ["की"]);
    }

    #[test]
    fn overlong_words_are_flagged() {
        let long = "x".repeat(101);
        let v = words(&BERT, &format!("ok {long} ok"));
        assert_eq!(v[0], ("ok".into(), WordKind::Normal));
        assert_eq!(v[1].1, WordKind::Overlong);
        assert_eq!(v[2], ("ok".into(), WordKind::Normal));
        let exact = "y".repeat(100);
        assert_eq!(words(&BERT, &exact), [(exact.clone(), WordKind::Normal)]);
    }

    #[test]
    fn special_tokens_match_literally_before_normalization() {
        let v = words(&BERT, "Hi[SEP]there [CLS] x[UNK]y [sep] [ CLS ]");
        let lit = |t: &str| (t.to_string(), WordKind::Literal);
        let n = |t: &str| (t.to_string(), WordKind::Normal);
        assert_eq!(
            v,
            [
                n("hi"),
                lit("[SEP]"),
                n("there"),
                lit("[CLS]"),
                n("x"),
                lit("[UNK]"),
                n("y"),
                n("["),
                n("sep"),
                n("]"),
                n("["),
                n("cls"),
                n("]"),
            ]
        );
        let none = TokenizerConfig {
            special_tokens: &[],
            ascii: {
                const T: AsciiTable = AsciiTable::new(true, true, &[]);
                &T
            },
            ..BERT
        };
        assert_eq!(words(&none, "[SEP]"), [n("["), n("sep"), n("]")]);
    }

    #[test]
    fn cjk_compatibility_ideograph_decomposes() {
        // U+F900 canonically decomposes to U+8C48.
        assert_eq!(plain(&BERT, "\u{F900}"), ["\u{8C48}"]);
    }

    /// The per-character pre-tokenizer alone: the behavior before the block
    /// path existed, used as the reference below.
    fn reference_words(cfg: &TokenizerConfig, s: &str) -> Vec<(String, WordKind, u64)> {
        let mut v = Vec::new();
        let mut f = |w: &str, k: WordKind, h: u64| v.push((w.to_string(), k, h));
        let w = &mut WordBuf::new(cfg);
        let mut i = 0;
        while i < s.len() {
            i = scalar_step(cfg, s, i, w, &mut f);
        }
        w.flush(&mut f);
        v
    }

    fn block_words(cfg: &TokenizerConfig, s: &str) -> Vec<(String, WordKind, u64)> {
        let mut v = Vec::new();
        for_each_word(cfg, s, |w: &str, k: WordKind, h: u64| {
            v.push((w.to_string(), k, h))
        });
        v
    }

    /// Overlong words carry whatever partial text and hashes were buffered;
    /// only their kind matters downstream.
    fn comparable(v: Vec<(String, WordKind, u64)>) -> Vec<(String, WordKind, u64)> {
        v.into_iter()
            .map(|(w, k, h)| match k {
                WordKind::Overlong => (String::new(), k, 0),
                _ => (w, k, h),
            })
            .collect()
    }

    #[test]
    fn classify_agrees_with_table() {
        let many: &[&str] = &["[A]", "<s>", "|x|", "madeup", "~", "é"];
        for clean in [false, true] {
            for lower in [false, true] {
                for specials in [&[][..], SPECIALS, many] {
                    let tab = AsciiTable::new(clean, lower, specials);
                    for base in [0usize, 64, 128, 192] {
                        let mut block = [0u8; BLOCK];
                        for (j, b) in block.iter_mut().enumerate() {
                            *b = (base + j) as u8;
                        }
                        let mut low = [0u8; BLOCK + 8];
                        let m = classify(&tab, &block, &mut low);
                        for (j, &c) in block.iter().enumerate() {
                            let class = tab.class[c as usize];
                            let slow = class & SPECIAL_START != 0
                                || matches!(class & CLASS_MASK, CLASS_DROP | CLASS_NON_ASCII);
                            assert_eq!(m.slow >> j & 1 != 0, slow, "slow {c} {clean} {lower}");
                            if !slow {
                                let cls = class & CLASS_MASK;
                                assert_eq!(m.word >> j & 1 != 0, cls == CLASS_WORD, "word {c}");
                                assert_eq!(m.punct >> j & 1 != 0, cls == CLASS_PUNCT, "punct {c}");
                                if cls == CLASS_WORD {
                                    assert_eq!(low[j], tab.lower[c as usize], "lower {c}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn block_path_matches_per_character_path() {
        const PIECES: &[&str] = &[
            "a", "b", "Z", "Q", "x", "7", " ", " ", "  ", "\t", "\n", "\r", "\0", "\u{7}", "\u{B}",
            "\u{7F}", ".", ",", "!", "[", "]", "#", "[UNK]", "[SEP]", "[sep]", "é", "É", "ß", "你",
            "好", "🙂", "\u{AD}", "\u{200B}", "\u{3000}", "\u{FFFD}", "hello", "World", "abcdefgh",
        ];
        let small = TokenizerConfig {
            max_word_len: 5,
            ..BERT
        };
        let flags_off = TokenizerConfig {
            lowercase: false,
            strip_accents: false,
            handle_chinese_chars: false,
            clean_text: false,
            ascii: {
                const T: AsciiTable = AsciiTable::new(false, false, SPECIALS);
                &T
            },
            ..BERT
        };
        let cfgs = [&BERT, &small, &flags_off];
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for case in 0..30_000 {
            let len = (next() % 160) as usize;
            let mut s = String::new();
            while s.len() < len {
                // Mostly long word runs, so words cross the 64-byte blocks.
                if next() % 3 == 0 {
                    for _ in 0..(next() % 70) {
                        s.push((b'a' + (next() % 26) as u8) as char);
                    }
                } else {
                    s.push_str(PIECES[(next() % PIECES.len() as u64) as usize]);
                }
            }
            for cfg in cfgs {
                assert_eq!(
                    comparable(block_words(cfg, &s)),
                    comparable(reference_words(cfg, &s)),
                    "case {case}: {s:?}"
                );
            }
        }
    }
}
