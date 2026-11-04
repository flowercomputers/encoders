//! `BertNormalizer` and `BertPreTokenizer`, as implemented by the
//! `tokenizers` crate, fused into one streaming pass that never allocates.
//!
//! Special added tokens are handled the way `tokenizers` does: matched
//! literally on the raw input before any normalization (so `[SEP]` matches
//! and `[sep]` does not), each becoming one token regardless of
//! `add_special_tokens`.
//!
//! Character classes and normalized forms come from `ese-unicode`'s
//! precomputed table, one lookup per non-ASCII character; ASCII is handled
//! by a per-model byte table. Normalized characters go into a fixed stack buffer and a word is handed
//! to the callback whenever whitespace, punctuation or a CJK character ends
//! it. WordPiece turns any word longer than `max_word_len` characters into a
//! single unknown token, so a word that overflows the limit is reported as
//! `overlong` and its remaining characters are not stored.

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

/// Bytes of stack reserved for one normalized word. Must hold
/// `max_word_len` characters of up to four bytes each; the `include_model!`
/// macro asserts this at compile time.
pub const WORD_BUF: usize = 512;

struct WordBuf {
    buf: [u8; WORD_BUF],
    /// `hashes[i]` is the key hash (`phf::hash_key`) of `buf[..=i]`, folded
    /// as bytes arrive so WordPiece's first probe needs no second pass.
    hashes: [MaybeUninit<u64>; WORD_BUF],
    hash: u64,
    len: usize,
    chars: usize,
    max_chars: usize,
    overlong: bool,
}

impl WordBuf {
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
        for i in self.len..self.len + n {
            self.hash = phf::fold_byte(self.hash, self.buf[i]);
            self.hashes[i].write(self.hash);
        }
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
        self.hash = phf::fold_byte(self.hash, b);
        self.hashes[self.len].write(self.hash);
        self.len += 1;
    }

    #[inline(always)]
    fn flush<F: FnMut(&str, WordKind, &[u64])>(&mut self, f: &mut F) {
        if self.len > 0 || self.overlong {
            // SAFETY: only whole chars are ever encoded into `buf[..len]`,
            // and `hashes[..len]` was written alongside.
            let word = unsafe { core::str::from_utf8_unchecked(&self.buf[..self.len]) };
            let hashes = unsafe {
                core::slice::from_raw_parts(self.hashes.as_ptr() as *const u64, self.len)
            };
            let kind = if self.overlong {
                WordKind::Overlong
            } else {
                WordKind::Normal
            };
            f(word, kind, hashes);
        }
        self.len = 0;
        self.hash = phf::INIT;
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
}

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
        let mut t = 0;
        while t < special_tokens.len() {
            let bytes = special_tokens[t].as_bytes();
            if !bytes.is_empty() {
                class[bytes[0] as usize] |= SPECIAL_START;
            }
            t += 1;
        }
        Self { class, lower }
    }
}

/// Normalize `input` and call `f(word, kind, hashes)` for every pre-token,
/// in order, where `hashes[i]` is the key hash of `word[..=i]` (empty for a
/// literal special token). Normalization is BertNormalizer's: `clean_text`, CJK padding,
/// accent stripping (NFD, drop non-spacing marks), lowercasing, each gated
/// by its flag. Splitting is BertPreTokenizer's: whitespace separates,
/// punctuation is isolated. Special tokens are matched literally first.
///
/// ASCII bytes take a table-driven path (one load, one branch per byte);
/// anything else goes through the general per-character code.
#[inline]
pub fn for_each_word<F: FnMut(&str, WordKind, &[u64])>(
    cfg: &TokenizerConfig,
    input: &str,
    mut f: F,
) {
    let mut w = WordBuf {
        buf: [0; WORD_BUF],
        hashes: [MaybeUninit::uninit(); WORD_BUF],
        hash: phf::INIT,
        len: 0,
        chars: 0,
        max_chars: cfg.max_word_len,
        overlong: false,
    };
    let tab = cfg.ascii;
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let class = tab.class[b as usize];
        if class & SPECIAL_START != 0
            && let Some(t) = cfg
                .special_tokens
                .iter()
                .filter(|t| bytes[i..].starts_with(t.as_bytes()))
                .max_by_key(|t| t.len())
        {
            w.flush(&mut f);
            f(t, WordKind::Literal, &[]);
            i += t.len();
            continue;
        }
        match class & CLASS_MASK {
            CLASS_WORD => {
                w.push_ascii(tab.lower[b as usize]);
                i += 1;
            }
            CLASS_SPACE => {
                w.flush(&mut f);
                i += 1;
            }
            CLASS_PUNCT => {
                w.flush(&mut f);
                // SAFETY: `b` is ASCII.
                let piece = unsafe { core::str::from_utf8_unchecked(&bytes[i..i + 1]) };
                f(piece, WordKind::Normal, &[phf::fold_byte(phf::INIT, b)]);
                i += 1;
            }
            CLASS_DROP => {
                i += 1;
            }
            _ => {
                // SAFETY: `i` is on a char boundary: it only ever advances by
                // whole chars or whole special tokens.
                let ch = unsafe { input.get_unchecked(i..) }.chars().next().unwrap();
                i += ch.len_utf8();
                let k = class_of(ch);
                if cfg.clean_text {
                    if ch == '\u{FFFD}' || k & class::CONTROL != 0 {
                        continue;
                    }
                    if k & class::WHITESPACE != 0 {
                        w.flush(&mut f);
                        continue;
                    }
                }
                if cfg.handle_chinese_chars && k & class::CJK != 0 {
                    w.flush(&mut f);
                    emit_normalized(cfg, ch, &mut w, &mut f);
                    w.flush(&mut f);
                } else {
                    emit_normalized(cfg, ch, &mut w, &mut f);
                }
            }
        }
    }
    w.flush(&mut f);
}

/// Accent stripping and lowercasing of one non-ASCII character, each gated
/// by its flag, then BertPreTokenizer on every resulting character.
#[inline(always)]
fn emit_normalized<F: FnMut(&str, WordKind, &[u64])>(
    cfg: &TokenizerConfig,
    ch: char,
    w: &mut WordBuf,
    f: &mut F,
) {
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
fn emit<F: FnMut(&str, WordKind, &[u64])>(ch: char, w: &mut WordBuf, f: &mut F) {
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
        let mut hashes = [0u64; 4];
        let mut h = phf::INIT;
        for (i, &b) in piece.as_bytes().iter().enumerate() {
            h = phf::fold_byte(h, b);
            hashes[i] = h;
        }
        f(piece, WordKind::Normal, &hashes[..piece.len()]);
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
        for_each_word(cfg, s, |w, k, hashes| {
            if k != WordKind::Literal {
                // Prefix hashes must be exactly what hashing the bytes gives.
                assert_eq!(hashes.len(), w.len());
                for (i, &h) in hashes.iter().enumerate() {
                    assert_eq!(h, phf::hash_key(&w.as_bytes()[..=i]), "{w:?} prefix {i}");
                }
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
}
