//! The `tokenizers` pipeline for tokenizers that go beyond plain BERT in the
//! one shape `ese-build` accepts (the gorp-code tables):
//!
//! - a `Sequence` normalizer of three `Replace` steps that split identifiers
//!   (`(?<=[a-z0-9])(?=[A-Z])`, then `(?<=[A-Z])(?=[A-Z][a-z])`, then `_`, each
//!   replaced by a space) followed by the usual `BertNormalizer`;
//! - non-special added tokens flagged `normalized` and `single_word`
//!   ("phrase tokens", e.g. `public static void main`), which `tokenizers`
//!   matches on the normalized text before pre-tokenization.
//!
//! The order is `tokenizers`': special tokens are split off the raw text, each
//! piece is normalized, phrase tokens are found on the normalized piece with a
//! leftmost-longest Aho-Corasick scan, and a match is kept only if no word
//! character touches it on either side (a rejected match is not retried
//! shorter; its text falls back to ordinary words). The text between kept
//! phrases goes through the regular streaming pre-tokenizer, whose own
//! normalization is idempotent on already-normalized text.
//!
//! This path allocates (one normalized copy per special-token-free piece);
//! plain BERT models never take it.

use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, MatchKind};
use ese_unicode::{class, class_of, normalize};

use crate::{
    TokenizerConfig,
    pretokenizer::{WordKind, for_each_word},
};

/// What the plain BERT path lacks. Built as a `static` by `include_model!`.
pub struct HfExtras {
    /// The three identifier-splitting `Replace` steps run before BERT.
    pub split_identifiers: bool,
    /// Phrase tokens, newline-separated, already in normalized form.
    pub phrases: &'static str,
    /// model2vec's character pre-cut (`max_length * median_token_length`
    /// code points); 0 for none.
    pub max_chars: usize,
    matcher: OnceLock<Option<AhoCorasick>>,
}

impl core::fmt::Debug for HfExtras {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HfExtras")
            .field("split_identifiers", &self.split_identifiers)
            .field("phrases", &self.phrases.lines().count())
            .field("max_chars", &self.max_chars)
            .finish()
    }
}

impl HfExtras {
    pub const fn new(split_identifiers: bool, phrases: &'static str, max_chars: usize) -> Self {
        Self {
            split_identifiers,
            phrases,
            max_chars,
            matcher: OnceLock::new(),
        }
    }

    /// Built on first use: about 46k patterns take a few milliseconds.
    fn matcher(&self) -> Option<&AhoCorasick> {
        self.matcher
            .get_or_init(|| {
                let pats: Vec<&str> = self.phrases.split('\n').filter(|p| !p.is_empty()).collect();
                if pats.is_empty() {
                    return None;
                }
                Some(
                    AhoCorasick::builder()
                        .match_kind(MatchKind::LeftmostLongest)
                        .build(pats)
                        .expect("phrase automaton"),
                )
            })
            .as_ref()
    }
}

/// `\w` as the `regex` crate defines it, which is what `tokenizers` uses for
/// the `single_word` check: alphabetic, marks, decimal digits, connector
/// punctuation and the joiners. Marks other than `Mn` and the non-ASCII
/// connectors are approximated with std's predicates; on normalized text
/// they are rare.
#[inline]
fn is_word_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    c.is_alphabetic()
        || c.is_numeric()
        || c == '\u{200C}'
        || c == '\u{200D}'
        || class_of(c) & class::NONSPACING_MARK != 0
}

/// The identifier-splitting `Replace` steps, in order, ASCII only like the
/// patterns.
fn split_identifiers(input: &str, out: &mut String) {
    let b = input.as_bytes();
    // Step 1 on the input: a space before an uppercase letter that follows a
    // lowercase letter or digit.
    let mut s1: Vec<u8> = Vec::with_capacity(b.len() + 8);
    for (i, &c) in b.iter().enumerate() {
        if i > 0 && (b[i - 1].is_ascii_lowercase() || b[i - 1].is_ascii_digit()) && c.is_ascii_uppercase() {
            s1.push(b' ');
        }
        s1.push(c);
    }
    // Step 2 on step 1's output: a space between an uppercase letter and an
    // uppercase letter followed by a lowercase one. Step 3: `_` to space.
    out.clear();
    for i in 0..s1.len() {
        let c = s1[i];
        if i > 0
            && s1[i - 1].is_ascii_uppercase()
            && c.is_ascii_uppercase()
            && s1.get(i + 1).is_some_and(|n| n.is_ascii_lowercase())
        {
            out.push(' ');
        }
        // Only ASCII bytes were inserted or replaced, so the UTF-8 is intact.
        if c == b'_' {
            out.push(' ');
        } else {
            // SAFETY: pushed byte by byte, but the sequence is the original
            // UTF-8 with ASCII inserted only between ASCII bytes.
            unsafe { out.as_mut_vec().push(c) };
        }
    }
}

/// `BertNormalizer` over a whole string, with the same per-character tables
/// as the streaming path.
fn bert_normalize(cfg: &TokenizerConfig, input: &str, out: &mut String) {
    out.clear();
    out.reserve(input.len());
    for ch in input.chars() {
        let k = class_of(ch);
        if cfg.clean_text {
            if ch == '\0' || ch == '\u{FFFD}' || (k & class::CONTROL != 0 && !matches!(ch, '\t' | '\n' | '\r')) {
                continue;
            }
            if k & class::WHITESPACE != 0 || matches!(ch, '\t' | '\n' | '\r') {
                out.push(' ');
                continue;
            }
        }
        if cfg.handle_chinese_chars && k & class::CJK != 0 {
            out.push(' ');
            out.push(ch);
            out.push(' ');
            continue;
        }
        if cfg.strip_accents || cfg.lowercase {
            out.push_str(&normalize(ch, cfg.strip_accents, cfg.lowercase));
        } else {
            out.push(ch);
        }
    }
}

/// Like [`for_each_word`], for a config that carries [`HfExtras`]. Phrase
/// tokens are reported as [`WordKind::Phrase`] and looked up exactly.
pub fn for_each_word_hf<F: FnMut(&str, WordKind, &[u64])>(
    cfg: &TokenizerConfig,
    extras: &HfExtras,
    input: &str,
    mut f: F,
) {
    let input = if extras.max_chars != 0 {
        match input.char_indices().nth(extras.max_chars) {
            Some((i, _)) => &input[..i],
            None => input,
        }
    } else {
        input
    };
    let mut split = String::new();
    let mut normalized = String::new();
    let mut piece = |raw: &str, f: &mut F| {
        let text = if extras.split_identifiers {
            split_identifiers(raw, &mut split);
            split.as_str()
        } else {
            raw
        };
        bert_normalize(cfg, text, &mut normalized);
        let s = normalized.as_str();
        let mut pos = 0;
        if let Some(ac) = extras.matcher() {
            for m in ac.find_iter(s) {
                let (a, b) = (m.start(), m.end());
                let before_ok = s[..a].chars().next_back().is_none_or(|c| !is_word_char(c));
                let after_ok = s[b..].chars().next().is_none_or(|c| !is_word_char(c));
                if !(before_ok && after_ok) {
                    continue;
                }
                for_each_word(cfg, &s[pos..a], &mut *f);
                f(&s[a..b], WordKind::Phrase, &[]);
                pos = b;
            }
        }
        for_each_word(cfg, &s[pos..], &mut *f);
    };
    // Special tokens are matched on the raw text first, longest first.
    let bytes = input.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        if let Some(t) = cfg
            .special_tokens
            .iter()
            .filter(|t| !t.is_empty() && bytes[i..].starts_with(t.as_bytes()))
            .max_by_key(|t| t.len())
        {
            piece(&input[start..i], &mut f);
            f(t, WordKind::Literal, &[]);
            i += t.len();
            start = i;
            continue;
        }
        i += 1;
        while i < bytes.len() && !input.is_char_boundary(i) {
            i += 1;
        }
    }
    piece(&input[start..], &mut f);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_splits_follow_the_replace_order() {
        let mut out = String::new();
        for (input, want) in [
            ("getUserName", "get User Name"),
            ("HTTPServer", "HTTP Server"),
            ("parse_json_v2", "parse json v2"),
            ("x2Y", "x2 Y"),
            ("ABCdef", "AB Cdef"),
            ("aB_cD", "a B c D"),
            ("éA", "éA"),
        ] {
            split_identifiers(input, &mut out);
            assert_eq!(out, want, "{input}");
        }
    }
}
