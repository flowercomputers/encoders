//! Reads the parts of `tokenizer.json` the runtime needs and rejects
//! tokenizers the runtime cannot reproduce.

use std::{fmt, path::Path};

use serde_json::Value;

use crate::{Error, read_json};

/// Tokenizer pipelines the runtime knows how to execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenizerFamily {
    /// `BertNormalizer` + `BertPreTokenizer` + `WordPiece`.
    BertWordPiece,
}

impl fmt::Display for TokenizerFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerFamily::BertWordPiece => f.write_str("bert-wordpiece"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Spec {
    pub family: TokenizerFamily,
    pub lowercase: bool,
    pub strip_accents: bool,
    pub handle_chinese_chars: bool,
    pub clean_text: bool,
    pub subword_prefix: String,
    pub max_word_len: usize,
    pub unk_token: String,
    pub unk_id: usize,
    /// `(token, id)` for every vocabulary entry.
    pub vocab: Vec<(String, usize)>,
    /// Contents of `added_tokens` entries flagged `special`.
    pub special_tokens: Vec<String>,
    /// The normalizer runs gorp-code's identifier-splitting `Replace` steps
    /// before `BertNormalizer`.
    pub split_identifiers: bool,
    /// Non-special added tokens flagged `normalized` and `single_word`,
    /// matched on the normalized text (also present in `vocab`).
    pub phrases: Vec<String>,
    /// `truncation.max_length`, if the tokenizer truncates.
    pub max_tokens: Option<usize>,
    /// model2vec's `median_token_length` over the whole vocabulary, in code
    /// points.
    pub median_token_len: usize,
}

/// The identifier-splitting steps gorp-code puts before `BertNormalizer`,
/// accepted only in exactly this form because the runtime hard-codes them.
fn identifier_split_steps(steps: &[Value]) -> bool {
    let replace = |v: &Value, kind: &str, pat: &str| {
        v["type"].as_str() == Some("Replace")
            && v["pattern"][kind].as_str() == Some(pat)
            && v["content"].as_str() == Some(" ")
    };
    steps.len() == 3
        && replace(&steps[0], "Regex", "(?<=[a-z0-9])(?=[A-Z])")
        && replace(&steps[1], "Regex", "(?<=[A-Z])(?=[A-Z][a-z])")
        && replace(&steps[2], "String", "_")
}

pub(crate) fn parse(path: &Path) -> Result<Spec, Error> {
    let tk = read_json(path)?;

    // gorp-code: Sequence[Replace x3 (identifier splitting), BertNormalizer].
    let mut split_identifiers = false;
    let mut normalizer = &tk["normalizer"];
    if normalizer["type"].as_str() == Some("Sequence")
        && let Some(steps) = normalizer["normalizers"].as_array()
        && steps.len() == 4
        && identifier_split_steps(&steps[..3])
    {
        split_identifiers = true;
        normalizer = &steps[3];
    }

    // Normalizer: absent means identity; otherwise it must be a BertNormalizer,
    // possibly wrapped in a single-element Sequence.
    let (lowercase, strip_accents, handle_chinese_chars, clean_text) = match normalizer {
        Value::Null => (false, false, false, false),
        n => {
            let n = unwrap_sequence(n, "normalizers");
            match n["type"].as_str() {
                Some("BertNormalizer") => {
                    let lowercase = n["lowercase"].as_bool().unwrap_or(true);
                    // tokenizers: strip_accents.unwrap_or(lowercase)
                    let strip = n["strip_accents"].as_bool().unwrap_or(lowercase);
                    let cjk = n["handle_chinese_chars"].as_bool().unwrap_or(true);
                    let clean = n["clean_text"].as_bool().unwrap_or(true);
                    (lowercase, strip, cjk, clean)
                }
                other => {
                    return Err(Error::UnsupportedTokenizer(format!(
                        "normalizer `{}` (only BertNormalizer is supported)",
                        other.unwrap_or("?")
                    )));
                }
            }
        }
    };

    let pre = unwrap_sequence(&tk["pre_tokenizer"], "pretokenizers");
    match pre["type"].as_str() {
        Some("BertPreTokenizer") => {}
        other => {
            return Err(Error::UnsupportedTokenizer(format!(
                "pre_tokenizer `{}` (only BertPreTokenizer is supported)",
                other.unwrap_or("none")
            )));
        }
    }

    let model = &tk["model"];
    match model["type"].as_str() {
        Some("WordPiece") => {}
        other => {
            return Err(Error::UnsupportedTokenizer(format!(
                "model `{}` (only WordPiece is supported)",
                other.unwrap_or("?")
            )));
        }
    }
    let subword_prefix = model["continuing_subword_prefix"]
        .as_str()
        .unwrap_or("##")
        .to_string();
    let max_word_len = model["max_input_chars_per_word"].as_u64().unwrap_or(100) as usize;
    let unk_token = model["unk_token"].as_str().unwrap_or("[UNK]").to_string();

    let vocab_obj = model["vocab"]
        .as_object()
        .ok_or_else(|| Error::UnsupportedTokenizer("WordPiece model has no vocab map".into()))?;
    let mut vocab: Vec<(String, usize)> = Vec::with_capacity(vocab_obj.len());
    for (token, id) in vocab_obj {
        let id = id
            .as_u64()
            .ok_or_else(|| Error::Invalid(format!("vocab id for {token:?} is not an integer")))?;
        vocab.push((token.clone(), id as usize));
    }
    vocab.sort_by_key(|(_, id)| *id);

    let unk_id = vocab
        .iter()
        .find(|(t, _)| *t == unk_token)
        .map(|(_, id)| *id)
        .ok_or_else(|| {
            Error::UnsupportedTokenizer(format!("unk_token {unk_token:?} is not in the vocabulary"))
        })?;

    // Added tokens are matched literally on the raw input at runtime, which
    // is exactly what `tokenizers` does for a special token with
    // `normalized: false`. Some exports flag `[PAD]` as normalized,
    // single-word and stripped; `tokenizers` then also matches it in
    // lowercase and only at word boundaries. The raw match covers the
    // realistic form, so those are accepted with a warning. Non-special
    // added tokens would alter ordinary text and are refused.
    let mut special_tokens = Vec::new();
    let mut approximated = Vec::new();
    let mut phrases = Vec::new();
    for t in tk["added_tokens"].as_array().into_iter().flatten() {
        let content = t["content"].as_str().unwrap_or("").to_string();
        if content.is_empty() {
            continue;
        }
        let flag = |k: &str| t[k].as_bool().unwrap_or(false);
        let special = flag("special");
        let normalized = t["normalized"].as_bool().unwrap_or(!special);
        if !special {
            // Phrase tokens: matched on the normalized text at word
            // boundaries. Their content must already be in normalized form
            // and must not hold a newline (the runtime list is one per line).
            if normalized
                && flag("single_word")
                && !flag("lstrip")
                && !flag("rstrip")
                && !content.contains('\n')
            {
                let id = t["id"].as_u64().ok_or_else(|| {
                    Error::Invalid(format!("added token {content:?} has no integer id"))
                })? as usize;
                if !vocab.iter().any(|(tok, _)| *tok == content) {
                    vocab.push((content.clone(), id));
                }
                phrases.push(content);
                continue;
            }
            return Err(Error::UnsupportedTokenizer(format!(
                "non-special added token {content:?}; only special tokens and normalized single_word phrase tokens are supported"
            )));
        }
        if normalized || flag("single_word") || flag("lstrip") || flag("rstrip") {
            approximated.push(content.clone());
        }
        special_tokens.push(content);
    }
    if !approximated.is_empty() {
        println!(
            "cargo:warning=ese-build: special tokens {approximated:?} are flagged normalized/single_word/stripped; \
             they are matched literally and case-sensitively, which differs from `tokenizers` only for \
             lowercase or mid-word occurrences"
        );
    }

    vocab.sort_by_key(|(_, id)| *id);
    if !phrases.is_empty() && !lowercase {
        return Err(Error::UnsupportedTokenizer(
            "phrase tokens need a lowercasing normalizer (their content is matched as-is)".into(),
        ));
    }
    let max_tokens = tk["truncation"]["max_length"].as_u64().map(|n| n as usize);
    // model2vec: int(np.median([len(t) for t in vocab])), added tokens included.
    let mut lens: Vec<usize> = vocab.iter().map(|(t, _)| t.chars().count()).collect();
    lens.sort_unstable();
    let median_token_len = match lens.len() {
        0 => 0,
        n if n % 2 == 1 => lens[n / 2],
        n => (lens[n / 2 - 1] + lens[n / 2]) / 2,
    };

    Ok(Spec {
        family: TokenizerFamily::BertWordPiece,
        lowercase,
        strip_accents,
        handle_chinese_chars,
        clean_text,
        subword_prefix,
        max_word_len,
        unk_token,
        unk_id,
        vocab,
        special_tokens,
        split_identifiers,
        phrases,
        max_tokens,
        median_token_len,
    })
}

/// A `Sequence` wrapping exactly one component is treated as that component.
fn unwrap_sequence<'a>(v: &'a Value, key: &str) -> &'a Value {
    if v["type"].as_str() == Some("Sequence")
        && let Some([only]) = v[key].as_array().map(Vec::as_slice)
    {
        return only;
    }
    v
}
