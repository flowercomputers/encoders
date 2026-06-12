// Unigram (SentencePiece) tokenizer runtime for ese.
//
// Three semantics facts extracted from the tokenizers crate oracle
// (tokenizers-0.21.4/src/models/unigram/model.rs) — these determine parity:
//
// 1. UNK penalty and unk_score formula:
//    model.rs:70   static K_UNK_PENALTY: f64 = 10.0;
//    model.rs:161  let unk_score = self.min_score - K_UNK_PENALTY;
//    (encode_optimized:262 mirrors this)
//    unk_score = (minimum log-prob score across all vocab entries) minus 10.0.
//    An UNK node is only inserted when no single-character token matches the
//    current byte position (has_single_node == false). It spans exactly one
//    Unicode scalar (mblen bytes).
//
// 2. fuse_unk default when absent from JSON:
//    serialization.rs:51   let mut byte_fallback: bool = false;  (not read from JSON)
//    model.rs:125           let fuse_unk = true;                  (hardcoded in from())
//    fuse_unk is NOT read from JSON; it is always true. Consecutive UNK nodes
//    are fused into a single token in the output. We implement fuse_unk=true.
//
// 3. Viterbi tie-breaking (encode_optimized, model.rs:281-286):
//    if target_node.starts_at.is_none() || candidate_best_path_score > target_node.best_path_score
//    Strict >: the FIRST candidate placed at a given end position wins on equal score.
//    We use the same strict-greater condition.

use crate::lookup;
use spm_precompiled::Precompiled;

// The charsmap blob decoded at build time from the Precompiled normalizer.
// For WordPiece builds this is a 0-byte dummy file; IS_UNIGRAM=false so
// this static is compiled-out by the const branch in lib.rs encode_single.
static CHARSMAP_RAW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/charsmap.bin"));

// Lazily initialized Precompiled charsmap normalizer.
// std::sync::OnceLock is stable since Rust 1.70.
static CHARSMAP: std::sync::OnceLock<Precompiled> = std::sync::OnceLock::new();

fn get_charsmap() -> &'static Precompiled {
    CHARSMAP.get_or_init(|| {
        Precompiled::from(CHARSMAP_RAW)
            .expect("charsmap.bin failed to parse — corrupt build artifact")
    })
}

/// Encode a single text using the Unigram (Viterbi) path.
/// Called from lib.rs when IS_UNIGRAM=true.
pub fn encode_single(text: &str) -> [f32; lookup::DIMENSIONS] {
    let normalized = normalize(text);
    // If the normalized string is empty (e.g. after stripping whitespace-only input),
    // the tokenizers pipeline returns an empty token list and thus a zero vector.
    // This matches oracle behavior: tokenizers returns [] for "" and "   ".
    if normalized.is_empty() {
        return [0.0f32; lookup::DIMENSIONS];
    }
    let with_metaspace = apply_metaspace(&normalized);
    let weight_rows = viterbi_segment(&with_metaspace);
    pool_and_mean(&weight_rows)
}

// ── Normalizer ────────────────────────────────────────────────────────────────

/// Apply the compiled normalizer chain in order:
///   1. Precompiled charsmap
///   2. Collapse runs of 2+ spaces to one space   (Regex Replace " {2,}" → " ")
///   3. Single-char pad replacements               (String Replace "X" → " X ")
///   4. Collapse any whitespace run to one space   (Regex Replace "\\s+" → " ")
///   5. Strip leading/trailing whitespace
fn normalize(text: &str) -> String {
    // Step 1: Precompiled charsmap — handles Unicode normalization for SentencePiece.
    let after_charsmap = get_charsmap().normalize_string(text);

    // Step 2: Collapse runs of 2+ spaces to one space.
    // Regex Replace{" {2,}" → " "}: only ASCII spaces, no other whitespace.
    let after_space_collapse = if lookup::COLLAPSE_SPACES {
        collapse_space_runs(&after_charsmap)
    } else {
        after_charsmap
    };

    // Step 3: Single-char pad replacements ("X" → " X ").
    // Each character in PAD_CHARS is replaced by space-char-space in a single pass.
    let after_pad = if !lookup::PAD_CHARS.is_empty() {
        apply_pad_chars(&after_space_collapse)
    } else {
        after_space_collapse
    };

    // Step 4: Collapse any Unicode whitespace run to one space.
    // Regex Replace{"\\s+" → " "}: any whitespace char qualifies.
    let after_ws_collapse = if lookup::COLLAPSE_WHITESPACE {
        collapse_whitespace_runs(&after_pad)
    } else {
        after_pad
    };

    // Step 5: Strip leading/trailing whitespace.
    if lookup::STRIP_LEFT || lookup::STRIP_RIGHT {
        let s = after_ws_collapse.as_str();
        let stripped = if lookup::STRIP_LEFT && lookup::STRIP_RIGHT {
            s.trim()
        } else if lookup::STRIP_LEFT {
            s.trim_start()
        } else {
            s.trim_end()
        };
        stripped.to_string()
    } else {
        after_ws_collapse
    }
}

/// Collapse runs of two or more ASCII spaces to a single space.
/// Non-space characters pass through unchanged.
fn collapse_space_runs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space_count = 0usize;
    for c in s.chars() {
        if c == ' ' {
            space_count += 1;
            if space_count == 1 {
                // Emit the first space of each run.
                out.push(' ');
            }
            // Skip additional spaces in the same run.
        } else {
            space_count = 0;
            out.push(c);
        }
    }
    out
}

/// Replace each character in PAD_CHARS with " <char> " (space-char-space).
/// A single character-walk emits the replacement for matching chars.
fn apply_pad_chars(s: &str) -> String {
    let pad_set = lookup::PAD_CHARS;
    let mut out = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        // Linear scan is fine: PAD_CHARS has 33 entries and is a compile-time constant.
        if pad_set.contains(&c) {
            out.push(' ');
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// Collapse any run of Unicode whitespace characters to a single ASCII space.
fn collapse_whitespace_runs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_whitespace = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !in_whitespace {
                out.push(' ');
                in_whitespace = true;
            }
            // Skip additional whitespace in the same run.
        } else {
            in_whitespace = false;
            out.push(c);
        }
    }
    out
}

// ── Metaspace pre-tokenizer ───────────────────────────────────────────────────

/// Apply Metaspace{replacement: ▁, prepend_scheme: always, split: false}.
///
/// Semantics (tokenizers pre_tokenizers/metaspace.rs, PrependScheme::Always + split=false):
///   - Replace every ASCII space with ▁ (U+2581, METASPACE_CHAR).
///   - If the result does not already start with ▁, prepend one ▁.
///   - split=false: do NOT split on ▁; return the entire string as one chunk.
///
/// The entire string is passed as one unit to the Viterbi segmenter.
fn apply_metaspace(s: &str) -> String {
    // Replace spaces with ▁.
    let mut out = String::with_capacity(s.len() + lookup::METASPACE_CHAR.len_utf8());
    for c in s.chars() {
        if c == ' ' {
            out.push(lookup::METASPACE_CHAR);
        } else {
            out.push(c);
        }
    }
    // Prepend ▁ if not already present (METASPACE_PREPEND=true for potion).
    if lookup::METASPACE_PREPEND && !out.starts_with(lookup::METASPACE_CHAR) {
        let mut prefixed = String::with_capacity(out.len() + lookup::METASPACE_CHAR.len_utf8());
        prefixed.push(lookup::METASPACE_CHAR);
        prefixed.push_str(&out);
        prefixed
    } else {
        out
    }
}

// ── Viterbi segmenter ─────────────────────────────────────────────────────────

/// Segment `sentence` using Viterbi and return the weight rows for pooling.
///
/// Algorithm: the "optimized" variant from tokenizers model.rs encode_optimized.
/// dp[i].best_path_score = best cumulative log-prob ending at byte position i.
/// dp[i].starts_at       = start byte position of the best token ending at i.
/// dp[i].weight          = pointer to the weight row for that token.
///
/// UNK: when no single-char vocab match exists, insert UNK (UNK_ID) at that position
/// with score = min_vocab_score - K_UNK_PENALTY (10.0).
///
/// Tie-breaking: strict > (first candidate wins on equal score, per oracle).
///
/// fuse_unk=true: consecutive UNK-id tokens are collapsed to one UNK weight row
/// in the output (matching tokenizers encode_optimized fuse logic).
///
/// Returns a Vec of static Param references for pooling.
fn viterbi_segment(sentence: &str) -> Vec<&'static lookup::Param> {
    if sentence.is_empty() {
        return vec![];
    }

    let size = sentence.len();

    // unk_score = min_score_in_vocab - K_UNK_PENALTY (10.0).
    // MIN_SCORE is the most-negative vocab log-prob score, emitted at build time.
    let unk_score = lookup::MIN_SCORE - 10.0f32;

    // dp[i]: best path ending at byte position i.
    // Positions that are not yet reachable have starts_at = None.
    struct Node {
        best_path_score: f32,
        starts_at: Option<usize>,
        weight: Option<&'static lookup::Param>,
        is_unk: bool,
    }
    let make_node = || Node {
        best_path_score: 0.0,
        starts_at: None,
        weight: None,
        is_unk: false,
    };

    let mut dp: Vec<Node> = (0..=size).map(|_| make_node()).collect();

    let mut starts_at = 0usize;
    while starts_at < size {
        let score_here = dp[starts_at].best_path_score;

        // Byte length of the first Unicode scalar at this position.
        let mblen = sentence[starts_at..]
            .chars()
            .next()
            .unwrap()
            .len_utf8();

        let mut has_single_node = false;
        let remaining = &sentence[starts_at..];
        let mut char_count = 0usize;
        let mut byte_end = 0usize;

        // Enumerate all vocab tokens that are a prefix of sentence[starts_at..].
        // Extend byte_end one char at a time, capped at MAX_TOKEN_CHARS.
        for c in remaining.chars() {
            char_count += 1;
            byte_end += c.len_utf8();

            let candidate = &remaining[..byte_end];
            let key_pos = starts_at + byte_end;

            if let Some((weight, score)) = lookup::lookup_with_score(candidate) {
                let candidate_score = score + score_here;
                let target = &mut dp[key_pos];
                // Strict > ties broken by first (shortest) candidate.
                if target.starts_at.is_none() || candidate_score > target.best_path_score {
                    target.best_path_score = candidate_score;
                    target.starts_at = Some(starts_at);
                    target.weight = Some(weight);
                    target.is_unk = false;
                }
                if !has_single_node && byte_end == mblen {
                    has_single_node = true;
                }
            }

            if char_count >= lookup::MAX_TOKEN_CHARS {
                break;
            }
        }

        // If no single-char vocab match exists, insert an UNK node.
        if !has_single_node {
            let target = &mut dp[starts_at + mblen];
            let candidate_score = unk_score + score_here;
            if target.starts_at.is_none() || candidate_score > target.best_path_score {
                target.best_path_score = candidate_score;
                target.starts_at = Some(starts_at);
                target.weight = Some(&lookup::UNK);
                target.is_unk = true;
            }
        }

        starts_at += mblen;
    }

    // Backtrack from end to start.
    let mut results: Vec<&'static lookup::Param> = vec![];
    let mut ends_at = size;
    let mut in_unk_run = false;

    while ends_at > 0 {
        let node = &dp[ends_at];
        let s_at = node.starts_at.expect("viterbi backtrack: unreachable position");
        let is_unk = node.is_unk;

        if is_unk {
            // fuse_unk=true: defer UNK emission until the run ends.
            in_unk_run = true;
        } else {
            if in_unk_run {
                // Flush fused UNK run: emit one UNK weight row — unless SKIP_UNK,
                // which drops UNK rows from pooling to match model2vec.StaticModel behavior.
                if !lookup::SKIP_UNK {
                    results.push(&lookup::UNK);
                }
                in_unk_run = false;
            }
            if let Some(w) = node.weight {
                results.push(w);
            }
        }

        ends_at = s_at;
    }

    // Flush any UNK run that reaches the beginning of the sentence.
    if in_unk_run && !lookup::SKIP_UNK {
        results.push(&lookup::UNK);
    }

    results.reverse();
    results
}

// ── Pool and mean ─────────────────────────────────────────────────────────────

/// Mean-pool the weight rows for the Viterbi token sequence, then optionally L2-normalize.
/// Uses the same dequantization formula as wordpiece.rs: QUANT_MIN + (p as f32) * QUANT_SCALE.
fn pool_and_mean(weight_rows: &[&'static lookup::Param]) -> [f32; lookup::DIMENSIONS] {
    let mut vector = [0.0f32; lookup::DIMENSIONS];
    let count = weight_rows.len();
    if count == 0 {
        // Empty input after normalization — return zero vector.
        return vector;
    }
    for row in weight_rows {
        crate::wordpiece::accumulate(&mut vector, row);
    }
    let scale = 1.0 / count as f32;
    for v in &mut vector {
        *v *= scale;
    }
    if lookup::NORMALIZE_OUTPUT {
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
    }
    vector
}
