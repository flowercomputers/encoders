#[inline(always)]
pub fn accumulate(vector: &mut [f32; crate::DIMENSIONS], param: &crate::lookup::Param) {
    for (v, &p) in vector.iter_mut().zip(param.iter()) {
        *v += crate::lookup::QUANT_MIN + (p as f32) * crate::lookup::QUANT_SCALE;
    }
}

#[inline]
pub fn wordpiece_accumulate(
    word: &str,
    vector: &mut [f32; crate::DIMENSIONS],
    token_count: &mut u32,
    wp_buf: &mut String,
) {
    if word.chars().count() > crate::lookup::MAX_WORD_LEN {
        // Word too long — falls back to UNK for the entire word.
        // When SKIP_UNK, neither accumulate nor count.
        if !crate::lookup::SKIP_UNK {
            *token_count += 1;
            accumulate(vector, &crate::lookup::UNK);
        }
        return;
    }

    // When SKIP_UNK, we must not emit partial tokens if the word later fails.
    // WordPiece semantics: if any sub-word position is unresolvable, the whole
    // word becomes UNK — partial tokens already emitted in that pass are discarded.
    // Solution: accumulate into a scratch vector; commit to the caller only on
    // full success. When SKIP_UNK=false, accumulate directly (unchanged behavior).
    if crate::lookup::SKIP_UNK {
        let mut scratch = [0.0f32; crate::DIMENSIONS];
        let mut scratch_count: u32 = 0;
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let mut matched = false;
            while end > start {
                while end < word.len() && !word.is_char_boundary(end) {
                    end += 1;
                }
                let embedding = if start == 0 {
                    crate::lookup::lookup(&word[start..end])
                } else {
                    wp_buf.clear();
                    wp_buf.push_str(crate::lookup::CONTINUING_PREFIX);
                    wp_buf.push_str(&word[start..end]);
                    crate::lookup::lookup(wp_buf.as_str())
                };
                if let Some(emb) = embedding {
                    scratch_count += 1;
                    accumulate(&mut scratch, emb);
                    start = end;
                    matched = true;
                    break;
                }
                end -= 1;
                while end > start && !word.is_char_boundary(end) {
                    end -= 1;
                }
            }
            if !matched {
                // Whole word is UNK — discard scratch and skip (SKIP_UNK=true).
                return;
            }
        }
        // Full word successfully segmented: commit scratch to caller's vector.
        *token_count += scratch_count;
        for (v, s) in vector.iter_mut().zip(scratch.iter()) {
            *v += s;
        }
        return;
    }

    // SKIP_UNK=false path: original behavior, accumulate directly.
    let mut start = 0;
    while start < word.len() {
        let mut end = word.len();
        let mut matched = false;
        while end > start {
            while end < word.len() && !word.is_char_boundary(end) {
                end += 1;
            }
            let embedding = if start == 0 {
                crate::lookup::lookup(&word[start..end])
            } else {
                wp_buf.clear();
                wp_buf.push_str(crate::lookup::CONTINUING_PREFIX);
                wp_buf.push_str(&word[start..end]);
                crate::lookup::lookup(wp_buf.as_str())
            };
            if let Some(emb) = embedding {
                *token_count += 1;
                accumulate(vector, emb);
                start = end;
                matched = true;
                break;
            }
            end -= 1;
            while end > start && !word.is_char_boundary(end) {
                end -= 1;
            }
        }
        if !matched {
            // No sub-word match — whole word is UNK.
            *token_count += 1;
            accumulate(vector, &crate::lookup::UNK);
            return;
        }
    }
}
