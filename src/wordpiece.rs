/// Accumulate a quantized weight row into `vector` using the row's own (min, scale) params.
///
/// quant builds: min and scale are the per-row params stored in the Slot struct
///   (or the special-token consts UNK_MIN/UNK_SCALE, CLS_MIN/CLS_SCALE, SEP_MIN/SEP_SCALE).
/// f32 build:   min=0.0, scale=1.0 (identity) — the compiler folds this to a plain add,
///   producing code identical to the original behavior.
#[inline(always)]
pub fn accumulate(
    vector: &mut [f32; crate::DIMENSIONS],
    param: &crate::lookup::Param,
    min: f32,
    scale: f32,
) {
    for (v, &p) in vector.iter_mut().zip(param.iter()) {
        *v += min + (p as f32) * scale;
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
            // UNK special token: use its per-row params (consts from model_constants.rs).
            // f32 build: UNK_MIN=0.0, UNK_SCALE=1.0 → identity, same as before.
            accumulate(vector, &crate::lookup::UNK, crate::lookup::UNK_MIN, crate::lookup::UNK_SCALE);
        }
        return;
    }

    // When SKIP_UNK, we must not emit partial tokens if the word later fails.
    // WordPiece semantics: if any sub-word position is unresolvable, the whole
    // word becomes UNK — partial tokens already emitted in that pass are discarded.
    // Solution: accumulate into a scratch vector; commit to the caller only on
    // full success. When SKIP_UNK=false, use the same scratch path but commit
    // one UNK on mid-word failure (matching HF WordPiece whole-word-UNK semantics).
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
            let slot = if start == 0 {
                crate::lookup::lookup_slot(&word[start..end])
            } else {
                wp_buf.clear();
                wp_buf.push_str(crate::lookup::CONTINUING_PREFIX);
                wp_buf.push_str(&word[start..end]);
                crate::lookup::lookup_slot(wp_buf.as_str())
            };
            if let Some(s) = slot {
                scratch_count += 1;
                // quant builds: read per-row params from the slot.
                // f32 build: row_min/row_scale fields absent; pass (0.0, 1.0) identity consts.
                #[cfg(any(feature = "quant-8", feature = "quant-16"))]
                accumulate(&mut scratch, &s.weight, s.row_min, s.row_scale);
                #[cfg(not(any(feature = "quant-8", feature = "quant-16")))]
                accumulate(&mut scratch, &s.weight, 0.0, 1.0);
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
            // Mid-word failure: the whole word maps to UNK.
            if crate::lookup::SKIP_UNK {
                // SKIP_UNK=true: discard scratch and contribute nothing.
                return;
            } else {
                // SKIP_UNK=false (default): contribute exactly one UNK for the whole word.
                // Discards any partial pieces already matched (HF WordPiece whole-word-UNK).
                *token_count += 1;
                accumulate(vector, &crate::lookup::UNK, crate::lookup::UNK_MIN, crate::lookup::UNK_SCALE);
                return;
            }
        }
    }
    // Full word successfully segmented: commit scratch to caller's vector.
    *token_count += scratch_count;
    for (v, s) in vector.iter_mut().zip(scratch.iter()) {
        *v += s;
    }
}
