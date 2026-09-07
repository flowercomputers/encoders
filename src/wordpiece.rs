/// Special-token vectors (UNK/CLS/SEP) are stored as exact f32.
#[inline(always)]
pub fn accumulate(vector: &mut [f32; crate::DIMENSIONS], param: &[f32; crate::DIMENSIONS]) {
    for (v, &p) in vector.iter_mut().zip(param.iter()) {
        *v += p;
    }
}

/// Vocab rows are dequantized with the slot's per-row min/scale.
#[inline(always)]
fn accumulate_slot(vector: &mut [f32; crate::DIMENSIONS], slot: &crate::lookup::Slot) {
    if crate::lookup::DTYPE_SIZE == 4 {
        for (v, &p) in vector.iter_mut().zip(slot.weight.iter()) {
            *v += p as f32;
        }
    } else {
        for (v, &p) in vector.iter_mut().zip(slot.weight.iter()) {
            *v += slot.min + (p as f32) * slot.scale;
        }
    }
}

/// MAX_TOKENS = 0 means no truncation.
#[inline(always)]
fn at_cap(produced: u32) -> bool {
    crate::lookup::MAX_TOKENS != 0 && produced as usize >= crate::lookup::MAX_TOKENS
}

/// An unknown word produces a single UNK token. model2vec models drop UNK
/// tokens before pooling (it still counts toward the truncation cap, since
/// the reference implementation truncates before removing UNK ids).
#[inline(always)]
fn unk(vector: &mut [f32; crate::DIMENSIONS], token_count: &mut u32, produced: &mut u32) {
    *produced += 1;
    if !crate::lookup::DROP_UNK {
        *token_count += 1;
        accumulate(vector, &crate::lookup::UNK);
    }
}

#[inline]
pub fn wordpiece_accumulate(
    word: &str,
    vector: &mut [f32; crate::DIMENSIONS],
    token_count: &mut u32,
    produced: &mut u32,
    wp_buf: &mut String,
    pieces: &mut Vec<&'static crate::lookup::Slot>,
) {
    if at_cap(*produced) {
        return;
    }
    if word.chars().count() > crate::lookup::MAX_WORD_LEN {
        unk(vector, token_count, produced);
        return;
    }
    // WordPiece emits UNK for the whole word if any piece fails to match, so
    // pieces are buffered and only accumulated once the full word is matched.
    pieces.clear();
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
                pieces.push(emb);
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
            unk(vector, token_count, produced);
            return;
        }
    }
    for emb in pieces.iter() {
        if at_cap(*produced) {
            return;
        }
        *produced += 1;
        *token_count += 1;
        accumulate_slot(vector, emb);
    }
}
