// BPE (ByteLevel) tokenizer runtime for ese.
//
// Three semantics facts extracted from the tokenizers-0.21.4 vendored source —
// these determine parity with the oracle; also recorded here per G1:
//
// FACT 1 — Merge rule (models/bpe/word.rs:162-250, merge_all):
//   Merges are applied by a min-heap ordered (ascending rank, ascending position).
//   word.rs Ord (lines 28-36): the Merge struct uses "other.rank.cmp(&self.rank).then(
//   other.pos.cmp(&self.pos))" which makes the BinaryHeap a min-heap by rank, then pos.
//   Algorithm: seed the heap with all adjacent (i, i+1) pairs that have a merge rule.
//   Each pop: skip if symbol at top.pos has been removed (len==0) or is last (next==-1).
//   Verify the pair is still current: merges.get(&(symbols[top.pos].c, symbols[next].c))
//   must yield new_id == top.new_id (stale entries skip, word.rs:199-205).
//   On merge: left.c = new_id, left.len += right.len, left.next = right.next.
//   Mark right.len = 0 (removed). Update right.next.prev = left_pos (word.rs:213-215).
//   Push new candidates (prev, current) and (current, next) if in merge map (word.rs:218-244).
//   Finally: symbols.retain(|s| s.len != 0) (word.rs:249).
//   We implement this exactly with BinaryHeap<Reverse<MergeEntry>>.
//
// FACT 2 — ByteLevel alphabet, prefix, and split (pre_tokenizers/byte_level.rs):
//   bytes_char() (byte_level.rs:15-38): maps all 256 byte values to printable Unicode chars.
//   Identity bytes (map to themselves as chars): b'!'..=b'~' (0x21-0x7E, 94 bytes),
//     b'\xA1'..=b'\xAC' (12 bytes), b'\xAE'..=b'\xFF' (82 bytes). Total: 188 bytes.
//   Non-identity bytes: remaining 68 bytes (0x00-0x20 = 33, 0x7F, 0xAD, ...) → iterated
//     over 0..=255 in ascending order, each assigned U+0100+n (n=0,1,…,67).
//   Key mapping: space (0x20) → U+0120 = Ġ  (the visible space marker in BPE tokens).
//   Split regex (byte_level.rs:44, the GPT-2 pattern):
//     "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
//   add_prefix_space (byte_level.rs:60, 123-125): when true, prepend " " to input if it
//     doesn't start with ' ', so the first word gets the Ġ prefix like subsequent words.
//     Controlled by IS_BPE_ADD_PREFIX_SPACE (build-time const from tokenizer.json).
//   Per-chunk encoding: each regex match chunk → each byte of each UTF-8 char → BYTES_CHAR
//     → a new string of Unicode chars from the 256-char alphabet, fed to BPE model.
//
// FACT 3 — UNK / byte-fallback (models/bpe/model.rs:382-466, merge_word):
//   ByteLevel BPE has all 256 byte values in its alphabet (one single-byte vocab entry per
//   byte), so no character ever maps to UNK for a well-formed ByteLevel BPE vocab — every
//   byte in the ByteLevel-encoded chunk string IS in the vocab. unk_token present in
//   all-distilroberta-v1's tokenizer.json but never emitted in practice. byte_fallback=false
//   (field absent in tokenizer.json → false per model.rs:52). ignore_merges=false (absent
//   → false per model.rs:55). We guard both at build time and reject non-null dropout too.
//   SKIP_UNK: if a single-byte token is somehow missing (malformed vocab), we skip it.
//
// Merge table representation (documented in src/lookup.rs merge-table section):
//   merges.bin:       [u8; MERGE_TABLE_SIZE * 8] — value per slot: rank:u32le + new_id:u32le
//   merge_seeds.bin:  [u32; MERGE_NUM_BUCKETS] — PHF bucket seeds (same hash as vocab PHF)
//   merge_verify.bin: [u64; MERGE_TABLE_SIZE] — verify hashes for the merge PHF
//   Key: (left_vocab_id as u64) << 32 | right_vocab_id as u64, encoded as 8-byte LE u64.
//   lookup_merge(left_id, right_id) → Option<(rank, new_id)>.
//   Vocab token IDs: slot_ids.bin ([u32; TABLE_SIZE]) maps PHF slot_index → original vocab_id.
//   lookup::lookup_vocab_id(token_str) uses slot_ids.bin to return the vocab_id.

use crate::lookup;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ── ByteLevel alphabet table (byte → Unicode char) ──────────────────────────

static BYTE_TO_CHAR_TABLE: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();

fn get_byte_to_char() -> &'static [char; 256] {
    BYTE_TO_CHAR_TABLE.get_or_init(build_byte_to_char)
}

// Reproduces bytes_char() from byte_level.rs:15-38.
// Identity bytes: b'!'..=b'~', b'\xA1'..=b'\xAC', b'\xAE'..=b'\xFF' → char with same codepoint.
// Non-identity bytes: iterated 0..=255 ascending, assigned U+0100+n in insertion order.
fn build_byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut is_identity = [false; 256];

    // Identity ranges (byte_level.rs lines 17-19):
    for b in b'!'..=b'~' { is_identity[b as usize] = true; }
    for b in b'\xA1'..=b'\xAC' { is_identity[b as usize] = true; }
    for b in b'\xAE'..=b'\xFF' { is_identity[b as usize] = true; }

    // Identity bytes map to themselves (byte value == char codepoint).
    for (b, &id) in is_identity.iter().enumerate() {
        if id {
            // Safety: 0x21..=0x7E and 0xA1..=0xFF are valid Unicode scalars.
            table[b] = unsafe { char::from_u32_unchecked(b as u32) };
        }
    }

    // Non-identity bytes: 0..=255 in ascending order → U+0100, U+0101, …
    // (byte_level.rs:24-30: for b in 0..=255 { if !identity { cs.push(2^8 + n); n += 1; } })
    let mut n: u32 = 0;
    for b in 0u8..=255u8 {
        if !is_identity[b as usize] {
            // Safety: 256..=323 are valid Unicode scalars (well below U+D800).
            table[b as usize] = unsafe { char::from_u32_unchecked(256 + n) };
            n += 1;
        }
    }
    table
}

// ── GPT-2 / ByteLevel split regex ────────────────────────────────────────────
// Exact pattern from byte_level.rs:44, compiled with Unicode + lookahead support.
// fancy_regex is used because the GPT-2 pattern contains a negative lookahead (?!\S)
// which is not supported by the `regex` crate. fancy_regex handles it correctly.
use fancy_regex::Regex as FancyRegex;

static BYTELEVEL_RE: std::sync::OnceLock<FancyRegex> = std::sync::OnceLock::new();

fn get_bytelevel_re() -> &'static FancyRegex {
    BYTELEVEL_RE.get_or_init(|| {
        FancyRegex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        )
        .expect("BPE ByteLevel GPT-2 regex compile failed")
    })
}

// ── Merge PHF (pair key → (rank, new_id)) ────────────────────────────────────

static MERGE_SEEDS_RAW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/merge_seeds.bin"));
static MERGES_RAW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/merges.bin"));
static MERGE_VERIFY_RAW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/merge_verify.bin"));

// Same hash polynomial as vocab PHF in lookup.rs — must stay in sync.
#[inline(always)]
fn phf_hash(key: &[u8], seed: u64) -> u64 {
    let mut h = seed ^ 0x517cc1b727220a95;
    for &b in key {
        h = (h ^ b as u64).wrapping_mul(0x2127599bf4325c37);
    }
    h ^ (h >> 32)
}

/// Look up a (left_vocab_id, right_vocab_id) merge pair.
/// Returns Some((rank, new_id)) if the pair is in the merge table.
/// Key encoding: 8-byte LE u64 = left_id << 32 | right_id.
#[inline(always)]
fn lookup_merge(left_id: u32, right_id: u32) -> Option<(u32, u32)> {
    let key_u64: u64 = (left_id as u64) << 32 | right_id as u64;
    let key_bytes = key_u64.to_le_bytes();
    let num_buckets = lookup::MERGE_NUM_BUCKETS;
    let table_mask = lookup::MERGE_TABLE_MASK;
    unsafe {
        let bucket = (phf_hash(&key_bytes, 0) as usize) % num_buckets;
        let seed_offset = bucket * 4;
        let seed = u32::from_le_bytes(
            MERGE_SEEDS_RAW[seed_offset..seed_offset + 4]
                .try_into()
                .unwrap_unchecked(),
        ) as u64;
        let slot = (phf_hash(&key_bytes, seed) as usize) & table_mask;
        // Verify hash
        let verify_offset = slot * 8;
        let stored_verify = u64::from_le_bytes(
            MERGE_VERIFY_RAW[verify_offset..verify_offset + 8]
                .try_into()
                .unwrap_unchecked(),
        );
        if stored_verify != phf_hash(&key_bytes, u64::MAX) {
            return None;
        }
        // Read (rank, new_id) value
        let data_offset = slot * 8;
        let rank = u32::from_le_bytes(
            MERGES_RAW[data_offset..data_offset + 4]
                .try_into()
                .unwrap_unchecked(),
        );
        let new_id = u32::from_le_bytes(
            MERGES_RAW[data_offset + 4..data_offset + 8]
                .try_into()
                .unwrap_unchecked(),
        );
        Some((rank, new_id))
    }
}

// ── BPE Symbol list ───────────────────────────────────────────────────────────

/// A symbol in the BPE symbol linked-list. Mirrors Symbol in word.rs.
/// c: current token id; prev/next: linked-list pointers (–1 = none); len: byte span in
/// the ByteLevel-encoded chunk string (0 = removed by a merge).
#[derive(Clone, Copy)]
struct Symbol {
    c: u32,
    prev: isize,
    next: isize,
    len: usize,
}

/// A pending merge candidate. We use Reverse<MergeEntry> with BinaryHeap (max-heap) to get
/// a min-heap ordered by ascending rank, then ascending pos — matching word.rs Ord semantics.
#[derive(Eq, PartialEq)]
struct MergeEntry {
    rank: u32,
    pos: usize,
    new_id: u32,
}

impl Ord for MergeEntry {
    // Natural order: smaller rank = higher priority = greater in this ordering.
    // Combined with Reverse<>, BinaryHeap pops lowest rank first.
    // On equal rank: smaller pos = higher priority.
    // This mirrors word.rs:28-36 (Merge::cmp reverses both comparisons for a max-heap,
    // which is equivalent to our Reverse<MergeEntry> on this natural order).
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank
            .cmp(&other.rank)
            .then(self.pos.cmp(&other.pos))
    }
}
impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── BPE merge_all ─────────────────────────────────────────────────────────────

/// Apply BPE merges to a symbol list, in-place. Mirrors merge_all() in word.rs:162-250.
/// Requires: symbols have valid prev/next linked-list pointers, no dropout (always None).
fn bpe_merge_all(symbols: &mut Vec<Symbol>) {
    // Seed the heap with all adjacent pairs that are valid merges (word.rs:167-178).
    let mut heap: BinaryHeap<Reverse<MergeEntry>> = BinaryHeap::new();
    for i in 0..symbols.len().saturating_sub(1) {
        if let Some((rank, new_id)) = lookup_merge(symbols[i].c, symbols[i + 1].c) {
            heap.push(Reverse(MergeEntry { rank, pos: i, new_id }));
        }
    }

    while let Some(Reverse(top)) = heap.pop() {
        // Skip if symbol at top.pos was merged away (word.rs:187-188).
        if symbols[top.pos].len == 0 {
            continue;
        }
        // Skip if top.pos is the last symbol (word.rs:190-192).
        if symbols[top.pos].next == -1 {
            continue;
        }
        let next_pos = symbols[top.pos].next as usize;
        let right_c = symbols[next_pos].c;

        // Verify still current: pair must yield top.new_id (word.rs:196-205 stale-skip).
        match lookup_merge(symbols[top.pos].c, right_c) {
            Some((_, nid)) if nid == top.new_id => {}
            _ => continue,
        }

        // Merge: update left symbol (word.rs:207-211).
        let right = symbols[next_pos];
        symbols[top.pos].c = top.new_id;
        symbols[top.pos].len += right.len;
        symbols[top.pos].next = right.next;
        // Mark right as removed (word.rs:210: symbols[next_pos].len = 0).
        symbols[next_pos].len = 0;
        // Update right.next.prev = left_pos (word.rs:213-215).
        if right.next >= 0 && (right.next as usize) < symbols.len() {
            symbols[right.next as usize].prev = top.pos as isize;
        }

        // Push new candidate (prev, current) if in merge map (word.rs:218-228).
        let cur_c = symbols[top.pos].c;
        let cur_prev = symbols[top.pos].prev;
        if cur_prev >= 0 {
            let prev_c = symbols[cur_prev as usize].c;
            if let Some((rank, new_id)) = lookup_merge(prev_c, cur_c) {
                heap.push(Reverse(MergeEntry { rank, pos: cur_prev as usize, new_id }));
            }
        }
        // Push new candidate (current, next) if in merge map (word.rs:230-244).
        let cur_next = symbols[top.pos].next;
        if cur_next >= 0 {
            let next2_c = symbols[cur_next as usize].c;
            if let Some((rank, new_id)) = lookup_merge(cur_c, next2_c) {
                heap.push(Reverse(MergeEntry { rank, pos: top.pos, new_id }));
            }
        }
    }

    // Retain only non-removed symbols (word.rs:249).
    symbols.retain(|s| s.len != 0);
}

// ── ByteLevel pre-tokenize ────────────────────────────────────────────────────

/// ByteLevel pre-tokenization: split with GPT-2 regex, then byte-map each chunk.
///
/// Faithfully implements byte_level.rs pre_tokenize() (FACT 2):
///   1. If IS_BPE_ADD_PREFIX_SPACE and input doesn't start with ' ', prepend " ".
///   2. Split using the GPT-2 regex (every char covered by some match; no gaps).
///   3. For each match, map every byte of every UTF-8 char through BYTES_CHAR table,
///      producing a new string of chars from the 256-char ByteLevel alphabet.
fn bytelevel_pretokenize(text: &str) -> Vec<String> {
    let re = get_bytelevel_re();
    let byte_to_char = get_byte_to_char();

    // Step 1: optional prefix space (byte_level.rs:123-125).
    let owned;
    let effective: &str = if lookup::IS_BPE_ADD_PREFIX_SPACE && !text.starts_with(' ') {
        owned = format!(" {text}");
        &owned
    } else {
        text
    };

    // Steps 2+3: find_iter gives all non-overlapping matches (GPT-2 regex is exhaustive).
    // fancy_regex::find_iter returns Result<Match> items; we unwrap each.
    let mut chunks = Vec::new();
    for m in re.find_iter(effective) {
        let m = m.expect("BPE ByteLevel regex match error");
        let chunk_str = m.as_str();
        let mut encoded = String::with_capacity(chunk_str.len() * 2);
        for &b in chunk_str.as_bytes() {
            encoded.push(byte_to_char[b as usize]);
        }
        if !encoded.is_empty() {
            chunks.push(encoded);
        }
    }
    chunks
}

// ── Encode one BPE chunk ──────────────────────────────────────────────────────

/// Tokenize one ByteLevel-encoded chunk through BPE merges, accumulate weight rows.
///
/// Steps (mirrors merge_word in model.rs:382-466, then word_to_tokens):
///   1. Build symbol list: each Unicode char in the ByteLevel-encoded chunk is one initial
///      token. Look it up in the vocab PHF via lookup_vocab_id to get its numeric id (needed
///      for merge pair keying). Record byte offsets for substring reconstruction after merges.
///   2. Run bpe_merge_all() — applies merges by ascending rank per FACT 1.
///   3. For each surviving symbol, use its byte-span (tracked via Symbol.len) to reconstruct
///      the token string from the encoded chunk. Look it up via lookup_slot for the weight row.
///   4. Accumulate the weight row; increment token_count.
///
/// If a single-char token is missing (malformed vocab): emit UNK unless SKIP_UNK (FACT 3).
fn encode_chunk(
    chunk: &str,
    vector: &mut [f32; lookup::DIMENSIONS],
    token_count: &mut usize,
) {
    if chunk.is_empty() {
        return;
    }

    // Step 1: build symbol list.
    let mut symbols: Vec<Symbol> = Vec::with_capacity(chunk.len());
    let mut byte_pos = 0usize;

    for ch in chunk.chars() {
        let ch_len = ch.len_utf8();
        let char_str = &chunk[byte_pos..byte_pos + ch_len];
        let sym_idx = symbols.len();

        match lookup::lookup_vocab_id(char_str) {
            Some(vocab_id) => {
                let prev = if sym_idx == 0 { -1isize } else { sym_idx as isize - 1 };
                symbols.push(Symbol { c: vocab_id, prev, next: -1isize, len: ch_len });
                if sym_idx > 0 {
                    symbols[sym_idx - 1].next = sym_idx as isize;
                }
            }
            None => {
                // Single-byte ByteLevel token not in vocab (should not happen, FACT 3).
                if !lookup::SKIP_UNK {
                    *token_count += 1;
                    crate::wordpiece::accumulate(
                        vector,
                        &lookup::UNK,
                        lookup::UNK_MIN,
                        lookup::UNK_SCALE,
                    );
                }
                // Cannot include this char in the symbol list for merging; skip it.
                byte_pos += ch_len;
                continue;
            }
        }
        byte_pos += ch_len;
    }

    if symbols.is_empty() {
        return;
    }

    // Step 2: apply BPE merges.
    bpe_merge_all(&mut symbols);

    // Steps 3+4: reconstruct token strings from byte spans, look up slots, accumulate.
    // After merges, symbols.len is the sequence of final BPE tokens. Walk the chunk
    // from byte 0 forward, each symbol.len gives the byte span of that token in `chunk`.
    let mut cur_byte = 0usize;
    for sym in &symbols {
        let token_str = &chunk[cur_byte..cur_byte + sym.len];
        cur_byte += sym.len;

        if let Some(slot) = lookup::lookup_slot(token_str) {
            *token_count += 1;
            #[cfg(any(feature = "quant-8", feature = "quant-16"))]
            crate::wordpiece::accumulate(vector, &slot.weight, slot.row_min, slot.row_scale);
            #[cfg(not(any(feature = "quant-8", feature = "quant-16")))]
            crate::wordpiece::accumulate(vector, &slot.weight, 0.0, 1.0);
        } else {
            // Merged token not in vocab (should not happen for valid BPE model).
            if !lookup::SKIP_UNK {
                *token_count += 1;
                crate::wordpiece::accumulate(
                    vector,
                    &lookup::UNK,
                    lookup::UNK_MIN,
                    lookup::UNK_SCALE,
                );
            }
        }
    }
}

// ── encode_single (public entry point) ───────────────────────────────────────

/// Encode a single text using the ByteLevel-BPE path.
/// Called from lib.rs when IS_BPE=true (const dispatch; this function is dead code otherwise).
///
/// Pipeline:
///   normalize (none for ByteLevel BPE models — no normalizer in all-distilroberta-v1)
///   → ByteLevel pre-tokenize (regex split + byte-map)
///   → per-chunk BPE merge_all
///   → accumulate slot weight rows
///   → mean-pool
///   → optional L2-normalize
pub fn encode_single(text: &str) -> [f32; lookup::DIMENSIONS] {
    // Empty input: the HF ByteLevel BPE tokenizer returns [] for "", not ["Ġ"].
    // We match that by returning the zero vector immediately, before prefix-space is applied.
    if text.is_empty() {
        return [0.0f32; lookup::DIMENSIONS];
    }

    let mut vector = [0.0f32; lookup::DIMENSIONS];
    let mut token_count: usize = 0;

    for chunk in bytelevel_pretokenize(text) {
        encode_chunk(&chunk, &mut vector, &mut token_count);
    }

    if token_count == 0 {
        return vector; // zero vector for empty input
    }

    let scale = 1.0 / token_count as f32;
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
