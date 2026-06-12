use base64::Engine as _;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    env,
    fs::{self},
    path::{Path, PathBuf},
};

// ── Default model (built-in pinned constants) ────────────────────────────────
// Pinned to upstream revision f60985c706f192d45d218078e49e5a8b6f15283a (resolved 2026-06-12).
// To bump: update the revision in both URLs, re-download, recompute the SHA-256s.
static DEFAULT_MODEL_NAME: &str =
    "sentence-transformers/static-retrieval-mrl-en-v1@f60985c706f192d45d218078e49e5a8b6f15283a";
static DEFAULT_MODEL_URL: &str = "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/model.safetensors";
static DEFAULT_TOKENIZER_URL: &str = "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/f60985c706f192d45d218078e49e5a8b6f15283a/0_StaticEmbedding/tokenizer.json";
static DEFAULT_MODEL_SHA256: &str =
    "164fc63ee9f9267be7378fcbd7df99d09788a2f45244c92aa99ae5a574925716";
static DEFAULT_TOKENIZER_SHA256: &str =
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66";

// ── Model resolution ─────────────────────────────────────────────────────────

struct ResolvedModel {
    /// display name, e.g. "sentence-transformers/static-retrieval-mrl-en-v1@<rev>"
    name: String,
    /// cache subdirectory key (sanitized name: '/' and '@' -> '-')
    cache_key: String,
    /// where artifacts come from
    source: ModelSource,
}

enum ModelSource {
    Remote {
        model_url: String,
        tokenizer_url: String,
        model_sha256: Option<String>,
        tokenizer_sha256: Option<String>,
    },
    LocalDir(PathBuf),
}

fn sanitize_cache_key(s: &str) -> String {
    s.replace('/', "-").replace('@', "-")
}

fn resolve_model() -> ResolvedModel {
    // Emit rerun-if-env-changed for every var we might read
    println!("cargo:rerun-if-env-changed=ESE_MODEL_DIR");
    println!("cargo:rerun-if-env-changed=ESE_MODEL");
    println!("cargo:rerun-if-env-changed=ESE_MODEL_SHA256");
    println!("cargo:rerun-if-env-changed=ESE_TOKENIZER_SHA256");
    println!("cargo:rerun-if-env-changed=ESE_MODEL_FILE");
    println!("cargo:rerun-if-env-changed=ESE_TOKENIZER_FILE");
    println!("cargo:rerun-if-env-changed=ESE_NORMALIZE_OUTPUT");

    // 1. ESE_MODEL_DIR: local directory, no download, no checksum
    if let Ok(dir_str) = env::var("ESE_MODEL_DIR") {
        let dir = PathBuf::from(&dir_str);
        let model_file = dir.join("model.safetensors");
        let tokenizer_file = dir.join("tokenizer.json");
        if !model_file.exists() || !tokenizer_file.exists() {
            panic!(
                "ESE_MODEL_DIR={} must contain both model.safetensors and tokenizer.json (missing: {}{})",
                dir_str,
                if !model_file.exists() { "model.safetensors " } else { "" },
                if !tokenizer_file.exists() { "tokenizer.json" } else { "" },
            );
        }
        let name = format!("local:{}", dir_str);
        let cache_key = sanitize_cache_key(&name);
        return ResolvedModel {
            name,
            cache_key,
            source: ModelSource::LocalDir(dir),
        };
    }

    // 2. ESE_MODEL: <hf-repo>@<revision> (revision is mandatory)
    if let Ok(model_spec) = env::var("ESE_MODEL") {
        // Require @revision
        let at_pos = model_spec.find('@').unwrap_or_else(|| {
            panic!(
                "ESE_MODEL={:?} must include a pinned revision separated by '@', \
                 e.g. ESE_MODEL=owner/repo@<40-char-sha>. \
                 Bare repo IDs are rejected — reproducibility is non-negotiable \
                 (see plans/003-pin-and-checksum-model-downloads.md).",
                model_spec
            )
        });
        let repo = &model_spec[..at_pos];
        let rev = &model_spec[at_pos + 1..];
        if rev.is_empty() {
            panic!(
                "ESE_MODEL={:?} has '@' but no revision after it. \
                 Provide the full commit SHA, e.g. owner/repo@abc1234...",
                model_spec
            );
        }

        // Subpath overrides (for models with non-standard paths)
        let model_file = env::var("ESE_MODEL_FILE").unwrap_or_else(|_| "model.safetensors".into());
        let tokenizer_file =
            env::var("ESE_TOKENIZER_FILE").unwrap_or_else(|_| "tokenizer.json".into());

        let model_url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            repo, rev, model_file
        );
        let tokenizer_url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            repo, rev, tokenizer_file
        );

        let model_sha256 = env::var("ESE_MODEL_SHA256").ok();
        let tokenizer_sha256 = env::var("ESE_TOKENIZER_SHA256").ok();

        if model_sha256.is_none() {
            println!("cargo:warning=ESE_MODEL_SHA256 not set — integrity of {} is unverified. Set it to the SHA-256 of model.safetensors for reproducible builds.", repo);
        }
        if tokenizer_sha256.is_none() {
            println!("cargo:warning=ESE_TOKENIZER_SHA256 not set — integrity of {} is unverified. Set it to the SHA-256 of tokenizer.json for reproducible builds.", repo);
        }

        let name = model_spec.clone();
        let cache_key = sanitize_cache_key(&name);
        return ResolvedModel {
            name,
            cache_key,
            source: ModelSource::Remote {
                model_url,
                tokenizer_url,
                model_sha256,
                tokenizer_sha256,
            },
        };
    }

    // 3. Built-in default
    ResolvedModel {
        name: DEFAULT_MODEL_NAME.to_string(),
        cache_key: sanitize_cache_key(DEFAULT_MODEL_NAME),
        source: ModelSource::Remote {
            model_url: DEFAULT_MODEL_URL.to_string(),
            tokenizer_url: DEFAULT_TOKENIZER_URL.to_string(),
            model_sha256: Some(DEFAULT_MODEL_SHA256.to_string()),
            tokenizer_sha256: Some(DEFAULT_TOKENIZER_SHA256.to_string()),
        },
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let out = Path::new(&env::var("OUT_DIR").unwrap()).to_path_buf();
    println!("cargo:rerun-if-env-changed=ESE_CACHE_DIR");

    let dtype = dtype();

    let model = resolve_model();
    let model_dir = shared_model_dir(&out, &model.cache_key);

    let (model_path, tokenizer_path) = match &model.source {
        ModelSource::LocalDir(dir) => (
            dir.join("model.safetensors"),
            dir.join("tokenizer.json"),
        ),
        ModelSource::Remote {
            model_url,
            tokenizer_url,
            model_sha256,
            tokenizer_sha256,
        } => {
            let mp = model_dir.join("model.safetensors");
            let tp = model_dir.join("tokenizer.json");
            download_if_missing(&mp, model_url, model_sha256.as_deref());
            download_if_missing(&tp, tokenizer_url, tokenizer_sha256.as_deref());
            (mp, tp)
        }
    };

    let (weights, dims) = parse_safetensors(&model_path);

    println!(
        "cargo:rerun-if-changed={}",
        tokenizer_path.to_str().unwrap()
    );
    let tokenizer_data = fs::read_to_string(&tokenizer_path).expect("failed to read tokenizer file");
    let tk: serde_json::Value = serde_json::from_str(tokenizer_data.as_str()).unwrap();

    // ── Guard: tokenizer family and normalizer ────────────────────────────────
    guard_tokenizer_family(&tk, &model.name);

    // NORMALIZE_OUTPUT: env var ESE_NORMALIZE_OUTPUT=1
    let normalize_output = env::var("ESE_NORMALIZE_OUTPUT").map(|v| v == "1").unwrap_or(false);

    let model_type = tk["model"]["type"].as_str().unwrap_or("WordPiece");

    if model_type == "Unigram" {
        build_unigram(&out, &tk, &model.name, &weights, dims as usize, normalize_output);
        #[cfg(feature = "tests")]
        generate_testdata(
            &out,
            &weights,
            dims as usize,
            &tokenizer_path,
            0.0,
            1.0,
            false, // Unigram models never add CLS/SEP
            normalize_output,
        );
        return;
    }

    // ── WordPiece path ────────────────────────────────────────────────────────
    let vocab = tk["model"]["vocab"]
        .as_object()
        .expect("missing model.vocab");

    // ── Derive model facts ────────────────────────────────────────────────────

    // UNK token: from model.unk_token if present, else "[UNK]"
    let unk_token = tk["model"]["unk_token"]
        .as_str()
        .unwrap_or("[UNK]")
        .to_string();

    let unk_id = special_token_id(vocab, &unk_token)
        .unwrap_or_else(|| panic!(
            "tokenizer vocab for '{}' is missing '{}' (UNK token) — cannot build embedding table",
            model.name, unk_token
        ));

    // ADD_SPECIAL_TOKENS: derived from post_processor
    let add_special_tokens = derive_add_special_tokens(&tk, &model.name);

    let cls_id = if add_special_tokens {
        Some(special_token_id(vocab, "[CLS]").unwrap_or_else(|| panic!(
            "model '{}' post_processor references CLS but '[CLS]' is missing from vocab",
            model.name
        )))
    } else {
        None
    };
    let sep_id = if add_special_tokens {
        Some(special_token_id(vocab, "[SEP]").unwrap_or_else(|| panic!(
            "model '{}' post_processor references SEP but '[SEP]' is missing from vocab",
            model.name
        )))
    } else {
        None
    };

    // CONTINUING_PREFIX: from model.continuing_subword_prefix, default "##"
    let continuing_prefix = tk["model"]["continuing_subword_prefix"]
        .as_str()
        .unwrap_or("##")
        .to_string();

    // MAX_WORD_LEN: from model.max_input_chars_per_word, default 100
    let max_word_len = tk["model"]["max_input_chars_per_word"]
        .as_u64()
        .unwrap_or(100) as usize;

    let zero_row = vec![0.0f32; dims as usize];
    let mut unk: Vec<f32> = zero_row.clone();
    let mut cls: Vec<f32> = zero_row.clone();
    let mut sep: Vec<f32> = zero_row.clone();

    // Collect (token, weight_index) pairs
    let mut entries: Vec<(String, usize)> = Vec::new();
    for (token, id_val) in vocab {
        let i = id_val.as_u64().unwrap() as usize;
        if i >= weights.len() {
            continue;
        }
        entries.push((token.clone(), i));
        if i == unk_id {
            unk = weights[i].clone();
        } else if Some(i) == cls_id {
            cls = weights[i].clone();
        } else if Some(i) == sep_id {
            sep = weights[i].clone();
        }
    }

    // hash table
    let vocab_size = entries.len();
    let table_size = vocab_size.next_power_of_two();
    let table_mask = table_size - 1;
    let num_buckets = vocab_size;

    // bucket entries by primary hash
    let mut buckets: Vec<Vec<usize>> = vec![vec![]; num_buckets];
    for (i, (token, _)) in entries.iter().enumerate() {
        let b = (phf_hash(token.as_bytes(), 0) as usize) % num_buckets;
        buckets[b].push(i);
    }

    let mut bucket_order: Vec<usize> = (0..num_buckets).collect();
    bucket_order.sort_by(|a, b| buckets[*b].len().cmp(&buckets[*a].len()));

    let mut seeds: Vec<u32> = vec![0; num_buckets];
    let mut verify: Vec<u64> = vec![0; table_size];
    let mut slot_to_weight: Vec<usize> = vec![0; table_size];
    let mut occupied: Vec<bool> = vec![false; table_size];

    for &bi in &bucket_order {
        if buckets[bi].is_empty() {
            continue;
        }

        let mut attempts = 0;
        let mut i: u32 = 0;
        'search: loop {
            let seed = i | 1;
            i += 1;

            attempts += 1;
            assert!(
                attempts < 1_000_000,
                "PHF construction stuck on bucket {bi} ({} entries)",
                buckets[bi].len()
            );

            let mut trial_slots: Vec<usize> = Vec::with_capacity(buckets[bi].len());
            let mut seen = HashSet::with_capacity(buckets[bi].len());

            for &entry_idx in &buckets[bi] {
                let slot =
                    (phf_hash(entries[entry_idx].0.as_bytes(), seed as u64) as usize) & table_mask;
                if occupied[slot] || !seen.insert(slot) {
                    continue 'search;
                }
                trial_slots.push(slot);
            }

            // all placed
            seeds[bi] = seed;
            for (j, &entry_idx) in buckets[bi].iter().enumerate() {
                let slot = trial_slots[j];
                occupied[slot] = true;
                verify[slot] = phf_hash(entries[entry_idx].0.as_bytes(), u64::MAX);
                slot_to_weight[slot] = entries[entry_idx].1;
            }
            break;
        }
    }

    // compute global min/max for quantization
    let (global_min, global_max) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for row in &weights {
            for &v in row {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        (mn, mx)
    } else {
        (0.0, 1.0) // unused
    };

    let quant_max = if cfg!(feature = "quant-8") {
        255.0
    } else if cfg!(feature = "quant-16") {
        65535.0
    } else {
        1.0
    };

    let quant_scale = if (global_max - global_min).abs() < f32::EPSILON {
        1.0
    } else {
        (global_max - global_min) / quant_max
    };

    // reorder weights into PHF slot with verify hash prepended
    // && zero-fill empty slots
    let slot_size = 8 + dims as usize * dtype_sz();
    let mut phf_slots: Vec<u8> = Vec::with_capacity(table_size * slot_size);
    for slot in 0..table_size {
        phf_slots.extend_from_slice(&verify[slot].to_le_bytes());
        let row = if occupied[slot] {
            &weights[slot_to_weight[slot]]
        } else {
            &zero_row
        };
        phf_slots.extend_from_slice(&quantize_f32_to_bytes(row, global_min, quant_scale));
    }
    fs::write(out.join("weights.bin"), &phf_slots).unwrap();

    // emit seeds as [u64; NUM_BUCKETS]
    let mut seeds_bin: Vec<u8> = Vec::with_capacity(num_buckets * 8);
    for &s in &seeds {
        seeds_bin.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(out.join("seeds.bin"), &seeds_bin).unwrap();

    // emit verify hashes as [u64; TABLE_SIZE]
    let mut verify_bin: Vec<u8> = Vec::with_capacity(table_size * 8);
    for &v in &verify {
        verify_bin.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(out.join("verify.bin"), &verify_bin).unwrap();

    let unk_str = format_quantized_param(&unk, global_min, quant_scale);
    let cls_str = format_quantized_param(&cls, global_min, quant_scale);
    let sep_str = format_quantized_param(&sep, global_min, quant_scale);

    let dtype_sz = dtype_sz();

    // emit constants
    fs::write(
        out.join("model_constants.rs"),
        format!(
            "\
pub const DIMENSIONS: usize = {dims};\n\
pub type Param = [{dtype}; DIMENSIONS];\n\
pub const TABLE_SIZE: usize = {table_size};\n\
pub const TABLE_MASK: usize = {table_mask};\n\
pub const NUM_BUCKETS: usize = {num_buckets};\n\
pub const SLOT_SIZE: usize = 8 + DIMENSIONS * DTYPE_SIZE;\n\
pub const UNK: Param = {unk_str};\n\
pub const CLS: Param = {cls_str};\n\
pub const SEP: Param = {sep_str};\n\
pub const MAX_WORD_LEN: usize = {max_word_len};\n\
pub const CONTINUING_PREFIX: &str = {continuing_prefix:?};\n\
pub const ADD_SPECIAL_TOKENS: bool = {add_special_tokens};\n\
pub const NORMALIZE_OUTPUT: bool = {normalize_output};\n\
pub const DTYPE_SIZE: usize = {dtype_sz};\n\
pub const QUANT_MIN: f32 = {global_min:?};\n\
pub const QUANT_SCALE: f32 = {quant_scale:?};\n\
pub const MODEL_NAME: &str = {model_name:?};\n\
// Unigram-specific constants. Dead in WordPiece builds; the compiler elides the
// Unigram code path via the IS_UNIGRAM const branch in encode_single.
#[allow(dead_code)] pub const IS_UNIGRAM: bool = false;\n\
#[allow(dead_code)] pub const MAX_TOKEN_CHARS: usize = 0;\n\
#[allow(dead_code)] pub const UNK_ID: usize = {unk_id};\n\
#[allow(dead_code)] pub const PAD_CHARS: &[char] = &[];\n\
#[allow(dead_code)] pub const COLLAPSE_SPACES: bool = false;\n\
#[allow(dead_code)] pub const COLLAPSE_WHITESPACE: bool = false;\n\
#[allow(dead_code)] pub const STRIP_LEFT: bool = false;\n\
#[allow(dead_code)] pub const STRIP_RIGHT: bool = false;\n\
#[allow(dead_code)] pub const METASPACE_CHAR: char = '\\u{{2581}}';\n\
#[allow(dead_code)] pub const METASPACE_PREPEND: bool = false;\n\
",
            model_name = model.name,
        ),
    )
    .unwrap();

    // Emit dummy scores.bin and charsmap.bin for WordPiece builds (IS_UNIGRAM=false,
    // so these are never referenced at runtime; they exist so include_bytes! compiles).
    fs::write(out.join("scores.bin"), &[0u8; 4]).unwrap();
    fs::write(out.join("charsmap.bin"), &[0u8; 0]).unwrap();

    #[cfg(feature = "tests")]
    generate_testdata(
        &out,
        &weights,
        dims as usize,
        &tokenizer_path,
        global_min,
        quant_scale,
        add_special_tokens,
        normalize_output,
    );
}

// ── Unigram build path ────────────────────────────────────────────────────────

/// Build all artifacts for an Unigram model: PHF table (same layout as WordPiece),
/// scores.bin (parallel f32 array indexed by PHF slot), charsmap.bin (decoded
/// Precompiled normalizer bytes), and model_constants.rs with Unigram-specific consts.
fn build_unigram(
    out: &Path,
    tk: &serde_json::Value,
    model_name: &str,
    weights: &[Vec<f32>],
    dims: usize,
    normalize_output: bool,
) {
    // ── Parse vocab: array of [token, score] pairs ────────────────────────────
    let vocab_arr = tk["model"]["vocab"]
        .as_array()
        .expect("Unigram model.vocab must be an array of [token, score] pairs");

    // id = index in array
    let unk_id = tk["model"]["unk_id"].as_u64().unwrap_or(0) as usize;

    // Collect (token, id=index) pairs — mirrors WordPiece entries
    let mut entries: Vec<(String, usize)> = Vec::with_capacity(vocab_arr.len());
    // Scores parallel to vocab index (id → score as f32)
    let mut vocab_scores: Vec<f32> = Vec::with_capacity(vocab_arr.len());

    // Find max token char length for MAX_TOKEN_CHARS
    let mut max_token_chars: usize = 0;
    // Collect pad chars for PAD_CHARS
    let mut pad_chars: Vec<char> = Vec::new();
    // Normalizer op flags
    let mut collapse_spaces = false;
    let mut collapse_whitespace = false;
    let mut strip_left = false;
    let mut strip_right = false;
    // Charsmap base64
    let mut charsmap_b64: Option<String> = None;

    for (id, entry) in vocab_arr.iter().enumerate() {
        let token = entry[0].as_str().unwrap_or("").to_string();
        let score = entry[1].as_f64().unwrap_or(0.0) as f32;
        let char_len = token.chars().count();
        if char_len > max_token_chars {
            max_token_chars = char_len;
        }
        entries.push((token, id));
        vocab_scores.push(score);
    }

    // ── Collect normalizer-op constants by walking the chain ──────────────────
    collect_unigram_norm_ops(
        &tk["normalizer"],
        &mut charsmap_b64,
        &mut pad_chars,
        &mut collapse_spaces,
        &mut collapse_whitespace,
        &mut strip_left,
        &mut strip_right,
    );

    // ── Decode and write charsmap.bin ─────────────────────────────────────────
    let charsmap_bytes = if let Some(b64) = &charsmap_b64 {
        base64::engine::general_purpose::STANDARD.decode(b64)
            .unwrap_or_else(|e| panic!(
                "model '{}' Precompiled charsmap base64 decode failed: {}",
                model_name, e
            ))
    } else {
        vec![]
    };
    fs::write(out.join("charsmap.bin"), &charsmap_bytes).unwrap();

    // ── PHF construction (same algorithm as WordPiece) ────────────────────────
    let vocab_size = entries.len();
    let table_size = vocab_size.next_power_of_two();
    let table_mask = table_size - 1;
    let num_buckets = vocab_size;

    let mut buckets: Vec<Vec<usize>> = vec![vec![]; num_buckets];
    for (i, (token, _)) in entries.iter().enumerate() {
        let b = (phf_hash(token.as_bytes(), 0) as usize) % num_buckets;
        buckets[b].push(i);
    }

    let mut bucket_order: Vec<usize> = (0..num_buckets).collect();
    bucket_order.sort_by(|a, b| buckets[*b].len().cmp(&buckets[*a].len()));

    let mut seeds: Vec<u32> = vec![0; num_buckets];
    let mut verify: Vec<u64> = vec![0; table_size];
    let mut slot_to_weight: Vec<usize> = vec![0; table_size];
    let mut occupied: Vec<bool> = vec![false; table_size];

    for &bi in &bucket_order {
        if buckets[bi].is_empty() {
            continue;
        }
        let mut i: u32 = 0;
        'search: loop {
            let seed = i | 1;
            i += 1;
            assert!(i < 1_000_000, "PHF construction stuck on bucket {bi}");

            let mut trial_slots: Vec<usize> = Vec::with_capacity(buckets[bi].len());
            let mut seen = HashSet::with_capacity(buckets[bi].len());

            for &entry_idx in &buckets[bi] {
                let slot = (phf_hash(entries[entry_idx].0.as_bytes(), seed as u64) as usize) & table_mask;
                if occupied[slot] || !seen.insert(slot) {
                    continue 'search;
                }
                trial_slots.push(slot);
            }

            seeds[bi] = seed;
            for (j, &entry_idx) in buckets[bi].iter().enumerate() {
                let slot = trial_slots[j];
                occupied[slot] = true;
                verify[slot] = phf_hash(entries[entry_idx].0.as_bytes(), u64::MAX);
                slot_to_weight[slot] = entries[entry_idx].1;
            }
            break;
        }
    }

    // ── Emit scores.bin: f32 array parallel to PHF slots ─────────────────────
    // slot → vocab-id → score. Empty slots get 0.0 (never read at runtime).
    let mut scores_bin: Vec<u8> = Vec::with_capacity(table_size * 4);
    for slot in 0..table_size {
        let score = if occupied[slot] {
            vocab_scores[slot_to_weight[slot]]
        } else {
            0.0f32
        };
        scores_bin.extend_from_slice(&score.to_le_bytes());
    }
    fs::write(out.join("scores.bin"), &scores_bin).unwrap();

    // ── Emit weights.bin (same layout as WordPiece — no quantization for Unigram) ─
    // Unigram models from model2vec are always F32; quantization is orthogonal.
    // For now emit raw f32 (quant-8/16 would apply identically; the scoring
    // array uses raw f32 scores regardless).
    let slot_size = 8 + dims * 4; // verify:u64 + row:f32*dims
    let mut phf_slots: Vec<u8> = Vec::with_capacity(table_size * slot_size);
    let zero_row = vec![0.0f32; dims];

    // compute global min/max for quantization
    let (global_min, global_max) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for row in weights {
            for &v in row {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        (mn, mx)
    } else {
        (0.0f32, 1.0f32)
    };
    let quant_max = if cfg!(feature = "quant-8") { 255.0 } else if cfg!(feature = "quant-16") { 65535.0 } else { 1.0 };
    let quant_scale = if (global_max - global_min).abs() < f32::EPSILON {
        1.0f32
    } else {
        (global_max - global_min) / quant_max
    };

    for slot in 0..table_size {
        phf_slots.extend_from_slice(&verify[slot].to_le_bytes());
        let row = if occupied[slot] {
            &weights[slot_to_weight[slot]]
        } else {
            &zero_row
        };
        phf_slots.extend_from_slice(&quantize_f32_to_bytes(row, global_min, quant_scale));
    }
    fs::write(out.join("weights.bin"), &phf_slots).unwrap();

    // seeds.bin and verify.bin
    let mut seeds_bin: Vec<u8> = Vec::with_capacity(num_buckets * 4);
    for &s in &seeds {
        seeds_bin.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(out.join("seeds.bin"), &seeds_bin).unwrap();

    let mut verify_bin: Vec<u8> = Vec::with_capacity(table_size * 8);
    for &v in &verify {
        verify_bin.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(out.join("verify.bin"), &verify_bin).unwrap();

    // ── Derive ADD_SPECIAL_TOKENS (always false for Unigram models here) ──────
    // The post_processor is TemplateProcessing with $A only — derive_add_special_tokens
    // will return false because no SpecialToken{id:[CLS]/[SEP]} appears in the template.
    // Re-use existing function for correctness.
    // Note: derive_add_special_tokens reads tk["model"]["vocab"] as object — for Unigram
    // we skip that cross-check path by providing an empty template (the function will
    // return false because there are no [CLS]/[SEP] SpecialTokens in $A-only template).
    let add_special_tokens = derive_add_special_tokens(tk, model_name);
    // Guard: Unigram models must not add special tokens (the unk row is the only special)
    assert!(!add_special_tokens, "Unigram model '{}' unexpectedly derived add_special_tokens=true; this should not happen for $A-only TemplateProcessing", model_name);

    // Emit the UNK embedding as the constant (used for UNK pooling)
    let unk_row = if unk_id < weights.len() { &weights[unk_id] } else { &zero_row };
    let dtype = dtype();
    let dtype_sz = dtype_sz();
    let unk_str = format_quantized_param(unk_row, global_min, quant_scale);
    let zero_str = format_quantized_param(&zero_row, global_min, quant_scale);

    // Format pad_chars as a Rust char literal array
    let pad_chars_str = pad_chars.iter()
        .map(|c| format!("'\\u{{{:04X}}}'", *c as u32))
        .collect::<Vec<_>>()
        .join(", ");

    let table_mask_val = table_mask;

    fs::write(
        out.join("model_constants.rs"),
        format!(
            "\
pub const DIMENSIONS: usize = {dims};\n\
pub type Param = [{dtype}; DIMENSIONS];\n\
pub const TABLE_SIZE: usize = {table_size};\n\
pub const TABLE_MASK: usize = {table_mask_val};\n\
pub const NUM_BUCKETS: usize = {num_buckets};\n\
pub const SLOT_SIZE: usize = 8 + DIMENSIONS * DTYPE_SIZE;\n\
pub const UNK: Param = {unk_str};\n\
pub const CLS: Param = {zero_str};\n\
pub const SEP: Param = {zero_str};\n\
pub const MAX_WORD_LEN: usize = 0;\n\
pub const CONTINUING_PREFIX: &str = \"##\";\n\
pub const ADD_SPECIAL_TOKENS: bool = false;\n\
pub const NORMALIZE_OUTPUT: bool = {normalize_output};\n\
pub const DTYPE_SIZE: usize = {dtype_sz};\n\
pub const QUANT_MIN: f32 = {global_min:?};\n\
pub const QUANT_SCALE: f32 = {quant_scale:?};\n\
pub const MODEL_NAME: &str = {model_name:?};\n\
// Unigram-specific constants. Some may be dead in certain feature combinations.\n\
#[allow(dead_code)] pub const IS_UNIGRAM: bool = true;\n\
#[allow(dead_code)] pub const MAX_TOKEN_CHARS: usize = {max_token_chars};\n\
#[allow(dead_code)] pub const UNK_ID: usize = {unk_id};\n\
#[allow(dead_code)] pub const PAD_CHARS: &[char] = &[{pad_chars_str}];\n\
#[allow(dead_code)] pub const COLLAPSE_SPACES: bool = {collapse_spaces};\n\
#[allow(dead_code)] pub const COLLAPSE_WHITESPACE: bool = {collapse_whitespace};\n\
#[allow(dead_code)] pub const STRIP_LEFT: bool = {strip_left};\n\
#[allow(dead_code)] pub const STRIP_RIGHT: bool = {strip_right};\n\
#[allow(dead_code)] pub const METASPACE_CHAR: char = '\\u{{2581}}';\n\
#[allow(dead_code)] pub const METASPACE_PREPEND: bool = true;\n\
",
        ),
    )
    .unwrap();
}

/// Walk the normalizer chain recursively to extract Unigram-specific op constants.
fn collect_unigram_norm_ops(
    norm: &serde_json::Value,
    charsmap_b64: &mut Option<String>,
    pad_chars: &mut Vec<char>,
    collapse_spaces: &mut bool,
    collapse_whitespace: &mut bool,
    strip_left: &mut bool,
    strip_right: &mut bool,
) {
    match norm["type"].as_str() {
        Some("Sequence") => {
            if let Some(items) = norm["normalizers"].as_array() {
                for item in items {
                    collect_unigram_norm_ops(
                        item, charsmap_b64, pad_chars,
                        collapse_spaces, collapse_whitespace,
                        strip_left, strip_right,
                    );
                }
            }
        }
        Some("Precompiled") => {
            if let Some(b64) = norm["precompiled_charsmap"].as_str() {
                *charsmap_b64 = Some(b64.to_string());
            }
        }
        Some("Replace") => {
            if let Some(pattern) = norm["pattern"].get("Regex").and_then(|p| p.as_str()) {
                if pattern == " {2,}" {
                    *collapse_spaces = true;
                } else if pattern == "\\s+" {
                    *collapse_whitespace = true;
                }
            } else if let Some(pattern) = norm["pattern"].get("String").and_then(|p| p.as_str()) {
                // Single-char pad replacement
                if let Some(c) = pattern.chars().next() {
                    pad_chars.push(c);
                }
            }
        }
        Some("Strip") => {
            *strip_left = norm["strip_left"].as_bool().unwrap_or(false);
            *strip_right = norm["strip_right"].as_bool().unwrap_or(false);
        }
        _ => {}
    }
}

// ── Tokenizer family guard ────────────────────────────────────────────────────

/// Accept tokenizer configs whose normalization and vocab type the compiled
/// runtime can handle faithfully. Anything else panics at build time with a
/// clear message.
///
/// Accepted model types:
///   - WordPiece: normalizer must match src/pretokenizer.rs shapes.
///   - Unigram: normalizer must be a Sequence of exactly the compiled ops
///     (Precompiled, space-collapse Regex Replace, single-char pad String
///     Replaces, whitespace-collapse Regex Replace, Strip). Pre-tokenizer
///     must be Metaspace{replacement: ▁, prepend_scheme: always, split: false}.
///     byte_fallback must be false/absent.
///
/// Everything else is rejected with a message naming the unsupported type and
/// pointing at docs/rfc-generalization.md.
fn guard_tokenizer_family(tk: &serde_json::Value, model_name: &str) {
    let model_type = tk["model"]["type"].as_str().unwrap_or("unknown");
    match model_type {
        "WordPiece" => {
            // Existing WordPiece path: check normalizer matches pretokenizer.rs
            check_normalizer(&tk["normalizer"], model_name);
        }
        "Unigram" => {
            // New Unigram path: validate the shapes this runtime implements
            guard_unigram_shapes(tk, model_name);
        }
        other => {
            panic!(
                "model '{}' uses a '{}' tokenizer; ese compiles WordPiece and Unigram \
                 (SentencePiece) models only — BPE, WordLevel, and other types are not \
                 supported (see docs/rfc-generalization.md).",
                model_name, other
            );
        }
    }
}

/// Validate the Unigram-specific shapes this runtime can faithfully compile.
/// Panics naming any component that does not match the accepted shape.
fn guard_unigram_shapes(tk: &serde_json::Value, model_name: &str) {
    // byte_fallback must be absent or false
    let byte_fallback = tk["model"]["byte_fallback"].as_bool().unwrap_or(false);
    if byte_fallback {
        panic!(
            "model '{}' has byte_fallback:true; ese's Unigram runtime does not implement \
             byte-fallback vocabs (see docs/rfc-generalization.md, layer A part 2).",
            model_name
        );
    }

    // Pre-tokenizer must be Metaspace{replacement: ▁, prepend_scheme: always, split: false}
    let pre = &tk["pre_tokenizer"];
    if pre["type"].as_str() != Some("Metaspace") {
        panic!(
            "model '{}' has pre_tokenizer type '{}'; ese's Unigram runtime only supports \
             Metaspace pre-tokenizers (see docs/rfc-generalization.md).",
            model_name,
            pre["type"].as_str().unwrap_or("unknown")
        );
    }
    let replacement = pre["replacement"].as_str().unwrap_or("");
    if replacement != "▁" {
        panic!(
            "model '{}' Metaspace has replacement {:?}; ese only supports ▁ (U+2581).",
            model_name, replacement
        );
    }
    let prepend_scheme = pre["prepend_scheme"].as_str().unwrap_or("always");
    if prepend_scheme != "always" {
        panic!(
            "model '{}' Metaspace has prepend_scheme={:?}; ese's Unigram runtime only \
             supports prepend_scheme=always (see docs/rfc-generalization.md, layer A part 2).",
            model_name, prepend_scheme
        );
    }
    let split = pre["split"].as_bool().unwrap_or(true);
    if split {
        panic!(
            "model '{}' Metaspace has split:true; ese's Unigram runtime only supports \
             split:false (the entire normalized string is passed as one chunk to Viterbi; \
             see docs/rfc-generalization.md, layer A part 2).",
            model_name
        );
    }

    // Normalizer chain must be a Sequence of exactly the supported ops
    check_unigram_normalizer(&tk["normalizer"], model_name);
}

/// Check that the Unigram normalizer chain consists only of the ops this
/// runtime compiles: Precompiled, Regex Replace with " {2,}", single-char
/// String Replaces (pad), Regex Replace with "\\s+", and Strip.
fn check_unigram_normalizer(norm: &serde_json::Value, model_name: &str) {
    match norm["type"].as_str() {
        Some("Sequence") => {
            if let Some(items) = norm["normalizers"].as_array() {
                for item in items {
                    check_unigram_normalizer(item, model_name);
                }
            }
        }
        Some("Precompiled") => {
            // The charsmap is decoded later; just confirm the field is present
            if norm["precompiled_charsmap"].as_str().is_none() {
                panic!(
                    "model '{}' Precompiled normalizer is missing precompiled_charsmap field.",
                    model_name
                );
            }
        }
        Some("Replace") => {
            let content = norm["content"].as_str().unwrap_or("");
            if let Some(pattern) = norm["pattern"].get("Regex").and_then(|p| p.as_str()) {
                // Only " {2,}" and "\\s+" are accepted
                if pattern != " {2,}" && pattern != "\\s+" {
                    panic!(
                        "model '{}' has a Regex Replace with pattern {:?}; ese's Unigram \
                         runtime only accepts ' {{2,}}' (collapse spaces) and '\\s+' \
                         (collapse whitespace) — any other regex is not compiled \
                         (see docs/rfc-generalization.md).",
                        model_name, pattern
                    );
                }
                // Both accepted regex patterns must replace with a single space
                if content != " " {
                    panic!(
                        "model '{}' Regex Replace {:?} has content {:?}; expected a single space.",
                        model_name, pattern, content
                    );
                }
            } else if let Some(pattern) = norm["pattern"].get("String").and_then(|p| p.as_str()) {
                // Pattern must be exactly one char; content must be " <char> "
                let chars: Vec<char> = pattern.chars().collect();
                if chars.len() != 1 {
                    panic!(
                        "model '{}' String Replace pattern {:?} is not exactly one character; \
                         ese's Unigram runtime only compiles single-char pad replacements \
                         (see docs/rfc-generalization.md).",
                        model_name, pattern
                    );
                }
                let expected_content = format!(" {} ", chars[0]);
                if content != expected_content {
                    panic!(
                        "model '{}' String Replace pattern {:?} has content {:?}; expected {:?} \
                         (space-char-space); ese only compiles that exact pad shape.",
                        model_name, pattern, content, expected_content
                    );
                }
            } else {
                panic!(
                    "model '{}' has a Replace normalizer with an unrecognized pattern shape; \
                     ese accepts only Regex or String patterns \
                     (see docs/rfc-generalization.md).",
                    model_name
                );
            }
        }
        Some("Strip") => {
            // Strip is accepted; strip_left/strip_right are read later
        }
        Some(other) => {
            panic!(
                "model '{}' has a '{}' normalizer op in the Unigram chain; ese's Unigram \
                 runtime only compiles: Sequence, Precompiled, Replace (single-char pad or \
                 the two exact whitespace regexes), and Strip \
                 (see docs/rfc-generalization.md).",
                model_name, other
            );
        }
        None => {
            panic!(
                "model '{}' has a normalizer object with no 'type' field in the Unigram chain.",
                model_name
            );
        }
    }
}

fn check_normalizer(norm: &serde_json::Value, model_name: &str) {
    if norm.is_null() || *norm == serde_json::Value::Null {
        // No normalizer declared. The runtime always applies BERT-uncased normalization
        // (lowercase + NFD + strip combining marks) regardless — not identity.
        println!(
            "cargo:warning=model '{}' has no normalizer field; the ese runtime \
             (src/pretokenizer.rs) applies BERT-uncased normalization (lowercase + NFD + \
             strip combining marks) unconditionally. This is only correct for \
             already-uncased inputs.",
            model_name
        );
        return;
    }
    match norm["type"].as_str() {
        Some("BertNormalizer") => {
            // Must have lowercase: true (or absent, defaulting to true for BERT)
            let lowercase = norm["lowercase"].as_bool().unwrap_or(true);
            if !lowercase {
                panic!(
                    "model '{}' uses a BertNormalizer with lowercase:false (a cased model); \
                     ese's runtime (src/pretokenizer.rs) is BERT-uncased only. \
                     Cased model support is planned — see docs/rfc-generalization.md",
                    model_name
                );
            }
            // strip_accents: null or true are both fine (BERT treats null as
            // "follow lowercase", which means strip for uncased)
        }
        Some("Sequence") => {
            // Accept if every element is acceptable
            if let Some(normalizers) = norm["normalizers"].as_array() {
                for n in normalizers {
                    check_normalizer(n, model_name);
                }
            }
        }
        Some("Lowercase") => {
            // Plain lowercase matches pretokenizer.rs (applied unconditionally)
        }
        Some("NFD") => {
            // NFD decomposition matches pretokenizer.rs (applied unconditionally)
        }
        Some("StripAccents") => {
            // Accent stripping matches BERT-uncased behaviour in pretokenizer.rs
        }
        Some(other @ "NFC") | Some(other @ "NFKC") | Some(other @ "NFKD")
        | Some(other @ "Replace") | Some(other) => {
            panic!(
                "model '{}' uses a '{}' normalizer that src/pretokenizer.rs does not implement \
                 (implemented: BertNormalizer{{lowercase:true}}, Sequence, Lowercase, NFD, \
                 StripAccents). Refusing to compile — the runtime would silently diverge from \
                 the reference tokenizer. See docs/rfc-generalization.md for the roadmap.",
                model_name, other
            );
        }
        None => {
            // No type field on a non-null normalizer object — treat as unrecognized
            panic!(
                "model '{}' has a normalizer object with no 'type' field; ese cannot determine \
                 whether the runtime matches. Refusing to compile.",
                model_name
            );
        }
    }
}

/// Derive ADD_SPECIAL_TOKENS from the tokenizer's post_processor section.
///
/// BertProcessing → true (always inserts CLS/SEP).
/// TemplateProcessing → true only when the single or pair *template* references
///   a SpecialToken with id "[CLS]" or "[SEP]". A special_tokens map entry that
///   no template references does not get inserted at encode time, so the
///   specials map alone is not a reliable signal.
/// Absent post_processor or unrecognized type → false.
///
/// When the template references [CLS]/[SEP], also cross-checks that
/// post_processor.special_tokens[*].ids[0] matches the vocabulary id for the
/// same token. A mismatch (known for model2vec-compacted vocabs) means the
/// reference tokenizer and ese would pool different rows — refused at compile
/// time with a clear error.
fn derive_add_special_tokens(tk: &serde_json::Value, model_name: &str) -> bool {
    let pp = &tk["post_processor"];
    if pp.is_null() || *pp == serde_json::Value::Null {
        return false;
    }
    let vocab = tk["model"]["vocab"].as_object();
    let pp_type = pp["type"].as_str().unwrap_or("");
    match pp_type {
        "BertProcessing" => {
            // BertProcessing always adds CLS and SEP; cross-check ids if available
            cross_check_special_token_ids(pp, vocab, model_name);
            true
        }
        "TemplateProcessing" => {
            // Determine CLS/SEP use from the template arrays only, not the specials map
            let single = pp["single"].as_array();
            let pair = pp["pair"].as_array();

            let template_has_cls_sep = |arr: Option<&Vec<serde_json::Value>>| -> bool {
                arr.map(|items| {
                    items.iter().any(|item| {
                        item.get("SpecialToken")
                            .and_then(|st| st.get("id"))
                            .and_then(|id| id.as_str())
                            .map(|id| id == "[CLS]" || id == "[SEP]")
                            .unwrap_or(false)
                    })
                }).unwrap_or(false)
            };

            let uses_cls_sep = template_has_cls_sep(single) || template_has_cls_sep(pair);
            if uses_cls_sep {
                cross_check_special_token_ids(pp, vocab, model_name);
            }
            uses_cls_sep
        }
        "RobertaProcessing" => {
            // RobertaProcessing uses BOS/EOS, not CLS/SEP — treat as false
            // (the guard step will catch mismatches at the normalizer level)
            println!(
                "cargo:warning=model '{}' has a RobertaProcessing post_processor; \
                 ADD_SPECIAL_TOKENS will be false (CLS/SEP rows will be zero). \
                 This may produce incorrect embeddings if the model expects BOS/EOS pooling.",
                model_name
            );
            false
        }
        "" => false,
        _ => {
            println!(
                "cargo:warning=model '{}' has an unrecognized post_processor type '{}'; \
                 ADD_SPECIAL_TOKENS will be false.",
                model_name, pp_type
            );
            false
        }
    }
}

/// Cross-check that post_processor.special_tokens[token].ids[0] matches
/// vocab[token] for [CLS] and [SEP]. Panics on mismatch; skips when ids
/// field is absent (template-by-name path is fine).
fn cross_check_special_token_ids(
    pp: &serde_json::Value,
    vocab: Option<&serde_json::Map<String, serde_json::Value>>,
    model_name: &str,
) {
    let vocab = match vocab {
        Some(v) => v,
        None => return, // no vocab to cross-check against
    };
    let specials_map = match pp["special_tokens"].as_object() {
        Some(m) => m,
        None => return, // BertProcessing may not have a separate specials map
    };
    for token in &["[CLS]", "[SEP]"] {
        let entry = match specials_map.get(*token) {
            Some(e) => e,
            None => continue,
        };
        // ids field absent → template-by-name path, no cross-check needed
        let ids = match entry["ids"].as_array() {
            Some(arr) => arr,
            None => continue,
        };
        let pp_id = match ids.first().and_then(|v| v.as_u64()) {
            Some(id) => id,
            None => continue,
        };
        let vocab_id = match vocab.get(*token).and_then(|v| v.as_u64()) {
            Some(id) => id,
            None => continue, // token absent from vocab — caught elsewhere
        };
        if pp_id != vocab_id {
            panic!(
                "model '{}' post_processor stores id {} for {} but the vocab maps {} to {} — \
                 the tokenizer.json is internally inconsistent (known for model2vec-compacted \
                 vocabs); refusing to compile because the reference tokenizer and ese would \
                 pool different rows.",
                model_name, pp_id, token, token, vocab_id
            );
        }
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

fn special_token_id(
    vocab: &serde_json::Map<String, serde_json::Value>,
    token: &str,
) -> Option<usize> {
    vocab.get(token).and_then(|id| id.as_u64()).map(|id| id as usize)
}

fn phf_hash(key: &[u8], seed: u64) -> u64 {
    let mut h = seed ^ 0x517cc1b727220a95;
    for &b in key {
        h = (h ^ b as u64).wrapping_mul(0x2127599bf4325c37);
    }
    h ^ (h >> 32)
}

fn parse_safetensors(model_path: &Path) -> (Vec<Vec<f32>>, u64) {
    println!("cargo:rerun-if-changed={}", model_path.to_str().unwrap());

    let data = fs::read(model_path).expect("failed to read safetensors file");
    let meta_len = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let meta: serde_json::Value = serde_json::from_slice(&data[8..8 + meta_len]).unwrap();

    // Reject ambiguous files with more than one non-metadata tensor
    let non_meta_keys: Vec<String> = meta
        .as_object()
        .map(|m| m.keys().filter(|k| !k.starts_with("__")).cloned().collect())
        .unwrap_or_default();
    if non_meta_keys.len() > 1 {
        panic!(
            "safetensors file {} contains {} tensors ({}); ese expects exactly one embedding \
             weight tensor per file. Provide a single-tensor safetensors file.",
            model_path.display(),
            non_meta_keys.len(),
            non_meta_keys.join(", ")
        );
    }

    let tensor_key = non_meta_keys
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!(
            "safetensors file {} contains no non-metadata tensors; ese expects exactly one \
             embedding weight tensor per file.",
            model_path.display()
        ));

    let tensor = &meta[&tensor_key];
    let dtype = tensor["dtype"].as_str().unwrap_or("F32");
    let shape = tensor["shape"].as_array().unwrap();
    let params = shape[0].as_u64().unwrap();
    let full_dims = shape[1].as_u64().unwrap();

    let dims = full_dims.min(trunc_dims());

    let offsets = tensor["data_offsets"].as_array().unwrap();
    let start = 8 + meta_len + offsets[0].as_u64().unwrap() as usize;

    let src_elem_size: usize = match dtype {
        "F16" | "BF16" => 2,
        "F32" => 4,
        "F64" => 8,
        other => panic!("unsupported dtype: {other}"),
    };

    let row_src_bytes = full_dims as usize * src_elem_size;
    let mut tensors: Vec<Vec<f32>> = Vec::with_capacity(params as usize);
    let mut embedding: Vec<f32> = Vec::with_capacity(dims as usize);

    for row in 0..params as usize {
        let row_start = start + row * row_src_bytes;
        for col in 0..dims as usize {
            let elem_start = row_start + col * src_elem_size;
            let f32_val = match dtype {
                "F16" => {
                    let bits =
                        u16::from_le_bytes(data[elem_start..elem_start + 2].try_into().unwrap());
                    f16_to_f32(bits)
                }
                "BF16" => {
                    let bits =
                        u16::from_le_bytes(data[elem_start..elem_start + 2].try_into().unwrap());
                    bf16_to_f32(bits)
                }
                "F32" => f32::from_le_bytes(data[elem_start..elem_start + 4].try_into().unwrap()),
                "F64" => {
                    f64::from_le_bytes(data[elem_start..elem_start + 8].try_into().unwrap()) as f32
                }
                _ => unreachable!(),
            };
            embedding.push(f32_val);
        }
        tensors.push(embedding.clone());
        embedding.clear();
    }

    (tensors, dims)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    let (f_exp, f_mant) = match exp {
        0 => {
            if mant == 0 {
                (0u32, 0u32)
            } else {
                let mut m = mant;
                let mut e = 1u32;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e += 1;
                }
                ((127 - 15 + 1 - e), (m & 0x3FF) << 13)
            }
        }
        31 => (255u32, mant << 13),
        _ => ((exp + 127 - 15), mant << 13),
    };

    f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Quantize one f32 value to u8. `as u8` saturates (Rust 1.45+), so no clamp needed.
fn quantize_u8(v: f32, min: f32, scale: f32) -> u8 {
    ((v - min) / scale).round() as u8
}

/// Quantize one f32 value to u16. `as u16` saturates (Rust 1.45+), so no clamp needed.
fn quantize_u16(v: f32, min: f32, scale: f32) -> u16 {
    ((v - min) / scale).round() as u16
}

fn quantize_f32_to_bytes(values: &[f32], min: f32, scale: f32) -> Vec<u8> {
    if cfg!(feature = "quant-8") {
        values
            .iter()
            .flat_map(|&v| quantize_u8(v, min, scale).to_le_bytes())
            .collect()
    } else if cfg!(feature = "quant-16") {
        values
            .iter()
            .flat_map(|&v| quantize_u16(v, min, scale).to_le_bytes())
            .collect()
    } else {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

fn format_quantized_param(values: &[f32], min: f32, scale: f32) -> String {
    if cfg!(feature = "quant-8") {
        let q: Vec<u8> = values.iter().map(|&v| quantize_u8(v, min, scale)).collect();
        format!("{q:?}")
    } else if cfg!(feature = "quant-16") {
        let q: Vec<u16> = values.iter().map(|&v| quantize_u16(v, min, scale)).collect();
        format!("{q:?}")
    } else {
        format!("{values:?}")
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Download `url` to `path` if not already cached.
/// When `expected_sha256` is `Some`, verify the hash on both the cached file and
/// the freshly downloaded bytes; when `None`, skip hashing (warn emitted upstream).
fn download_if_missing(path: &Path, url: &str, expected_sha256: Option<&str>) {
    if path.exists() {
        if let Some(sha) = expected_sha256 {
            let existing = fs::read(path).expect("failed to read cached file");
            let actual = sha256_hex(&existing);
            assert!(
                actual == sha,
                "SHA-256 mismatch for cached file {}\n  expected: {}\n  actual:   {}\n  url: {}\n  hint: the cache may be poisoned or truncated; delete {} and rebuild.",
                path.display(), sha, actual, url, path.display()
            );
        }
        return;
    }
    eprintln!("cargo:warning=Downloading {}...", path.display());
    let resp = minreq::get(url).send().expect("download failed");
    assert!(resp.status_code == 200, "HTTP {}", resp.status_code);
    let bytes = resp.as_bytes();
    if let Some(sha) = expected_sha256 {
        let actual = sha256_hex(bytes);
        assert!(
            actual == sha,
            "SHA-256 mismatch for downloaded file {}\n  expected: {}\n  actual:   {}\n  url: {}\n  hint: the upstream or cache may be poisoned.",
            path.display(), sha, actual, url
        );
    }
    fs::write(path, bytes).unwrap();
}

fn trunc_dims() -> u64 {
    if cfg!(feature = "dim-1024") {
        1024
    } else if cfg!(feature = "dim-768") {
        768
    } else if cfg!(feature = "dim-512") {
        512
    } else if cfg!(feature = "dim-256") {
        256
    } else if cfg!(feature = "dim-128") {
        128
    } else if cfg!(feature = "dim-64") {
        64
    } else if cfg!(feature = "dim-32") {
        32
    } else {
        512
    }
}

fn dtype() -> &'static str {
    if cfg!(feature = "quant-16") {
        "u16"
    } else if cfg!(feature = "quant-8") {
        "u8"
    } else {
        "f32"
    }
}

fn dtype_sz() -> usize {
    if cfg!(feature = "quant-16") {
        2
    } else if cfg!(feature = "quant-8") {
        1
    } else {
        4
    }
}

fn shared_model_dir(out: &Path, cache_key: &str) -> PathBuf {
    let dir = env::var_os("ESE_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            out.ancestors()
                .find(|p| p.file_name().map_or(false, |f| f == "target"))
                .map(|target_dir| target_dir.join(concat!(env!("CARGO_PKG_NAME"), "-cache")))
        })
        .unwrap_or_else(|| {
            env::temp_dir()
                .join("ese-cache")
                .join(env!("CARGO_PKG_NAME"))
        });
    let dir = dir.join(cache_key);
    fs::create_dir_all(&dir)
        .unwrap_or_else(|err| panic!("failed to create cache dir {}: {err}", dir.display()));
    dir
}

#[cfg(feature = "tests")]
fn oracle_weight(w: f32, global_min: f32, quant_scale: f32) -> f32 {
    if cfg!(feature = "quant-8") {
        global_min + quantize_u8(w, global_min, quant_scale) as f32 * quant_scale
    } else if cfg!(feature = "quant-16") {
        global_min + quantize_u16(w, global_min, quant_scale) as f32 * quant_scale
    } else {
        w
    }
}

#[cfg(feature = "tests")]
fn normalize_vec(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vector {
            *v /= norm;
        }
    }
}

#[cfg(feature = "tests")]
fn generate_testdata(
    out: &Path,
    weights: &[Vec<f32>],
    _dims: usize,
    tokenizer_path: &Path,
    global_min: f32,
    quant_scale: f32,
    add_special_tokens: bool,
    normalize_output: bool,
) {
    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_path).expect("failed to load tokenizer");

    let inputs: Vec<&str> = vec![
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "",
        "   ",
        "café résumé naïve",
        "你好世界",
        "Hello, world! How are you?",
        "UPPER CASE text",
        "multiple   spaces   between",
        "a]b[c{d}e",
        "\x00\x01\x02\x03",
    ];

    let mut code = String::from("const TEST_INPUTS: &[&str] = &[\n");
    let mut outputs: Vec<Vec<f32>> = Vec::new();

    for &input in &inputs {
        code.push_str(&format!("    {:?},\n", input));
        let tokens = tokenizer.encode(input, add_special_tokens).unwrap();
        let token_ids = tokens.get_ids();
        let mut vector = vec![0.0f32; _dims];
        for &tok_id in token_ids {
            for (v, &w) in vector
                .iter_mut()
                .zip(unsafe { weights.get_unchecked(tok_id as usize) }.iter())
            {
                *v += oracle_weight(w, global_min, quant_scale);
            }
        }
        let scale = if token_ids.is_empty() { 1.0 } else { 1.0 / token_ids.len() as f32 };
        for v in vector.iter_mut() {
            *v *= scale;
        }
        if normalize_output {
            normalize_vec(&mut vector);
        }
        outputs.push(vector);
    }

    code.push_str("];\n\n");
    code.push_str(&format!(
        "const TEST_OUTPUTS: &[[f32; {}]; {}] = &[\n",
        _dims,
        outputs.len()
    ));
    for vec in &outputs {
        code.push_str(&format!("    {:?},\n", vec.as_slice()));
    }
    code.push_str("];\n");

    fs::write(out.join("optional_testdata.rs"), code).unwrap();
}
