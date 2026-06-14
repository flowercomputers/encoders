use base64::Engine as _;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    env,
    fs::{self},
    path::{Path, PathBuf},
};

/// Result of guard_tokenizer_family: identifies which build path to take.
enum TokenizerFamily {
    /// WordPiece (BERT-style, continuing_subword_prefix "##").
    WordPiece,
    /// Unigram (SentencePiece/Viterbi). Carries METASPACE_SPLIT_WHITESPACE.
    Unigram { metaspace_split_whitespace: bool },
    /// BPE with ByteLevel pre-tokenizer. Carries add_prefix_space from tokenizer.json.
    Bpe { add_prefix_space: bool },
}

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
    println!("cargo:rerun-if-env-changed=ESE_SKIP_UNK");
    println!("cargo:rerun-if-env-changed=ESE_ADD_SPECIAL_TOKENS");

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
    let family = guard_tokenizer_family(&tk, &model.name);

    // NORMALIZE_OUTPUT: env var ESE_NORMALIZE_OUTPUT=1
    let normalize_output = env::var("ESE_NORMALIZE_OUTPUT").map(|v| v == "1").unwrap_or(false);
    // SKIP_UNK: env var ESE_SKIP_UNK=1 — exclude UNK-id rows from mean pooling,
    // matching model2vec.StaticModel.tokenize() which filters UNK before pooling.
    // Only set this for distilled model2vec artifacts; default (false) keeps the
    // original ese behavior (UNK rows are included in the mean).
    let skip_unk = env::var("ESE_SKIP_UNK").map(|v| v == "1").unwrap_or(false);

    // quant_max is shared by the packer, oracle, and generate_testdata.
    let quant_max: f32 = if cfg!(feature = "quant-8") {
        255.0
    } else if cfg!(feature = "quant-16") {
        65535.0
    } else {
        1.0 // f32 build: unused (row_quant_params called but oracle_weight returns w)
    };

    match &family {
        TokenizerFamily::Unigram { metaspace_split_whitespace } => {
            build_unigram(&out, &tk, &model.name, &weights, dims as usize, normalize_output, skip_unk, *metaspace_split_whitespace);
            #[cfg(feature = "tests")]
            {
                let unk_id = tk["model"]["unk_id"].as_u64().unwrap_or(0) as usize;
                generate_testdata(
                    &out,
                    &weights,
                    dims as usize,
                    &tokenizer_path,
                    quant_max,
                    false, // Unigram models never add CLS/SEP
                    normalize_output,
                    skip_unk,
                    unk_id,
                );
            }
            return;
        }
        TokenizerFamily::Bpe { add_prefix_space } => {
            build_bpe(&out, &tk, &model.name, &weights, dims as usize, normalize_output, skip_unk, *add_prefix_space);
            #[cfg(feature = "tests")]
            {
                // BPE models (distilled from model2vec) use unk_token from tokenizer.json.
                // For SKIP_UNK purposes, find the unk_id from the vocab.
                let unk_token = tk["model"]["unk_token"].as_str().unwrap_or("<unk>");
                let vocab = tk["model"]["vocab"].as_object().expect("BPE model.vocab must be an object");
                let unk_id = vocab.get(unk_token).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                generate_testdata(
                    &out,
                    &weights,
                    dims as usize,
                    &tokenizer_path,
                    quant_max,
                    false, // BPE (model2vec distilled) models do not add CLS/SEP
                    normalize_output,
                    skip_unk,
                    unk_id,
                );
            }
            return;
        }
        TokenizerFamily::WordPiece => {
            // Fall through to WordPiece path below.
        }
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

    // ADD_SPECIAL_TOKENS: env-based opt-in, default false.
    //
    // Static-embedding references (sentence-transformers StaticEmbedding AND
    // model2vec.StaticModel) do NOT pool special tokens: mean-pool of content
    // tokens only reproduces SentenceTransformer.encode() at cosine 1.0000,
    // while pooling WITH [CLS]/[SEP] lands at 0.9994 (S-matrix geometry error
    // 0.00141 vs 0.00000). Verified in spikes/specials-parity; see also
    // MinishLab/model2vec#335. Specials-off is therefore the correct default;
    // set ESE_ADD_SPECIAL_TOKENS=1 only if you have a specific model that
    // genuinely requires CLS/SEP in the pooled mean (rare).
    let add_special_tokens = env::var("ESE_ADD_SPECIAL_TOKENS").map(|v| v == "1").unwrap_or(false);

    // When the opt-in is set, cross-check that post_processor special_token ids
    // match the vocab — meaningful only when specials are actually pooled.
    if add_special_tokens {
        cross_check_special_token_ids(&tk["post_processor"], Some(vocab), &model.name);
    }

    let cls_id = if add_special_tokens {
        Some(special_token_id(vocab, "[CLS]").unwrap_or_else(|| panic!(
            "model '{}': ESE_ADD_SPECIAL_TOKENS=1 but '[CLS]' is missing from vocab",
            model.name
        )))
    } else {
        None
    };
    let sep_id = if add_special_tokens {
        Some(special_token_id(vocab, "[SEP]").unwrap_or_else(|| panic!(
            "model '{}': ESE_ADD_SPECIAL_TOKENS=1 but '[SEP]' is missing from vocab",
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

    // global_min / quant_scale: kept for backward compat (emitted as consts for f32 builds;
    // quant builds still emit them but the runtime reads per-row params from the slot instead).
    let (global_min, quant_scale) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for row in &weights {
            for &v in row {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        let scale = if (mx - mn).abs() < f32::EPSILON { 1.0 } else { (mx - mn) / quant_max };
        (mn, scale)
    } else {
        (0.0_f32, 1.0_f32) // f32 build: const pass-through (0.0 / 1.0 → identity)
    };

    // reorder weights into PHF slots with verify hash (and per-row params in quant builds).
    //
    // quant builds (quant-8 / quant-16):
    //   slot layout: { verify: u64, row_min: f32, row_scale: f32, weight: [uN; DIMS] }
    //   SLOT_SIZE   = 8 + 8 + DIMENSIONS * DTYPE_SIZE
    //
    // f32 build:
    //   slot layout: { verify: u64, weight: [f32; DIMS] }
    //   SLOT_SIZE   = 8 + DIMENSIONS * DTYPE_SIZE  (unchanged)
    let slot_size = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        8 + 8 + dims as usize * dtype_sz() // verify + row_min + row_scale + weights
    } else {
        8 + dims as usize * dtype_sz() // verify + weights
    };
    // Padding check: SLOT_SIZE must be a multiple of 8 (align(8) on the Slot struct).
    // For quant builds: 8 + 8 + DIMS * dtype_sz. DIMS is a multiple of 8 (all supported dims).
    // quant-8:  16 + DIMS*1 — multiple of 8 when DIMS%8==0 ✓
    // quant-16: 16 + DIMS*2 — multiple of 8 when DIMS%8==0 ✓
    // f32:       8 + DIMS*4 — multiple of 8 when DIMS%8==0 ✓
    assert_eq!(slot_size % 8, 0, "slot_size={slot_size} is not a multiple of 8 for dims={dims}");

    let mut phf_slots: Vec<u8> = Vec::with_capacity(table_size * slot_size);
    for slot in 0..table_size {
        phf_slots.extend_from_slice(&verify[slot].to_le_bytes());
        let row = if occupied[slot] {
            &weights[slot_to_weight[slot]]
        } else {
            &zero_row
        };
        // In quant builds, write per-row params before the quantized weights,
        // then quantize the weights using those per-row params.
        // In f32 builds, skip per-row params (unchanged layout); write raw f32.
        if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
            // Empty slots get (0.0, 1.0) — dequantize(0) = 0.0 * 1.0 + 0.0 = 0.0.
            let (row_min, row_scale) = if occupied[slot] {
                row_quant_params(row, quant_max)
            } else {
                (0.0_f32, 1.0_f32)
            };
            phf_slots.extend_from_slice(&row_min.to_le_bytes());
            phf_slots.extend_from_slice(&row_scale.to_le_bytes());
            // Quantize weights with per-row params (not global).
            phf_slots.extend_from_slice(&quantize_f32_to_bytes(row, row_min, row_scale));
        } else {
            // f32 build: raw f32 weights, no per-row params.
            phf_slots.extend_from_slice(&quantize_f32_to_bytes(row, global_min, quant_scale));
        }
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

    // Per-row params for the special tokens (UNK/CLS/SEP).
    // In quant builds, each special token's row has its own (min, scale) pair, just
    // like slot rows. The runtime reads these consts directly (no slot lookup for specials).
    // In f32 builds these are (0.0, 1.0) identity consts — the compiler eliminates them.
    let (unk_min, unk_scale) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        row_quant_params(&unk, quant_max)
    } else {
        (0.0_f32, 1.0_f32)
    };
    let (cls_min, cls_scale) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        row_quant_params(&cls, quant_max)
    } else {
        (0.0_f32, 1.0_f32)
    };
    let (sep_min, sep_scale) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        row_quant_params(&sep, quant_max)
    } else {
        (0.0_f32, 1.0_f32)
    };

    // Quantize special-token rows with their own per-row params.
    let unk_str = format_quantized_param(&unk, unk_min, unk_scale);
    let cls_str = format_quantized_param(&cls, cls_min, cls_scale);
    let sep_str = format_quantized_param(&sep, sep_min, sep_scale);

    let dtype_sz = dtype_sz();

    // SLOT_SIZE: in quant builds the slot carries two extra f32 fields (row_min, row_scale).
    // In f32 builds the layout is unchanged (verify:u64 + weight:[f32; DIMS]).
    // The emitted const must match the Slot struct in src/lookup.rs exactly.
    let slot_size_const = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        "8 + 8 + DIMENSIONS * DTYPE_SIZE".to_string() // verify + row_min + row_scale + weights
    } else {
        "8 + DIMENSIONS * DTYPE_SIZE".to_string() // f32 build: unchanged
    };

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
pub const SLOT_SIZE: usize = {slot_size_const};\n\
pub const UNK: Param = {unk_str};\n\
pub const CLS: Param = {cls_str};\n\
pub const SEP: Param = {sep_str};\n\
// Per-row quantization params for the special tokens (UNK/CLS/SEP).
// In quant builds the runtime uses these consts directly when accumulating special tokens.
// In f32 builds these are (0.0, 1.0) identity values — the compiler elides them.
#[allow(dead_code)] pub const UNK_MIN: f32 = {unk_min:?};\n\
#[allow(dead_code)] pub const UNK_SCALE: f32 = {unk_scale:?};\n\
#[allow(dead_code)] pub const CLS_MIN: f32 = {cls_min:?};\n\
#[allow(dead_code)] pub const CLS_SCALE: f32 = {cls_scale:?};\n\
#[allow(dead_code)] pub const SEP_MIN: f32 = {sep_min:?};\n\
#[allow(dead_code)] pub const SEP_SCALE: f32 = {sep_scale:?};\n\
pub const MAX_WORD_LEN: usize = {max_word_len};\n\
pub const CONTINUING_PREFIX: &str = {continuing_prefix:?};\n\
pub const ADD_SPECIAL_TOKENS: bool = {add_special_tokens};\n\
pub const NORMALIZE_OUTPUT: bool = {normalize_output};\n\
// When SKIP_UNK=true, UNK-id rows are excluded from pooling (neither accumulated nor counted).
// Matches model2vec.StaticModel.tokenize() behavior for distilled model2vec artifacts.
// Default (false) pools UNK rows, which is correct for sentence-transformers static models.
#[allow(dead_code)] pub const SKIP_UNK: bool = {skip_unk};\n\
pub const DTYPE_SIZE: usize = {dtype_sz};\n\
// QUANT_MIN / QUANT_SCALE: global params retained for the f32 build (0.0 / 1.0 = identity).
// In quant builds, slot weights are decoded with the per-row params stored in each Slot;
// these global consts are no longer used for weight decoding in quant builds.
#[allow(dead_code)] pub const QUANT_MIN: f32 = {global_min:?};\n\
#[allow(dead_code)] pub const QUANT_SCALE: f32 = {quant_scale:?};\n\
pub const MODEL_NAME: &str = {model_name:?};\n\
// Unigram-specific constants. Dead in WordPiece builds; the compiler elides the
// Unigram code path via the IS_UNIGRAM const branch in encode_single.
#[allow(dead_code)] pub const IS_UNIGRAM: bool = false;\n\
#[allow(dead_code)] pub const MIN_SCORE: f32 = 0.0;\n\
#[allow(dead_code)] pub const MAX_TOKEN_CHARS: usize = 0;\n\
#[allow(dead_code)] pub const UNK_ID: usize = {unk_id};\n\
#[allow(dead_code)] pub const PAD_CHARS: &[char] = &[];\n\
#[allow(dead_code)] pub const COLLAPSE_SPACES: bool = false;\n\
#[allow(dead_code)] pub const COLLAPSE_WHITESPACE: bool = false;\n\
#[allow(dead_code)] pub const STRIP_LEFT: bool = false;\n\
#[allow(dead_code)] pub const STRIP_RIGHT: bool = false;\n\
#[allow(dead_code)] pub const METASPACE_CHAR: char = '\\u{{2581}}';\n\
#[allow(dead_code)] pub const METASPACE_PREPEND: bool = false;\n\
// METASPACE_SPLIT_WHITESPACE: dead in WordPiece builds (IS_UNIGRAM=false).\n\
#[allow(dead_code)] pub const METASPACE_SPLIT_WHITESPACE: bool = false;\n\
// BPE-specific constants. Dead in WordPiece builds (IS_BPE=false elides the bpe path).\n\
#[allow(dead_code)] pub const IS_BPE: bool = false;\n\
#[allow(dead_code)] pub const IS_BPE_ADD_PREFIX_SPACE: bool = false;\n\
#[allow(dead_code)] pub const MERGE_NUM_BUCKETS: usize = 1;\n\
#[allow(dead_code)] pub const MERGE_TABLE_SIZE: usize = 1;\n\
#[allow(dead_code)] pub const MERGE_TABLE_MASK: usize = 0;\n\
",
            model_name = model.name,
        ),
    )
    .unwrap();

    // Emit dummy files for WordPiece builds (non-BPE/non-Unigram paths are dead code).
    // scores.bin and charsmap.bin: Unigram-specific, dummy for WordPiece.
    // slot_ids.bin, merges.bin, merge_seeds.bin, merge_verify.bin: BPE-specific, dummy.
    fs::write(out.join("scores.bin"), &[0u8; 4]).unwrap();
    fs::write(out.join("charsmap.bin"), &[0u8; 0]).unwrap();
    fs::write(out.join("slot_ids.bin"), &[0u8; 4]).unwrap();
    fs::write(out.join("merges.bin"), &[0u8; 8]).unwrap();
    fs::write(out.join("merge_seeds.bin"), &[0u8; 4]).unwrap();
    fs::write(out.join("merge_verify.bin"), &[0u8; 8]).unwrap();

    #[cfg(feature = "tests")]
    generate_testdata(
        &out,
        &weights,
        dims as usize,
        &tokenizer_path,
        quant_max,
        add_special_tokens,
        normalize_output,
        skip_unk,
        unk_id,
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
    skip_unk: bool,
    metaspace_split_whitespace: bool,
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

    let mut min_score = 0.0f32; // tracks the most-negative vocab log-prob score
    for (id, entry) in vocab_arr.iter().enumerate() {
        let token = entry[0].as_str().unwrap_or("").to_string();
        let score = entry[1].as_f64().unwrap_or(0.0) as f32;
        let char_len = token.chars().count();
        if char_len > max_token_chars {
            max_token_chars = char_len;
        }
        if score < min_score {
            min_score = score;
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

    // ── Emit weights.bin (same layout as WordPiece, including per-row params in quant builds) ─
    // Unigram models from model2vec are always F32, but quant builds apply quantization
    // orthogonally. The slot layout matches the WordPiece path so src/lookup.rs is shared.
    //
    // quant builds: slot = { verify:u64, row_min:f32, row_scale:f32, weight:[uN; DIMS] }
    // f32 build:    slot = { verify:u64, weight:[f32; DIMS] }
    let quant_max_uni: f32 = if cfg!(feature = "quant-8") { 255.0 }
        else if cfg!(feature = "quant-16") { 65535.0 }
        else { 1.0 };
    let slot_size_uni = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        8 + 8 + dims * dtype_sz() // verify + row_min + row_scale + weights
    } else {
        8 + dims * 4 // verify + raw f32 weights
    };
    let zero_row = vec![0.0f32; dims];
    let mut phf_slots: Vec<u8> = Vec::with_capacity(table_size * slot_size_uni);

    for slot in 0..table_size {
        phf_slots.extend_from_slice(&verify[slot].to_le_bytes());
        let row = if occupied[slot] {
            &weights[slot_to_weight[slot]]
        } else {
            &zero_row
        };
        if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
            let (row_min, row_scale) = if occupied[slot] {
                row_quant_params(row, quant_max_uni)
            } else {
                (0.0_f32, 1.0_f32)
            };
            phf_slots.extend_from_slice(&row_min.to_le_bytes());
            phf_slots.extend_from_slice(&row_scale.to_le_bytes());
            phf_slots.extend_from_slice(&quantize_f32_to_bytes(row, row_min, row_scale));
        } else {
            phf_slots.extend_from_slice(&quantize_f32_to_bytes(row, 0.0, 1.0));
        }
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

    // ── ADD_SPECIAL_TOKENS for Unigram models ────────────────────────────────
    // Unigram (model2vec distilled) models never pool specials. The env opt-in
    // is read here for consistency, but it is always false in practice and we
    // guard against any accidental activation.
    let add_special_tokens = env::var("ESE_ADD_SPECIAL_TOKENS").map(|v| v == "1").unwrap_or(false);
    // Guard: Unigram models must not add special tokens (the unk row is the only special)
    assert!(!add_special_tokens, "Unigram model '{}': ESE_ADD_SPECIAL_TOKENS=1 is not supported for Unigram models (specials are never pooled by model2vec artifacts)", model_name);

    // Emit the UNK embedding as the constant (used for UNK pooling).
    let unk_row = if unk_id < weights.len() { &weights[unk_id] } else { &zero_row };
    let dtype = dtype();
    let dtype_sz = dtype_sz();

    // Per-row params for UNK (CLS/SEP are zero for Unigram models — (0.0, 1.0) identity).
    let (unk_min_uni, unk_scale_uni) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        row_quant_params(unk_row, quant_max_uni)
    } else {
        (0.0_f32, 1.0_f32)
    };
    let unk_str = format_quantized_param(unk_row, unk_min_uni, unk_scale_uni);
    let zero_str = format_quantized_param(&zero_row, 0.0, 1.0);

    // SLOT_SIZE must match the Slot struct in src/lookup.rs.
    let slot_size_const_uni = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        "8 + 8 + DIMENSIONS * DTYPE_SIZE".to_string()
    } else {
        "8 + DIMENSIONS * DTYPE_SIZE".to_string()
    };

    // Global params for the emitted QUANT_MIN/QUANT_SCALE consts (retained for compat;
    // quant builds no longer use them for weight decoding).
    let (global_min_uni, quant_scale_uni) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for row in weights {
            for &v in row {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        let s = if (mx - mn).abs() < f32::EPSILON { 1.0f32 } else { (mx - mn) / quant_max_uni };
        (mn, s)
    } else {
        (0.0_f32, 1.0_f32)
    };

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
pub const SLOT_SIZE: usize = {slot_size_const_uni};\n\
pub const UNK: Param = {unk_str};\n\
pub const CLS: Param = {zero_str};\n\
pub const SEP: Param = {zero_str};\n\
// Per-row quantization params for special tokens. CLS/SEP are zero for Unigram models.\n\
#[allow(dead_code)] pub const UNK_MIN: f32 = {unk_min_uni:?};\n\
#[allow(dead_code)] pub const UNK_SCALE: f32 = {unk_scale_uni:?};\n\
#[allow(dead_code)] pub const CLS_MIN: f32 = 0.0;\n\
#[allow(dead_code)] pub const CLS_SCALE: f32 = 1.0;\n\
#[allow(dead_code)] pub const SEP_MIN: f32 = 0.0;\n\
#[allow(dead_code)] pub const SEP_SCALE: f32 = 1.0;\n\
pub const MAX_WORD_LEN: usize = 0;\n\
pub const CONTINUING_PREFIX: &str = \"##\";\n\
pub const ADD_SPECIAL_TOKENS: bool = false;\n\
pub const NORMALIZE_OUTPUT: bool = {normalize_output};\n\
// When SKIP_UNK=true, UNK rows are excluded from pooling (neither accumulated nor counted).
// Matches model2vec.StaticModel.tokenize() behavior for distilled model2vec artifacts.
// Default (false) pools UNK rows, which is correct for sentence-transformers static models.
#[allow(dead_code)] pub const SKIP_UNK: bool = {skip_unk};\n\
pub const DTYPE_SIZE: usize = {dtype_sz};\n\
#[allow(dead_code)] pub const QUANT_MIN: f32 = {global_min_uni:?};\n\
#[allow(dead_code)] pub const QUANT_SCALE: f32 = {quant_scale_uni:?};\n\
pub const MODEL_NAME: &str = {model_name:?};\n\
// Unigram-specific constants. Some may be dead in certain feature combinations.\n\
#[allow(dead_code)] pub const IS_UNIGRAM: bool = true;\n\
#[allow(dead_code)] pub const MIN_SCORE: f32 = {min_score:?};\n\
#[allow(dead_code)] pub const MAX_TOKEN_CHARS: usize = {max_token_chars};\n\
#[allow(dead_code)] pub const UNK_ID: usize = {unk_id};\n\
#[allow(dead_code)] pub const PAD_CHARS: &[char] = &[{pad_chars_str}];\n\
#[allow(dead_code)] pub const COLLAPSE_SPACES: bool = {collapse_spaces};\n\
#[allow(dead_code)] pub const COLLAPSE_WHITESPACE: bool = {collapse_whitespace};\n\
#[allow(dead_code)] pub const STRIP_LEFT: bool = {strip_left};\n\
#[allow(dead_code)] pub const STRIP_RIGHT: bool = {strip_right};\n\
#[allow(dead_code)] pub const METASPACE_CHAR: char = '\\u{{2581}}';\n\
#[allow(dead_code)] pub const METASPACE_PREPEND: bool = true;\n\
// METASPACE_SPLIT_WHITESPACE: when true, normalize() output is split on Unicode\n\
// whitespace (char::is_whitespace), each word has Metaspace prepended independently,\n\
// and Viterbi is run per-word. This implements the Sequence([WhitespaceSplit, Metaspace])\n\
// pre-tokenizer shape used by XLM-R-family teachers. When false (bare Metaspace), the\n\
// entire normalized string is passed as one chunk (existing behavior, unchanged).\n\
#[allow(dead_code)] pub const METASPACE_SPLIT_WHITESPACE: bool = {metaspace_split_whitespace};\n\
// BPE-specific constants. Dead in Unigram builds (IS_BPE=false elides the bpe path).\n\
#[allow(dead_code)] pub const IS_BPE: bool = false;\n\
#[allow(dead_code)] pub const IS_BPE_ADD_PREFIX_SPACE: bool = false;\n\
#[allow(dead_code)] pub const MERGE_NUM_BUCKETS: usize = 1;\n\
#[allow(dead_code)] pub const MERGE_TABLE_SIZE: usize = 1;\n\
#[allow(dead_code)] pub const MERGE_TABLE_MASK: usize = 0;\n\
",
        ),
    )
    .unwrap();

    // Emit BPE-specific dummy binary files for Unigram builds (dead code, but
    // include_bytes! in src/bpe.rs must compile even when IS_BPE=false).
    fs::write(out.join("slot_ids.bin"), &[0u8; 4]).unwrap();
    fs::write(out.join("merges.bin"), &[0u8; 8]).unwrap();
    fs::write(out.join("merge_seeds.bin"), &[0u8; 4]).unwrap();
    fs::write(out.join("merge_verify.bin"), &[0u8; 8]).unwrap();
}

// ── BPE build path ────────────────────────────────────────────────────────────

/// Build all artifacts for a BPE (ByteLevel) model:
///   - weights.bin / seeds.bin / verify.bin: vocab PHF (same layout as WordPiece/Unigram)
///   - slot_ids.bin: [u32; TABLE_SIZE] mapping PHF slot → original vocab token id
///   - merges.bin / merge_seeds.bin / merge_verify.bin: merge pair PHF
///   - scores.bin / charsmap.bin: dummy files (not used by BPE)
///   - model_constants.rs: BPE-specific consts (IS_BPE=true, MERGE_*, IS_BPE_ADD_PREFIX_SPACE)
///
/// The merge PHF key is (left_vocab_id as u64 << 32 | right_vocab_id as u64), encoded as
/// 8-byte LE, using the same hash polynomial as the vocab PHF. Value per slot:
///   rank:u32le + new_id:u32le (8 bytes total).
/// Merge rank = index in the tokenizer.json "merges" array (0-based), exactly matching
/// the vendored BPE model.rs:179 (i as u32 → rank) and word.rs merge_all ordering.
#[allow(clippy::too_many_arguments)]
fn build_bpe(
    out: &Path,
    tk: &serde_json::Value,
    model_name: &str,
    weights: &[Vec<f32>],
    dims: usize,
    normalize_output: bool,
    skip_unk: bool,
    add_prefix_space: bool,
) {
    // ── Parse vocab: object of token_string → vocab_id ───────────────────────
    let vocab_obj = tk["model"]["vocab"]
        .as_object()
        .expect("BPE model.vocab must be an object (token → id)");

    // Build (token_string, vocab_id) pairs sorted by vocab_id for determinism.
    let mut entries: Vec<(String, usize)> = vocab_obj
        .iter()
        .filter_map(|(tok, id_val)| {
            let id = id_val.as_u64()? as usize;
            Some((tok.clone(), id))
        })
        .collect();
    entries.sort_by_key(|(_, id)| *id);

    // UNK token id (for UNK embedding extraction and SKIP_UNK oracle).
    let unk_token = tk["model"]["unk_token"].as_str().unwrap_or("<unk>");
    let unk_id = vocab_obj
        .get(unk_token)
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let zero_row = vec![0.0f32; dims];
    let unk_row = if unk_id < weights.len() { &weights[unk_id] } else { &zero_row };

    // ── PHF construction (same algorithm as WordPiece and Unigram) ─────────
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
    let mut slot_to_entry: Vec<usize> = vec![0; table_size]; // slot → entries index
    let mut slot_to_vocab_id: Vec<u32> = vec![u32::MAX; table_size]; // slot → vocab_id
    let mut occupied: Vec<bool> = vec![false; table_size];

    for &bi in &bucket_order {
        if buckets[bi].is_empty() {
            continue;
        }
        let mut i: u32 = 0;
        'search: loop {
            let seed = i | 1;
            i += 1;
            assert!(i < 1_000_000, "BPE PHF construction stuck on bucket {bi}");

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
                slot_to_entry[slot] = entries[entry_idx].1; // vocab_id as weight row index
                slot_to_vocab_id[slot] = entries[entry_idx].1 as u32;
            }
            break;
        }
    }

    // ── Emit slot_ids.bin: [u32; TABLE_SIZE] (slot → vocab_id) ──────────────
    let mut slot_ids_bin: Vec<u8> = Vec::with_capacity(table_size * 4);
    for slot in 0..table_size {
        slot_ids_bin.extend_from_slice(&slot_to_vocab_id[slot].to_le_bytes());
    }
    fs::write(out.join("slot_ids.bin"), &slot_ids_bin).unwrap();

    // ── Emit weights.bin (same layout as WordPiece/Unigram) ─────────────────
    let quant_max_bpe: f32 = if cfg!(feature = "quant-8") { 255.0 }
        else if cfg!(feature = "quant-16") { 65535.0 }
        else { 1.0 };
    let slot_size_bpe = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        8 + 8 + dims * dtype_sz()
    } else {
        8 + dims * 4
    };
    let mut phf_slots: Vec<u8> = Vec::with_capacity(table_size * slot_size_bpe);
    for slot in 0..table_size {
        phf_slots.extend_from_slice(&verify[slot].to_le_bytes());
        let row = if occupied[slot] {
            let vid = slot_to_entry[slot];
            if vid < weights.len() { &weights[vid] } else { &zero_row }
        } else {
            &zero_row
        };
        if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
            let (row_min, row_scale) = if occupied[slot] {
                row_quant_params(row, quant_max_bpe)
            } else {
                (0.0_f32, 1.0_f32)
            };
            phf_slots.extend_from_slice(&row_min.to_le_bytes());
            phf_slots.extend_from_slice(&row_scale.to_le_bytes());
            phf_slots.extend_from_slice(&quantize_f32_to_bytes(row, row_min, row_scale));
        } else {
            phf_slots.extend_from_slice(&quantize_f32_to_bytes(row, 0.0, 1.0));
        }
    }
    fs::write(out.join("weights.bin"), &phf_slots).unwrap();

    // seeds.bin and verify.bin
    let mut seeds_bin: Vec<u8> = Vec::with_capacity(num_buckets * 4);
    for &s in &seeds { seeds_bin.extend_from_slice(&s.to_le_bytes()); }
    fs::write(out.join("seeds.bin"), &seeds_bin).unwrap();
    let mut verify_bin: Vec<u8> = Vec::with_capacity(table_size * 8);
    for &v in &verify { verify_bin.extend_from_slice(&v.to_le_bytes()); }
    fs::write(out.join("verify.bin"), &verify_bin).unwrap();

    // ── Parse merges: array of ["left", "right"] pairs in rank order ─────────
    // model.rs:174-192: merges[i] = (a, b) → rank=i, new_id=vocab[a+b] (or a+b[prefix_len..])
    // For ByteLevel BPE: continuing_subword_prefix is absent (no prefix). new_token = a + b.
    // We store (left_vocab_id, right_vocab_id) → (rank, new_id) in the merge PHF.
    let merges_arr = tk["model"]["merges"]
        .as_array()
        .expect("BPE model.merges must be an array");

    // Build merge entries: (key_u64, rank, new_id)
    let mut merge_entries: Vec<(u64, u32, u32)> = Vec::with_capacity(merges_arr.len());

    // continuing_subword_prefix: absent for ByteLevel BPE → "" (no prefix stripped).
    let prefix_len = tk["model"]["continuing_subword_prefix"]
        .as_str()
        .map(|s| s.len())
        .unwrap_or(0);

    for (rank, merge_val) in merges_arr.iter().enumerate() {
        // Each merge entry is either ["a", "b"] or "a b"
        let (left_str, right_str) = if let Some(arr) = merge_val.as_array() {
            (
                arr.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                arr.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string(),
            )
        } else if let Some(s) = merge_val.as_str() {
            // "a b" form
            let mut parts = s.splitn(2, ' ');
            (
                parts.next().unwrap_or("").to_string(),
                parts.next().unwrap_or("").to_string(),
            )
        } else {
            panic!("BPE model '{}': merge entry at rank {} is neither array nor string", model_name, rank);
        };

        // Look up left_id and right_id in the vocab.
        let left_id = vocab_obj.get(&left_str)
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!(
                "BPE model '{}': merge rank {} left token '{}' not in vocab",
                model_name, rank, left_str
            )) as u32;
        let right_id = vocab_obj.get(&right_str)
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!(
                "BPE model '{}': merge rank {} right token '{}' not in vocab",
                model_name, rank, right_str
            )) as u32;

        // new_token = left_str + right_str[prefix_len..] (model.rs:186).
        let new_token = format!("{}{}", left_str, &right_str[prefix_len..]);
        let new_id = vocab_obj.get(&new_token)
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!(
                "BPE model '{}': merge rank {} new token '{}' not in vocab",
                model_name, rank, new_token
            )) as u32;

        let key_u64: u64 = (left_id as u64) << 32 | right_id as u64;
        merge_entries.push((key_u64, rank as u32, new_id));
    }

    // ── Merge PHF construction ────────────────────────────────────────────────
    let merge_count = merge_entries.len();
    let merge_table_size = merge_count.next_power_of_two();
    let merge_table_mask = merge_table_size - 1;
    let merge_num_buckets = merge_count;

    let mut m_buckets: Vec<Vec<usize>> = vec![vec![]; merge_num_buckets];
    for (i, (key, _, _)) in merge_entries.iter().enumerate() {
        let b = (phf_hash(&key.to_le_bytes(), 0) as usize) % merge_num_buckets;
        m_buckets[b].push(i);
    }
    let mut m_bucket_order: Vec<usize> = (0..merge_num_buckets).collect();
    m_bucket_order.sort_by(|a, b| m_buckets[*b].len().cmp(&m_buckets[*a].len()));

    let mut m_seeds: Vec<u32> = vec![0; merge_num_buckets];
    let mut m_verify: Vec<u64> = vec![0; merge_table_size];
    let mut m_values: Vec<(u32, u32)> = vec![(0, 0); merge_table_size]; // (rank, new_id) per slot
    let mut m_occupied: Vec<bool> = vec![false; merge_table_size];

    for &bi in &m_bucket_order {
        if m_buckets[bi].is_empty() {
            continue;
        }
        let mut i: u32 = 0;
        'msearch: loop {
            let seed = i | 1;
            i += 1;
            assert!(i < 1_000_000, "BPE merge PHF stuck on bucket {bi}");

            let mut trial_slots: Vec<usize> = Vec::with_capacity(m_buckets[bi].len());
            let mut seen = HashSet::with_capacity(m_buckets[bi].len());

            for &entry_idx in &m_buckets[bi] {
                let key_bytes = merge_entries[entry_idx].0.to_le_bytes();
                let slot = (phf_hash(&key_bytes, seed as u64) as usize) & merge_table_mask;
                if m_occupied[slot] || !seen.insert(slot) {
                    continue 'msearch;
                }
                trial_slots.push(slot);
            }

            m_seeds[bi] = seed;
            for (j, &entry_idx) in m_buckets[bi].iter().enumerate() {
                let slot = trial_slots[j];
                let key_bytes = merge_entries[entry_idx].0.to_le_bytes();
                m_occupied[slot] = true;
                m_verify[slot] = phf_hash(&key_bytes, u64::MAX);
                m_values[slot] = (merge_entries[entry_idx].1, merge_entries[entry_idx].2);
            }
            break;
        }
    }

    // ── Emit merge PHF files ──────────────────────────────────────────────────
    let mut m_seeds_bin: Vec<u8> = Vec::with_capacity(merge_num_buckets * 4);
    for &s in &m_seeds { m_seeds_bin.extend_from_slice(&s.to_le_bytes()); }
    fs::write(out.join("merge_seeds.bin"), &m_seeds_bin).unwrap();

    let mut m_verify_bin: Vec<u8> = Vec::with_capacity(merge_table_size * 8);
    for &v in &m_verify { m_verify_bin.extend_from_slice(&v.to_le_bytes()); }
    fs::write(out.join("merge_verify.bin"), &m_verify_bin).unwrap();

    let mut m_data_bin: Vec<u8> = Vec::with_capacity(merge_table_size * 8);
    for &(rank, new_id) in &m_values {
        m_data_bin.extend_from_slice(&rank.to_le_bytes());
        m_data_bin.extend_from_slice(&new_id.to_le_bytes());
    }
    fs::write(out.join("merges.bin"), &m_data_bin).unwrap();

    // ── Emit scores.bin and charsmap.bin (dummies; not used by BPE) ──────────
    fs::write(out.join("scores.bin"), &[0u8; 4]).unwrap();
    fs::write(out.join("charsmap.bin"), &[0u8; 0]).unwrap();

    // ── Per-row quantization params for UNK (CLS/SEP dummy for BPE) ──────────
    let (unk_min_bpe, unk_scale_bpe) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        row_quant_params(unk_row, quant_max_bpe)
    } else {
        (0.0_f32, 1.0_f32)
    };
    let unk_str = format_quantized_param(unk_row, unk_min_bpe, unk_scale_bpe);
    let zero_str = format_quantized_param(&zero_row, 0.0, 1.0);

    let dtype = dtype();
    let dtype_sz = dtype_sz();
    let slot_size_const_bpe = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        "8 + 8 + DIMENSIONS * DTYPE_SIZE".to_string()
    } else {
        "8 + DIMENSIONS * DTYPE_SIZE".to_string()
    };

    let (global_min_bpe, quant_scale_bpe_g) = if cfg!(feature = "quant-8") || cfg!(feature = "quant-16") {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for row in weights { for &v in row { mn = mn.min(v); mx = mx.max(v); } }
        let s = if (mx - mn).abs() < f32::EPSILON { 1.0f32 } else { (mx - mn) / quant_max_bpe };
        (mn, s)
    } else {
        (0.0_f32, 1.0_f32)
    };

    // ── Emit model_constants.rs ───────────────────────────────────────────────
    fs::write(
        out.join("model_constants.rs"),
        format!(
            "\
pub const DIMENSIONS: usize = {dims};\n\
pub type Param = [{dtype}; DIMENSIONS];\n\
pub const TABLE_SIZE: usize = {table_size};\n\
pub const TABLE_MASK: usize = {table_mask};\n\
pub const NUM_BUCKETS: usize = {num_buckets};\n\
pub const SLOT_SIZE: usize = {slot_size_const_bpe};\n\
pub const UNK: Param = {unk_str};\n\
pub const CLS: Param = {zero_str};\n\
pub const SEP: Param = {zero_str};\n\
// Per-row quantization params for special tokens. CLS/SEP are zero for BPE models.\n\
#[allow(dead_code)] pub const UNK_MIN: f32 = {unk_min_bpe:?};\n\
#[allow(dead_code)] pub const UNK_SCALE: f32 = {unk_scale_bpe:?};\n\
#[allow(dead_code)] pub const CLS_MIN: f32 = 0.0;\n\
#[allow(dead_code)] pub const CLS_SCALE: f32 = 1.0;\n\
#[allow(dead_code)] pub const SEP_MIN: f32 = 0.0;\n\
#[allow(dead_code)] pub const SEP_SCALE: f32 = 1.0;\n\
pub const MAX_WORD_LEN: usize = 0;\n\
pub const CONTINUING_PREFIX: &str = \"##\";\n\
pub const ADD_SPECIAL_TOKENS: bool = false;\n\
pub const NORMALIZE_OUTPUT: bool = {normalize_output};\n\
#[allow(dead_code)] pub const SKIP_UNK: bool = {skip_unk};\n\
pub const DTYPE_SIZE: usize = {dtype_sz};\n\
#[allow(dead_code)] pub const QUANT_MIN: f32 = {global_min_bpe:?};\n\
#[allow(dead_code)] pub const QUANT_SCALE: f32 = {quant_scale_bpe_g:?};\n\
pub const MODEL_NAME: &str = {model_name:?};\n\
// Unigram-specific constants. Dead in BPE builds (IS_UNIGRAM=false, IS_BPE=true).\n\
#[allow(dead_code)] pub const IS_UNIGRAM: bool = false;\n\
#[allow(dead_code)] pub const MIN_SCORE: f32 = 0.0;\n\
#[allow(dead_code)] pub const MAX_TOKEN_CHARS: usize = 0;\n\
#[allow(dead_code)] pub const UNK_ID: usize = {unk_id};\n\
#[allow(dead_code)] pub const PAD_CHARS: &[char] = &[];\n\
#[allow(dead_code)] pub const COLLAPSE_SPACES: bool = false;\n\
#[allow(dead_code)] pub const COLLAPSE_WHITESPACE: bool = false;\n\
#[allow(dead_code)] pub const STRIP_LEFT: bool = false;\n\
#[allow(dead_code)] pub const STRIP_RIGHT: bool = false;\n\
#[allow(dead_code)] pub const METASPACE_CHAR: char = '\\u{{2581}}';\n\
#[allow(dead_code)] pub const METASPACE_PREPEND: bool = false;\n\
#[allow(dead_code)] pub const METASPACE_SPLIT_WHITESPACE: bool = false;\n\
// BPE-specific constants.\n\
pub const IS_BPE: bool = true;\n\
// IS_BPE_ADD_PREFIX_SPACE: when true, the ByteLevel pre-tokenizer prepends a space to\n\
// the input text before splitting if it doesn't already start with one. Controls whether\n\
// the first word gets the Ġ space-marker prefix (byte_level.rs:123-125).\n\
pub const IS_BPE_ADD_PREFIX_SPACE: bool = {add_prefix_space};\n\
// Merge PHF constants: MERGE_NUM_BUCKETS, MERGE_TABLE_SIZE, MERGE_TABLE_MASK.\n\
// The merge PHF maps (left_id<<32 | right_id) → (rank:u32, new_id:u32) per BPE merge rule.\n\
pub const MERGE_NUM_BUCKETS: usize = {merge_num_buckets};\n\
#[allow(dead_code)] pub const MERGE_TABLE_SIZE: usize = {merge_table_size};\n\
pub const MERGE_TABLE_MASK: usize = {merge_table_mask};\n\
",
            model_name = model_name,
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
///     must be either:
///       (a) bare Metaspace{replacement: ▁, prepend_scheme: always, split: false}
///       (b) Sequence([WhitespaceSplit, Metaspace{replacement: ▁, prepend_scheme: always}])
///     byte_fallback must be false/absent.
///
/// Returns the METASPACE_SPLIT_WHITESPACE bool (true for shape b, false for shape a).
/// This value is meaningless (false) for WordPiece builds — IS_UNIGRAM=false elides it.
///
/// Everything else is rejected with a message naming the unsupported type and
/// pointing at docs/rfc-generalization.md.
fn guard_tokenizer_family(tk: &serde_json::Value, model_name: &str) -> TokenizerFamily {
    let model_type = tk["model"]["type"].as_str().unwrap_or("unknown");
    match model_type {
        "WordPiece" => {
            // Existing WordPiece path: check normalizer matches pretokenizer.rs
            check_normalizer(&tk["normalizer"], model_name);
            TokenizerFamily::WordPiece
        }
        "Unigram" => {
            // Unigram path: validate the shapes this runtime implements
            let metaspace_split_whitespace = guard_unigram_shapes(tk, model_name);
            TokenizerFamily::Unigram { metaspace_split_whitespace }
        }
        "BPE" => {
            // BPE path: validate ByteLevel pre-tokenizer shape and reject unsupported options.
            let add_prefix_space = guard_bpe_shapes(tk, model_name);
            TokenizerFamily::Bpe { add_prefix_space }
        }
        other => {
            panic!(
                "model '{}' uses a '{}' tokenizer; ese compiles WordPiece, Unigram \
                 (SentencePiece), and BPE (ByteLevel) models — WordLevel and other types \
                 are not supported (see docs/rfc-generalization.md).",
                model_name, other
            );
        }
    }
}

/// Validate BPE-with-ByteLevel shapes this runtime can faithfully compile.
/// Panics naming the unsupported component. Returns add_prefix_space.
///
/// Accepted shape:
///   model.type == "BPE"
///   pre_tokenizer.type == "ByteLevel"   (the only supported BPE pre-tokenizer)
///   model.dropout: absent or null       (dropout sampling changes results; reject non-null)
///   model.ignore_merges: absent or false (not implemented)
///   model.byte_fallback: absent or false (ByteLevel BPE needs no byte fallback)
///
/// Anything else rejected by name (G5).
fn guard_bpe_shapes(tk: &serde_json::Value, model_name: &str) -> bool {
    // Pre-tokenizer must be ByteLevel (the only BPE pre-tokenizer we implement).
    let pre_type = tk["pre_tokenizer"]["type"].as_str().unwrap_or("unknown");
    if pre_type != "ByteLevel" {
        panic!(
            "model '{}' has pre_tokenizer type '{}'; ese's BPE runtime only supports \
             the 'ByteLevel' pre-tokenizer. Other BPE pre-tokenizers are not supported \
             (see docs/rfc-generalization.md).",
            model_name, pre_type
        );
    }

    // dropout must be absent or null (non-null dropout changes tokenization; reject it).
    let dropout = &tk["model"]["dropout"];
    if !dropout.is_null() && !dropout.is_null() {
        // Check if it's actually present and non-null
        if let Some(d) = dropout.as_f64() {
            if d > 0.0 {
                panic!(
                    "model '{}' has BPE dropout={d}; ese's BPE runtime does not support \
                     dropout sampling (results would be non-deterministic). \
                     Reject and re-distill with dropout=null.",
                    model_name
                );
            }
        }
    }

    // ignore_merges must be absent or false (not implemented).
    let ignore_merges = tk["model"]["ignore_merges"].as_bool().unwrap_or(false);
    if ignore_merges {
        panic!(
            "model '{}' has ignore_merges:true; ese's BPE runtime does not implement \
             ignore_merges. This option is rarely used in static embedding models.",
            model_name
        );
    }

    // byte_fallback must be absent or false (ByteLevel BPE doesn't need it).
    let byte_fallback = tk["model"]["byte_fallback"].as_bool().unwrap_or(false);
    if byte_fallback {
        panic!(
            "model '{}' has byte_fallback:true; ese's BPE runtime does not implement \
             byte-fallback (ByteLevel BPE models don't need it — all bytes are in the vocab).",
            model_name
        );
    }

    // Extract add_prefix_space from the ByteLevel pre-tokenizer.
    // Default is true for ByteLevel (byte_level.rs:75-82 Default impl).
    let add_prefix_space = tk["pre_tokenizer"]["add_prefix_space"]
        .as_bool()
        .unwrap_or(true);

    add_prefix_space
}

/// Validate the Unigram-specific shapes this runtime can faithfully compile.
/// Panics naming any component that does not match the accepted shape.
///
/// Returns METASPACE_SPLIT_WHITESPACE:
///   false — bare Metaspace{replacement: ▁, prepend_scheme: always, split: false}
///   true  — Sequence([WhitespaceSplit, Metaspace{replacement: ▁, prepend_scheme: always}])
///
/// Three semantics facts extracted from the vendored tokenizers-0.21.4 source
/// (recorded here per A1 — also in src/unigram.rs at the implementation site):
///
///   Fact 1 (legacy Metaspace deserialization): metaspace.rs:63-74
///     `if let Some(false) = helper.add_prefix_space` — only `false` overrides
///     prepend_scheme to Never. `add_prefix_space: true` (or absent) leaves the
///     serde default (`default_prepend_scheme_value = PrependScheme::Always`, line 45-47)
///     unchanged. `split.unwrap_or(true)` (line 74): absent `split` field defaults to true,
///     NOT false. So legacy `{add_prefix_space: true}` = `{prepend_scheme: always, split: true}`.
///
///   Fact 2 (WhitespaceSplit whitespace definition): whitespace.rs:38
///     `normalized.split(char::is_whitespace, SplitDelimiterBehavior::Removed)`
///     Splits on Rust's `char::is_whitespace` (Unicode whitespace), delimiters removed
///     (not included in any output chunk).
///
///   Fact 3 (Sequence chaining): sequence.rs:40-44
///     `for pretokenizer in &self.pretokenizers { pretokenizer.pre_tokenize(pretokenized)?; }`
///     Each stage operates on the PreTokenizedString produced by the previous stage,
///     splitting each existing chunk further. So WhitespaceSplit first splits the input
///     into words; Metaspace then runs independently on each word-chunk.
fn guard_unigram_shapes(tk: &serde_json::Value, model_name: &str) -> bool {
    // byte_fallback must be absent or false
    let byte_fallback = tk["model"]["byte_fallback"].as_bool().unwrap_or(false);
    if byte_fallback {
        panic!(
            "model '{}' has byte_fallback:true; ese's Unigram runtime does not implement \
             byte-fallback vocabs (see docs/rfc-generalization.md, layer A part 2).",
            model_name
        );
    }

    // Pre-tokenizer: accept either shape and return the mode.
    //
    // Shape (a) — bare Metaspace{replacement: ▁, prepend_scheme: always, split: false}
    //   → METASPACE_SPLIT_WHITESPACE = false (existing behavior, unchanged)
    //
    // Shape (b) — Sequence([WhitespaceSplit, Metaspace{replacement: ▁, prepend_scheme: always}])
    //   → METASPACE_SPLIT_WHITESPACE = true (new per-word Viterbi path)
    //   The Sequence shape is used by XLM-R-family teachers (distilled by model2vec).
    //   In the distilled artifact the Metaspace uses the NEW serialization
    //   (prepend_scheme/split fields explicit). The legacy form (add_prefix_space: true,
    //   no split field) is the teacher's form; both map to prepend_scheme=always, split=true.
    //
    // Anything else is rejected by name.
    let pre = &tk["pre_tokenizer"];
    let pre_type = pre["type"].as_str().unwrap_or("unknown");

    let metaspace_split_whitespace = match pre_type {
        "Metaspace" => {
            // Shape (a): bare Metaspace — must have split: false
            check_metaspace_replacement(pre, model_name);
            check_metaspace_prepend_scheme(pre, model_name);
            let split = pre["split"].as_bool().unwrap_or(true);
            if split {
                panic!(
                    "model '{}' Metaspace has split:true; ese's Unigram runtime only supports \
                     split:false for bare Metaspace (the entire normalized string is passed as \
                     one chunk to Viterbi; see docs/rfc-generalization.md, layer A part 2).",
                    model_name
                );
            }
            false
        }
        "Sequence" => {
            // Shape (b): Sequence([WhitespaceSplit, Metaspace{…}])
            // The Sequence must contain exactly [WhitespaceSplit, Metaspace].
            // Any other component is rejected by name.
            let items = pre["pretokenizers"].as_array().unwrap_or_else(|| panic!(
                "model '{}' Sequence pre_tokenizer is missing 'pretokenizers' array.",
                model_name
            ));
            if items.len() != 2 {
                panic!(
                    "model '{}' has a Sequence pre_tokenizer with {} components; ese only \
                     supports exactly [WhitespaceSplit, Metaspace] (2 components).",
                    model_name, items.len()
                );
            }
            let first_type = items[0]["type"].as_str().unwrap_or("unknown");
            if first_type != "WhitespaceSplit" {
                panic!(
                    "model '{}' Sequence pre_tokenizer[0] is '{}'; ese only supports \
                     [WhitespaceSplit, Metaspace] — component 0 must be WhitespaceSplit \
                     (see docs/rfc-generalization.md).",
                    model_name, first_type
                );
            }
            let second_type = items[1]["type"].as_str().unwrap_or("unknown");
            if second_type != "Metaspace" {
                panic!(
                    "model '{}' Sequence pre_tokenizer[1] is '{}'; ese only supports \
                     [WhitespaceSplit, Metaspace] — component 1 must be Metaspace \
                     (see docs/rfc-generalization.md).",
                    model_name, second_type
                );
            }
            check_metaspace_replacement(&items[1], model_name);
            check_metaspace_prepend_scheme(&items[1], model_name);
            // split may be true or absent (default true per Fact 1) — both are fine here
            // because WhitespaceSplit already removed spaces; the Metaspace split on ▁
            // only produces one chunk per word (▁word has one ▁ at the front).
            true
        }
        other => {
            panic!(
                "model '{}' has pre_tokenizer type '{}'; ese's Unigram runtime only supports \
                 Metaspace (bare, split:false) or Sequence([WhitespaceSplit, Metaspace]) \
                 (see docs/rfc-generalization.md).",
                model_name, other
            );
        }
    };

    // Normalizer chain must be a Sequence of exactly the supported ops
    check_unigram_normalizer(&tk["normalizer"], model_name);

    metaspace_split_whitespace
}

/// Check Metaspace replacement is ▁ (U+2581).
fn check_metaspace_replacement(pre: &serde_json::Value, model_name: &str) {
    let replacement = pre["replacement"].as_str().unwrap_or("");
    if replacement != "▁" {
        panic!(
            "model '{}' Metaspace has replacement {:?}; ese only supports ▁ (U+2581).",
            model_name, replacement
        );
    }
}

/// Check Metaspace prepend_scheme is "always" (or absent, which defaults to always).
/// Also accepts legacy add_prefix_space:true (Fact 1: maps to Always; the absence
/// of add_prefix_space, or add_prefix_space:true, both leave prepend_scheme=Always).
fn check_metaspace_prepend_scheme(pre: &serde_json::Value, model_name: &str) {
    // Legacy form: if add_prefix_space is present and false → Never (rejected below).
    // add_prefix_space:true or absent → prepend_scheme remains Always (fine).
    if let Some(false) = pre["add_prefix_space"].as_bool() {
        panic!(
            "model '{}' Metaspace has add_prefix_space:false (prepend_scheme=Never); \
             ese's Unigram runtime only supports prepend_scheme=always.",
            model_name
        );
    }
    // New-form: prepend_scheme field explicit.
    let prepend_scheme = pre["prepend_scheme"].as_str().unwrap_or("always");
    if prepend_scheme != "always" {
        panic!(
            "model '{}' Metaspace has prepend_scheme={:?}; ese's Unigram runtime only \
             supports prepend_scheme=always (see docs/rfc-generalization.md, layer A part 2).",
            model_name, prepend_scheme
        );
    }
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

/// Cross-check that post_processor.special_tokens[token].ids[0] matches
/// vocab[token] for [CLS] and [SEP]. Called only under ESE_ADD_SPECIAL_TOKENS=1
/// (meaningful only when specials are actually pooled). Panics on mismatch;
/// skips when ids field is absent (template-by-name path is fine).
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

/// Compute per-row quantization params: (row_min, row_scale).
/// Shared by the packer (weights.bin layout) and the oracle (generate_testdata),
/// so they cannot drift. quant_max is 255.0 for u8 or 65535.0 for u16.
fn row_quant_params(row: &[f32], quant_max: f32) -> (f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in row {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let scale = if (mx - mn).abs() < f32::EPSILON { 1.0_f32 } else { (mx - mn) / quant_max };
    (mn, scale)
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

/// Dequantize a single weight value using the row's own (min, scale) params.
/// Shared by generate_testdata so the oracle mirrors exactly what the packer wrote.
/// In f32 builds, row_min=0.0 and row_scale=1.0 → returns w unchanged.
#[cfg(feature = "tests")]
fn oracle_weight(w: f32, row_min: f32, row_scale: f32) -> f32 {
    if cfg!(feature = "quant-8") {
        row_min + quantize_u8(w, row_min, row_scale) as f32 * row_scale
    } else if cfg!(feature = "quant-16") {
        row_min + quantize_u16(w, row_min, row_scale) as f32 * row_scale
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

// quant_max for generate_testdata: 255.0 (u8), 65535.0 (u16), 1.0 (f32 unused).
#[cfg(feature = "tests")]
fn generate_testdata(
    out: &Path,
    weights: &[Vec<f32>],
    _dims: usize,
    tokenizer_path: &Path,
    quant_max: f32,
    add_special_tokens: bool,
    normalize_output: bool,
    skip_unk: bool,
    unk_id: usize,
) {
    let mut tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_path).expect("failed to load tokenizer");
    // Disable padding: some tokenizers (e.g. distilroberta) have padding configured
    // (max_length=128), which causes pad tokens to be included in the oracle mean.
    // For static embedding parity tests we want only the content tokens.
    tokenizer.with_padding(None);

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
        // B1 probe: "theȝ" (U+021D yogh) — 'the' (id=1996) matches as a first piece,
        // but '##ȝ' has no vocabulary entry, so HF WordPiece emits exactly one [UNK]
        // (id=100) for the whole word (oracle: [CLS,UNK,SEP]). The old ese default
        // path (SKIP_UNK=false, no scratch buffer) accumulated 'the' + [UNK] — two
        // contributions instead of one — diverging from the reference. The unified
        // scratch-buffer path (plan 009 item B) discards partial pieces on mid-word
        // failure, emitting one UNK to match HF WordPiece whole-word-UNK semantics.
        "the\u{021D}",
    ];

    let mut code = String::from("const TEST_INPUTS: &[&str] = &[\n");
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    // f32_outputs: raw f32 oracle (no quantize→dequantize); used by test_quantization_quality
    // in quant builds to measure quality vs the ideal f32 reference. In f32 builds this is
    // identical to TEST_OUTPUTS.
    let mut f32_outputs: Vec<Vec<f32>> = Vec::new();

    for &input in &inputs {
        code.push_str(&format!("    {:?},\n", input));
        let tokens = tokenizer.encode(input, add_special_tokens).unwrap();
        // When skip_unk, filter UNK ids before pooling — mirrors model2vec.StaticModel.tokenize()
        // which drops UNK tokens before mean-pooling. Default (false) includes them.
        let token_ids: Vec<u32> = if skip_unk {
            tokens.get_ids().iter().copied().filter(|&id| id as usize != unk_id).collect()
        } else {
            tokens.get_ids().to_vec()
        };

        // Quantized oracle: quantize→dequantize each weight with PER-ROW params.
        // row_quant_params is the same function used by the packer — they cannot drift.
        // Used by test_embeddings_match to check parity between packer and runtime.
        let mut vector = vec![0.0f32; _dims];
        // f32 oracle: raw weights, no quantization noise. Used by test_quantization_quality.
        let mut f32_vector = vec![0.0f32; _dims];
        for &tok_id in &token_ids {
            let row = unsafe { weights.get_unchecked(tok_id as usize) };
            // Compute per-row params using the shared function (same as packer).
            // In f32 builds: quant_max=1.0, row_quant_params returns (mn, range) but
            // oracle_weight returns w unchanged (f32 branch), so row params are harmless.
            let (row_min, row_scale) = row_quant_params(row, quant_max);
            for (v, &w) in vector.iter_mut().zip(row.iter()) {
                *v += oracle_weight(w, row_min, row_scale);
            }
            for (v, &w) in f32_vector.iter_mut().zip(row.iter()) {
                *v += w;
            }
        }
        let scale = if token_ids.is_empty() { 1.0 } else { 1.0 / token_ids.len() as f32 };
        for v in vector.iter_mut() {
            *v *= scale;
        }
        for v in f32_vector.iter_mut() {
            *v *= scale;
        }
        if normalize_output {
            normalize_vec(&mut vector);
            normalize_vec(&mut f32_vector);
        }
        outputs.push(vector);
        f32_outputs.push(f32_vector);
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
    code.push_str("];\n\n");

    // TEST_OUTPUTS_F32: ideal f32 reference (no quantization noise).
    // Only meaningful in quant builds; in f32 builds it equals TEST_OUTPUTS.
    // Used by test_quantization_quality to assert cosine vs f32 reference.
    // The #[allow(dead_code)] suppresses the warning in f32 builds where the
    // quality test is not compiled.
    code.push_str(&format!(
        "#[allow(dead_code)] const TEST_OUTPUTS_F32: &[[f32; {}]; {}] = &[\n",
        _dims,
        f32_outputs.len()
    ));
    for vec in &f32_outputs {
        code.push_str(&format!("    {:?},\n", vec.as_slice()));
    }
    code.push_str("];\n");

    fs::write(out.join("optional_testdata.rs"), code).unwrap();
}
