use std::{
    env, fs,
    path::{Path, PathBuf},
};

use ese_build::{Builder, Quant};

/// A model selectable with a cargo feature of the same name. Revisions are pinned
/// to the commit that was current when the catalog was written. `dims` lists
/// the widths the model was trained at, native first; that cannot be
/// detected from the model's files, so it is recorded here and enforced.
struct ModelSpec {
    feature: &'static str,
    repo: &'static str,
    revision: &'static str,
    dims: &'static [usize],
    /// SPDX identifier of the model's own license.
    license: &'static str,
    multilingual: bool,
}

/// Batch size from which `encode` goes parallel with the `rayon` feature.
const PARALLEL_MIN: usize = 128;

const MRL: &[usize] = &[1024, 512, 256, 128, 64, 32];

const CATALOG: &[ModelSpec] = &[
    ModelSpec {
        feature: "bekko-static-v1-slim-100k",
        repo: "flower-computer/bekko-static-v1-slim-100k",
        revision: "635eb094a82ce43f0eb882c12b31406b3d93ec26",
        dims: MRL,
        license: "MIT",
        multilingual: true,
    },
    ModelSpec {
        feature: "bekko-static-v1-slim-75k",
        repo: "flower-computer/bekko-static-v1-slim-75k",
        revision: "48d57794b1d6b94f4535817341e6f8bf2f5b31e2",
        dims: MRL,
        license: "MIT",
        multilingual: true,
    },
    ModelSpec {
        feature: "bekko-static-v1-slim-50k",
        repo: "flower-computer/bekko-static-v1-slim-50k",
        revision: "63abf2b319526cc06cacf85bb86e2bfcd1b5f987",
        dims: MRL,
        license: "MIT",
        multilingual: true,
    },
    ModelSpec {
        feature: "bekko-static-v1-slim-32k",
        repo: "flower-computer/bekko-static-v1-slim-32k",
        revision: "c4ce4db02b1e50d858745b40cb80da627e26ee4c",
        dims: MRL,
        license: "MIT",
        multilingual: true,
    },
    ModelSpec {
        feature: "bekko-static-embedding-v1",
        repo: "hotchpotch/bekko-static-embedding-v1",
        revision: "c18167b8f244c1bc6de21ce9b710bac8a29c4615",
        dims: MRL,
        license: "MIT",
        multilingual: true,
    },
    ModelSpec {
        feature: "static-retrieval-mrl-en-v1",
        repo: "sentence-transformers/static-retrieval-mrl-en-v1",
        revision: "f60985c706f192d45d218078e49e5a8b6f15283a",
        dims: MRL,
        license: "Apache-2.0",
        multilingual: false,
    },
    ModelSpec {
        feature: "static-similarity-mrl-multilingual-v1",
        repo: "sentence-transformers/static-similarity-mrl-multilingual-v1",
        revision: "b68f4122911bcffcd6e1f695f2d99cd6788972d8",
        dims: MRL,
        license: "Apache-2.0",
        multilingual: true,
    },
    ModelSpec {
        feature: "potion-base-2m",
        repo: "minishlab/potion-base-2M",
        revision: "389b9f64be5aa4ae7a6bc6fe95ef20ce485ae5da",
        dims: &[64],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "potion-base-4m",
        repo: "minishlab/potion-base-4M",
        revision: "9b3cff412d30be9ae8603fe10224c224f3401869",
        dims: &[128],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "potion-base-8m",
        repo: "minishlab/potion-base-8M",
        revision: "bf8b056651a2c21b8d2565580b8569da283cab23",
        dims: &[256],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "potion-base-32m",
        repo: "minishlab/potion-base-32M",
        revision: "1e5a03f8eeb2c98b928fbbd846f22f816360919f",
        dims: &[512],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "potion-retrieval-32m",
        repo: "minishlab/potion-retrieval-32M",
        revision: "6fc8051fab2a1e0ee76689cf08c853792ac285e7",
        dims: &[512],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "potion-science-8m",
        repo: "minishlab/potion-science-8M",
        revision: "b186d1c45c8205d552a00f7a0979896ca2dc9e87",
        dims: &[256],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "potion-science-32m",
        repo: "minishlab/potion-science-32M",
        revision: "7366079845507de14a4330007cdfa01bb92bca52",
        dims: &[256],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "potion-code-16m",
        repo: "minishlab/potion-code-16M",
        revision: "1b0ff71095656b23306542bbad34a09109673720",
        dims: &[256],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "nife-mxbai-embed-large-v1",
        repo: "stephantulkens/NIFE-mxbai-embed-large-v1",
        revision: "970a0151f3faeddbab5c68be38d52bbc5ea719ab",
        dims: &[1024],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "nife-gte-modernbert-base",
        repo: "stephantulkens/NIFE-gte-modernbert-base",
        revision: "9b85ba93d5a7729db11dfea90d20a5f392a88747",
        dims: &[768],
        license: "MIT",
        multilingual: false,
    },
    ModelSpec {
        feature: "m2v-base-output",
        repo: "minishlab/M2V_base_output",
        revision: "0b86653131492ba2a858911d5ccf84ae774b40f3",
        dims: &[256],
        license: "MIT",
        multilingual: false,
    },
];

/// Used when no model feature is enabled.
const DEFAULT_MODEL: &str = "bekko-static-v1-slim-50k";

const DIM_FEATURES: &[(&str, usize)] = &[
    ("mrl-1024", 1024),
    ("mrl-768", 768),
    ("mrl-512", 512),
    ("mrl-256", 256),
    ("mrl-128", 128),
    ("mrl-64", 64),
    ("mrl-32", 32),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    let model = selected_model();
    let dims = selected_dims(model);
    let quant = selected_quant();

    // docs.rs builds without network access, so document against a tiny
    // placeholder model of the same width instead of downloading the real one.
    let docs_rs = env::var_os("DOCS_RS").is_some();
    let builder = if docs_rs {
        Builder::local(placeholder_model(&out, dims.unwrap_or(model.dims[0])))
    } else {
        Builder::hub(model.repo).revision(model.revision)
    };
    let summary = builder
        .dims(dims)
        .truncation_points(Some(model.dims))
        .license(model.license)
        .multilingual(model.multilingual)
        .parallel_min(PARALLEL_MIN)
        .quant(quant)
        .test_data(cfg!(feature = "tests"))
        .generate(&out)
        .unwrap_or_else(|e| panic!("ese-build failed for {}: {e}", model.repo));
    if docs_rs {
        // Document the real model's identity rather than the placeholder's.
        let path = out.join("model_constants.rs");
        let constants = fs::read_to_string(&path).expect("read model_constants.rs");
        let constants = constants
            .lines()
            .map(|l| {
                if l.starts_with("pub const MODEL_ID:") {
                    format!("pub const MODEL_ID: &str = {:?};", model.repo)
                } else if l.starts_with("pub const MODEL_REVISION:") {
                    format!("pub const MODEL_REVISION: &str = {:?};", model.revision)
                } else {
                    l.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, constants).expect("write model_constants.rs");
    }

    println!("cargo:warning={summary}");
}

/// Write a model2vec-layout model with a handful of tokens and `dims`-wide
/// zero rows, enough for ese-build to generate tables that compile.
fn placeholder_model(out: &Path, dims: usize) -> PathBuf {
    const VOCAB: &[&str] = &["[PAD]", "[UNK]", "[CLS]", "[SEP]"];
    let dir = out.join("docsrs-model");
    fs::create_dir_all(&dir).expect("create placeholder model dir");

    let vocab: Vec<String> = VOCAB
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{t:?}: {i}"))
        .collect();
    let tokenizer = format!(
        r###"{{"added_tokens": [{{"id": 0, "content": "[PAD]", "special": true}}, {{"id": 1, "content": "[UNK]", "special": true}}],
"normalizer": {{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}},
"pre_tokenizer": {{"type": "BertPreTokenizer"}},
"model": {{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {{{}}}}}}}"###,
        vocab.join(", ")
    );
    fs::write(dir.join("tokenizer.json"), tokenizer).expect("write placeholder tokenizer");

    let body = vec![0u8; VOCAB.len() * dims * 4];
    let header = format!(
        r#"{{"embeddings":{{"dtype":"F32","shape":[{},{dims}],"data_offsets":[0,{}]}}}}"#,
        VOCAB.len(),
        body.len()
    );
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header.as_bytes());
    st.extend_from_slice(&body);
    fs::write(dir.join("model.safetensors"), st).expect("write placeholder weights");
    dir
}

/// Cargo features are exposed to build scripts as `CARGO_FEATURE_<NAME>`
/// with dashes turned into underscores and upper-cased.
fn feature_on(name: &str) -> bool {
    env::var_os(format!(
        "CARGO_FEATURE_{}",
        name.replace('-', "_").to_ascii_uppercase()
    ))
    .is_some()
}

fn selected_model() -> &'static ModelSpec {
    let on: Vec<&ModelSpec> = CATALOG.iter().filter(|m| feature_on(m.feature)).collect();
    match on.as_slice() {
        [] => CATALOG
            .iter()
            .find(|m| m.feature == DEFAULT_MODEL)
            .expect("DEFAULT_MODEL is in the catalog"),
        [m] => m,
        many => panic!(
            "encoders: several models selected ({}). Cargo features are additive: \
             depend on encoders with `default-features = false` and enable exactly one of {}.",
            many.iter()
                .map(|m| m.feature)
                .collect::<Vec<_>>()
                .join(", "),
            feature_list()
        ),
    }
}

fn feature_list() -> String {
    CATALOG
        .iter()
        .map(|m| m.feature)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The truncation an mrl-* feature asks for, validated against the model.
fn selected_dims(model: &ModelSpec) -> Option<usize> {
    let on: Vec<usize> = DIM_FEATURES
        .iter()
        .filter(|(f, _)| feature_on(f))
        .map(|(_, d)| *d)
        .collect();
    let dims = match on.as_slice() {
        [] => return None,
        [d] => *d,
        many => panic!("encoders: several mrl-* features enabled ({many:?}); enable exactly one"),
    };
    if !model.dims.contains(&dims) {
        let native = model.dims[0];
        if dims > native {
            panic!(
                "encoders: mrl-{dims} requested but {} has {native} dimensions",
                model.repo
            );
        }
        if model.dims.len() == 1 {
            panic!(
                "encoders: mrl-{dims} requested but {} is not Matryoshka-trained and only \
                 supports its native {native} dimensions; drop the mrl-* feature",
                model.repo
            );
        }
        panic!(
            "encoders: mrl-{dims} requested but {} was trained at {:?}",
            model.repo, model.dims
        );
    }
    Some(dims)
}

/// Storage: `int16` is the default feature and the fallback when nothing is
/// set; `float32` or `int8` passed explicitly override it (so leaving the
/// default features on and adding one of them works); `int8` and `float32`
/// together is a contradiction.
fn selected_quant() -> Quant {
    match (feature_on("int8"), feature_on("float32")) {
        (true, true) => panic!("encoders: int8 and float32 are mutually exclusive"),
        (true, false) => Quant::U8,
        (false, true) => Quant::F32,
        (false, false) => Quant::U16,
    }
}
