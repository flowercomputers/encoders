use std::{env, path::PathBuf};

use ese_build::{Builder, Quant};

/// A model selectable with a `model-*` cargo feature. Revisions are pinned
/// to the commit that was current when the catalog was written. `dims` lists
/// the widths the model was trained at, native first; that cannot be
/// detected from the model's files, so it is recorded here and enforced.
struct ModelSpec {
    feature: &'static str,
    repo: &'static str,
    revision: &'static str,
    dims: &'static [usize],
}

const MRL: &[usize] = &[1024, 512, 256, 128, 64, 32];

const CATALOG: &[ModelSpec] = &[
    ModelSpec {
        feature: "model-static-retrieval-mrl-en-v1",
        repo: "sentence-transformers/static-retrieval-mrl-en-v1",
        revision: "f60985c706f192d45d218078e49e5a8b6f15283a",
        dims: MRL,
    },
    ModelSpec {
        feature: "model-static-similarity-mrl-multilingual-v1",
        repo: "sentence-transformers/static-similarity-mrl-multilingual-v1",
        revision: "b68f4122911bcffcd6e1f695f2d99cd6788972d8",
        dims: MRL,
    },
    ModelSpec {
        feature: "model-bekko-static-embedding-v1",
        repo: "hotchpotch/bekko-static-embedding-v1",
        revision: "c18167b8f244c1bc6de21ce9b710bac8a29c4615",
        dims: MRL,
    },
    ModelSpec {
        feature: "model-potion-base-2m",
        repo: "minishlab/potion-base-2M",
        revision: "389b9f64be5aa4ae7a6bc6fe95ef20ce485ae5da",
        dims: &[64],
    },
    ModelSpec {
        feature: "model-potion-base-4m",
        repo: "minishlab/potion-base-4M",
        revision: "9b3cff412d30be9ae8603fe10224c224f3401869",
        dims: &[128],
    },
    ModelSpec {
        feature: "model-potion-base-8m",
        repo: "minishlab/potion-base-8M",
        revision: "bf8b056651a2c21b8d2565580b8569da283cab23",
        dims: &[256],
    },
    ModelSpec {
        feature: "model-potion-base-32m",
        repo: "minishlab/potion-base-32M",
        revision: "1e5a03f8eeb2c98b928fbbd846f22f816360919f",
        dims: &[512],
    },
    ModelSpec {
        feature: "model-potion-retrieval-32m",
        repo: "minishlab/potion-retrieval-32M",
        revision: "6fc8051fab2a1e0ee76689cf08c853792ac285e7",
        dims: &[512],
    },
    ModelSpec {
        feature: "model-potion-science-8m",
        repo: "minishlab/potion-science-8M",
        revision: "b186d1c45c8205d552a00f7a0979896ca2dc9e87",
        dims: &[256],
    },
    ModelSpec {
        feature: "model-potion-science-32m",
        repo: "minishlab/potion-science-32M",
        revision: "7366079845507de14a4330007cdfa01bb92bca52",
        dims: &[256],
    },
    ModelSpec {
        feature: "model-potion-code-16m",
        repo: "minishlab/potion-code-16M",
        revision: "1b0ff71095656b23306542bbad34a09109673720",
        dims: &[256],
    },
    ModelSpec {
        feature: "model-nife-mxbai-embed-large-v1",
        repo: "stephantulkens/NIFE-mxbai-embed-large-v1",
        revision: "970a0151f3faeddbab5c68be38d52bbc5ea719ab",
        dims: &[1024],
    },
    ModelSpec {
        feature: "model-nife-gte-modernbert-base",
        repo: "stephantulkens/NIFE-gte-modernbert-base",
        revision: "9b85ba93d5a7729db11dfea90d20a5f392a88747",
        dims: &[768],
    },
    ModelSpec {
        feature: "model-m2v-base-output",
        repo: "minishlab/M2V_base_output",
        revision: "0b86653131492ba2a858911d5ccf84ae774b40f3",
        dims: &[256],
    },
];

const DIM_FEATURES: &[(&str, usize)] = &[
    ("dim-1024", 1024),
    ("dim-768", 768),
    ("dim-512", 512),
    ("dim-256", 256),
    ("dim-128", 128),
    ("dim-64", 64),
    ("dim-32", 32),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    let model = selected_model();
    let dims = selected_dims(model);
    let quant = selected_quant();

    let summary = Builder::hub(model.repo)
        .revision(model.revision)
        .dims(dims)
        .truncation_points(Some(model.dims))
        .quant(quant)
        .test_data(cfg!(feature = "tests"))
        .generate(&out)
        .unwrap_or_else(|e| panic!("ese-build failed for {}: {e}", model.repo));

    println!("cargo:warning={summary}");
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
        [m] => m,
        [] => panic!(
            "encoders: no model selected. Enable exactly one of: {}. \
             If you set `default-features = false`, add a model feature, a dim-* \
             feature if the model supports truncation, and a quant-* feature.",
            feature_list()
        ),
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

/// The truncation a dim-* feature asks for, validated against the model.
fn selected_dims(model: &ModelSpec) -> Option<usize> {
    let on: Vec<usize> = DIM_FEATURES
        .iter()
        .filter(|(f, _)| feature_on(f))
        .map(|(_, d)| *d)
        .collect();
    let dims = match on.as_slice() {
        [] => return None,
        [d] => *d,
        many => panic!("encoders: several dim-* features enabled ({many:?}); enable exactly one"),
    };
    if !model.dims.contains(&dims) {
        let native = model.dims[0];
        if dims > native {
            panic!(
                "encoders: dim-{dims} requested but {} has {native} dimensions",
                model.repo
            );
        }
        if model.dims.len() == 1 {
            panic!(
                "encoders: dim-{dims} requested but {} is not Matryoshka-trained and only \
                 supports its native {native} dimensions; drop the dim-* feature",
                model.repo
            );
        }
        panic!(
            "encoders: dim-{dims} requested but {} was trained at {:?}",
            model.repo, model.dims
        );
    }
    Some(dims)
}

fn selected_quant() -> Quant {
    match (feature_on("quant-8"), feature_on("quant-16")) {
        (true, true) => panic!("encoders: quant-8 and quant-16 are mutually exclusive"),
        (true, false) => Quant::U8,
        (false, true) => Quant::U16,
        (false, false) => Quant::F32,
    }
}
