//! Build-script support for [`ese`](https://github.com/flowercomputers/encoders/tree/master/ese).
//!
//! Point a [`Builder`] at a sentence-transformers or model2vec static embedding
//! model from `build.rs`, call [`Builder::generate`], and then invoke
//! `ese::include_model!()` in the consuming crate. Everything the runtime needs
//! (the perfect-hash table, the quantized rows, the tokenizer parameters and
//! the pooling policy) is detected from the model's own artifacts and written
//! to `OUT_DIR`.
//!
//! ```no_run
//! // build.rs
//! let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
//! let summary = ese_build::Builder::hub("minishlab/potion-base-8M")
//!     .quant(ese_build::Quant::U8)
//!     .generate(&out)
//!     .unwrap();
//! println!("cargo:warning={summary}");
//! ```
//!
//! What is detected, and from where:
//!
//! | property | source |
//! |---|---|
//! | where the weights and tokenizer live | `modules.json`, path of the `StaticEmbedding` module |
//! | L2-normalize the output | a `Normalize` module in `modules.json`, else `config.json`'s `normalize` |
//! | drop `[UNK]` before pooling | `config.json` says `model_type: model2vec` |
//! | embedding tensor | `embedding.weight`, `embeddings`, `0`, or the only 2-D tensor |
//! | vocabulary quantization | optional `mapping` and `weights` tensors |
//! | normalizer flags, subword prefix, max word length, unk token | `tokenizer.json` |
//!
//! Only the BERT family (`BertNormalizer` + `BertPreTokenizer` + `WordPiece`)
//! is supported at runtime today. Other tokenizers fail the build with a
//! message naming what was found.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

mod emit;
mod fetch;
mod phf;
mod safetensors;
#[cfg(feature = "test-data")]
mod testdata;
mod tokenizer;

pub use tokenizer::TokenizerFamily;

/// Storage type for each embedding element in the generated tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// Unquantized `f32`.
    F32,
    /// Linear 16-bit quantization with each column's own min and scale;
    /// rows far smaller than the median are boosted by a power of two first.
    U16,
    /// Linear 8-bit quantization, same scheme as [`Quant::U16`].
    U8,
}

impl Quant {
    pub(crate) fn elem_name(self) -> &'static str {
        match self {
            Quant::F32 => "f32",
            Quant::U16 => "u16",
            Quant::U8 => "u8",
        }
    }

    pub(crate) fn elem_size(self) -> usize {
        match self {
            Quant::F32 => 4,
            Quant::U16 => 2,
            Quant::U8 => 1,
        }
    }

    pub(crate) fn levels(self) -> Option<f32> {
        match self {
            Quant::F32 => None,
            Quant::U16 => Some(65535.0),
            Quant::U8 => Some(255.0),
        }
    }
}

impl fmt::Display for Quant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.elem_name())
    }
}

/// Everything that can go wrong while generating a model.
#[derive(Debug)]
pub enum Error {
    /// Reading model files or writing the generated ones failed.
    Io(std::io::Error),
    /// The hub answered a download with an error status.
    Http {
        url: String,
        status: i32,
    },
    /// A download failed before any HTTP status arrived.
    Network {
        url: String,
        source: minreq::Error,
    },
    /// A model JSON file (`config.json`, `modules.json`, `tokenizer.json`)
    /// did not parse.
    Json {
        file: PathBuf,
        source: serde_json::Error,
    },
    /// A required file was neither on disk nor on the hub.
    MissingFile(String),
    /// `modules.json` describes something other than a static embedding model.
    NotStatic(String),
    /// `tokenizer.json` describes a pipeline the runtime cannot reproduce.
    UnsupportedTokenizer(String),
    /// The safetensors file holds no usable embedding matrix or uses an
    /// unsupported dtype.
    UnsupportedTensor(String),
    /// Anything else: inconsistent model files or builder settings.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Http { url, status } => write!(f, "HTTP {status} fetching {url}"),
            Error::Network { url, source } => write!(f, "network error fetching {url}: {source}"),
            Error::Json { file, source } => {
                write!(f, "invalid JSON in {}: {source}", file.display())
            }
            Error::MissingFile(s) => write!(f, "missing model file: {s}"),
            Error::NotStatic(s) => write!(f, "not a static embedding model: {s}"),
            Error::UnsupportedTokenizer(s) => write!(f, "unsupported tokenizer: {s}"),
            Error::UnsupportedTensor(s) => write!(f, "unsupported safetensors layout: {s}"),
            Error::Invalid(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Source {
    Hub { repo: String, revision: String },
    Local(PathBuf),
}

/// Configures where the model comes from and how it is compiled.
#[derive(Clone, Debug)]
pub struct Builder {
    source: Source,
    subfolder: Option<String>,
    dims: Option<usize>,
    quant: Quant,
    test_data: bool,
    cache_dir: Option<PathBuf>,
    truncation: Option<Vec<usize>>,
    license: Option<String>,
    multilingual: bool,
    parallel_min: usize,
}

impl Builder {
    /// A model on the Hugging Face hub, e.g. `minishlab/potion-base-8M`.
    pub fn hub(repo: impl Into<String>) -> Self {
        Self::new(Source::Hub {
            repo: repo.into(),
            revision: "main".into(),
        })
    }

    /// A model directory on disk, laid out like a hub repo.
    pub fn local(dir: impl Into<PathBuf>) -> Self {
        Self::new(Source::Local(dir.into()))
    }

    fn new(source: Source) -> Self {
        Self {
            source,
            subfolder: None,
            dims: None,
            quant: Quant::F32,
            test_data: false,
            cache_dir: None,
            truncation: None,
            license: None,
            multilingual: false,
            parallel_min: 128,
        }
    }

    /// Git revision (branch, tag or commit sha) for hub models. Defaults to
    /// `main`; pin a sha for reproducible builds.
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        if let Source::Hub { revision: r, .. } = &mut self.source {
            *r = revision.into();
        }
        self
    }

    /// Override the subfolder holding `tokenizer.json` and
    /// `model.safetensors` instead of reading it from `modules.json`.
    pub fn subfolder(mut self, subfolder: impl Into<String>) -> Self {
        self.subfolder = Some(subfolder.into());
        self
    }

    /// Truncate embeddings to this many dimensions. `None` keeps the model's
    /// native width. A request wider than the model fails; a request below
    /// it is checked against [`Builder::truncation_points`] when given and
    /// otherwise only warned about, since it is only sound for
    /// Matryoshka-trained models.
    pub fn dims(mut self, dims: Option<usize>) -> Self {
        self.dims = dims;
        self
    }

    /// Widths the model was trained at. Cannot be detected from the model's
    /// files, so it is the caller's knowledge. `Some(&[])` or a list holding
    /// only the native width means truncation is unsupported; `None` (the
    /// default) means unknown. Emitted as `TRUNCATION_POINTS`.
    pub fn truncation_points(mut self, points: Option<&[usize]>) -> Self {
        self.truncation = points.map(|p| p.to_vec());
        self
    }

    /// The model's license as an SPDX identifier, emitted as `LICENSE`. Not
    /// in the model's files in any reliable form, so the caller's knowledge.
    pub fn license(mut self, spdx: impl Into<String>) -> Self {
        self.license = Some(spdx.into());
        self
    }

    /// Whether the model targets more than one language, emitted as
    /// `MULTILINGUAL`. Defaults to `false`.
    pub fn multilingual(mut self, multilingual: bool) -> Self {
        self.multilingual = multilingual;
        self
    }

    /// Batch size from which the consumer's `encode` goes parallel with the
    /// `rayon` feature, emitted as `PARALLEL_MIN`. Defaults to 128.
    pub fn parallel_min(mut self, n: usize) -> Self {
        self.parallel_min = n.max(1);
        self
    }

    /// Storage type of the generated rows. Defaults to [`Quant::F32`].
    pub fn quant(mut self, quant: Quant) -> Self {
        self.quant = quant;
        self
    }

    /// Also write `optional_testdata.rs` holding reference embeddings computed
    /// with the `tokenizers` crate. Requires the `test-data` feature.
    pub fn test_data(mut self, enabled: bool) -> Self {
        self.test_data = enabled;
        self
    }

    /// Where downloaded files are kept. Defaults to `target/ese-cache` when
    /// `OUT_DIR` is inside a `target` directory (so a workspace shares one
    /// download), otherwise an `ese-cache` directory inside `OUT_DIR`.
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Fetch, detect, and write the generated files into `out_dir`.
    pub fn generate(&self, out_dir: &Path) -> Result<Summary, Error> {
        #[cfg(not(feature = "test-data"))]
        if self.test_data {
            return Err(Error::Invalid(
                "test data was requested but ese-build was compiled without the `test-data` feature"
                    .into(),
            ));
        }

        let files = fetch::Files::resolve(&self.source, self.cache_dir.as_deref(), out_dir)?;
        let layout = Layout::detect(&files, self.subfolder.as_deref())?;
        let tokenizer_path = files.require(&layout.join("tokenizer.json"))?;
        let model_path = files.require(&layout.join("model.safetensors"))?;

        let spec = tokenizer::parse(&tokenizer_path)?;

        let st = safetensors::SafeTensors::open(&model_path)?;
        let emb = st.find_embedding()?;
        let native_dims = emb.shape[1];
        let dims = self.resolve_dims(native_dims)?;
        let truncation_points: Vec<usize> = match &self.truncation {
            Some(p) => p.iter().copied().filter(|&d| d <= native_dims).collect(),
            None => Vec::new(),
        };
        let base = st.read_rows(emb, dims)?;
        let mapping = match st.get("mapping") {
            Some(t) => Some(st.read_indices(t)?),
            None => None,
        };
        let weights = match st.get("weights") {
            Some(t) => Some(st.read_f32_1d(t)?),
            None => None,
        };
        if let (Some(w), Quant::U8) = (&weights, self.quant) {
            // Frequent tokens carry weights far below 1 in SIF-weighted exports;
            // after materialization their rows are tiny next to the per-column
            // range set by rare tokens, and 8 bits cannot resolve them.
            let (lo, hi) = w
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| {
                    (l.min(x), h.max(x))
                });
            println!(
                "cargo:warning=ese-build: {} applies per-token weights spanning {lo:.3}..{hi:.3}; \
                 8-bit per-column quantization loses precision on its frequent tokens. Prefer 16-bit or f32 storage.",
                match &self.source {
                    Source::Hub { repo, .. } => repo.as_str(),
                    Source::Local(_) => "this model",
                }
            );
        }
        let rows = materialize(base, mapping, weights, &spec)?;

        let keys: Vec<&[u8]> = spec.vocab.iter().map(|(t, _)| t.as_bytes()).collect();
        let phf = phf::build(&keys)?;
        let entry_row: Vec<usize> = spec.vocab.iter().map(|(_, id)| *id).collect();

        let (model_id, revision) = match &self.source {
            Source::Hub { repo, revision } => (repo.clone(), revision.clone()),
            Source::Local(dir) => (dir.display().to_string(), String::new()),
        };

        let quant_info = emit::write(&emit::Emit {
            out: out_dir,
            dims,
            quant: self.quant,
            phf: &phf,
            rows: &rows,
            entry_row: &entry_row,
            spec: &spec,
            layout: &layout,
            model_id: &model_id,
            revision: &revision,
            native_dims,
            truncation_points: &truncation_points,
            license: self.license.as_deref(),
            multilingual: self.multilingual,
            parallel_min: self.parallel_min,
        })?;

        #[cfg(feature = "test-data")]
        if self.test_data {
            testdata::write(
                out_dir,
                &tokenizer_path,
                &rows,
                dims,
                spec.unk_id,
                layout.drop_unk,
                layout.normalize,
                self.quant,
                &quant_info,
            )?;
        }
        #[cfg(not(feature = "test-data"))]
        let _ = quant_info;

        let embedded_bytes = ["weights.bin", "verify.bin", "seeds.bin"]
            .iter()
            .map(|f| fs::metadata(out_dir.join(f)).map(|m| m.len()).unwrap_or(0))
            .sum();

        Ok(Summary {
            embedded_bytes,
            model: model_id,
            revision,
            framework: layout.framework,
            tokenizer: spec.family,
            vocab_size: spec.vocab.len(),
            native_dims,
            dims,
            quant: self.quant,
            table_size: phf.table_size,
            drop_unk: layout.drop_unk,
            normalize: layout.normalize,
            truncation_points,
            cache_dir: files.root().to_path_buf(),
        })
    }

    fn resolve_dims(&self, native: usize) -> Result<usize, Error> {
        let Some(d) = self.dims else {
            return Ok(native);
        };
        if d == native {
            return Ok(native);
        }
        if d > native {
            return Err(Error::Invalid(format!(
                "requested {d} dimensions but the model has {native}"
            )));
        }
        match &self.truncation {
            Some(points) => {
                if points.contains(&d) {
                    Ok(d)
                } else {
                    let supported: Vec<usize> =
                        points.iter().copied().filter(|&p| p < native).collect();
                    Err(Error::Invalid(if supported.is_empty() {
                        format!(
                            "requested {d} dimensions but the model is not Matryoshka-trained and only supports its native {native}"
                        )
                    } else {
                        format!(
                            "requested {d} dimensions but the model was only trained at {supported:?} below its native {native}"
                        )
                    }))
                }
            }
            None => {
                println!(
                    "cargo:warning=ese-build: truncating {native}-dimensional embeddings to {d}; this is only sound for Matryoshka-trained models"
                );
                Ok(d)
            }
        }
    }
}

/// What was detected and generated. `Display` gives a one-line summary that
/// is handy to echo as a `cargo:warning`.
#[derive(Clone, Debug)]
pub struct Summary {
    /// Bytes of generated tables the consumer embeds with `include_bytes!`:
    /// the rows, the verifiers (which also carry the row exponents) and the
    /// seeds. This is what the model adds to the final binary, give or take
    /// the few KB of constants.
    pub embedded_bytes: u64,
    /// Hub repo id, or the directory for a local model.
    pub model: String,
    /// Hub revision the files came from; empty for a local model.
    pub revision: String,
    /// `sentence-transformers` or `model2vec`, as detected.
    pub framework: &'static str,
    pub tokenizer: TokenizerFamily,
    pub vocab_size: usize,
    /// Width of the stored embeddings before truncation.
    pub native_dims: usize,
    /// Width of the generated embeddings.
    pub dims: usize,
    pub quant: Quant,
    /// Slots in the perfect-hash table, a power of two at least `vocab_size`.
    pub table_size: usize,
    /// Unknown tokens are dropped before pooling (model2vec semantics).
    pub drop_unk: bool,
    /// Pooled vectors are L2-normalized.
    pub normalize: bool,
    /// Widths the model was trained at, as given to the builder.
    pub truncation_points: Vec<usize>,
    /// Where the model's files were read from.
    pub cache_dir: PathBuf,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ese: model={} rev={} framework={} tokenizer={} vocab={} dims={}/{} quant={} table={} drop_unk={} normalize={} embedded={}",
            self.model,
            if self.revision.is_empty() {
                "local"
            } else {
                &self.revision
            },
            self.framework,
            self.tokenizer,
            self.vocab_size,
            self.dims,
            self.native_dims,
            self.quant,
            self.table_size,
            self.drop_unk,
            self.normalize,
            human_bytes(self.embedded_bytes),
        )
    }
}

fn human_bytes(n: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    if n >= MIB as u64 {
        format!("{:.1} MiB", n as f64 / MIB)
    } else {
        format!("{:.0} KiB", n as f64 / 1024.0)
    }
}

/// Where the files live and how the model expects to be pooled.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    pub subfolder: String,
    pub normalize: bool,
    pub drop_unk: bool,
    pub framework: &'static str,
}

impl Layout {
    fn join(&self, name: &str) -> String {
        if self.subfolder.is_empty() {
            name.to_string()
        } else {
            format!("{}/{name}", self.subfolder)
        }
    }

    fn detect(files: &fetch::Files, override_sub: Option<&str>) -> Result<Layout, Error> {
        let mut subfolder: Option<String> = None;
        let mut normalize = false;

        if let Some(path) = files.get("modules.json")? {
            let modules = read_json(&path)?;
            let modules = modules
                .as_array()
                .ok_or_else(|| Error::Invalid("modules.json is not an array".into()))?;
            for m in modules {
                let ty = m["type"].as_str().unwrap_or("");
                let path = m["path"].as_str().unwrap_or(".");
                if ty.ends_with("StaticEmbedding") {
                    if subfolder.is_none() {
                        subfolder = Some(norm_path(path));
                    }
                } else if ty.ends_with("Normalize") {
                    normalize = true;
                } else {
                    return Err(Error::NotStatic(format!(
                        "modules.json contains a `{ty}` module; only StaticEmbedding and Normalize are supported"
                    )));
                }
            }
            if subfolder.is_none() {
                return Err(Error::NotStatic(
                    "modules.json lists no StaticEmbedding module".into(),
                ));
            }
        }

        let mut drop_unk = false;
        let mut framework = "sentence-transformers";
        if let Some(path) = files.get("config.json")? {
            let cfg = read_json(&path)?;
            // sentence-transformers static models never ship a root config.json
            // (they have config_sentence_transformers.json), while every
            // model2vec export does. Older exports say `model_type`; newer ones
            // (potion-code-16M) only carry model2vec-specific keys.
            let is_model2vec = cfg["model_type"].as_str() == Some("model2vec")
                || [
                    "vocabulary_quantization",
                    "embedding_dtype",
                    "apply_pca",
                    "apply_zipf",
                    "sif_coefficient",
                    "hidden_dim",
                    "tokenizer_name",
                ]
                .iter()
                .any(|k| !cfg[*k].is_null())
                || subfolder.as_deref() == Some("");
            if is_model2vec {
                // model2vec removes the unk token before pooling; sentence-transformers keeps it.
                drop_unk = true;
                framework = "model2vec";
            }
            if subfolder.is_none()
                && let Some(b) = cfg["normalize"].as_bool()
            {
                normalize = b;
            }
        }

        let subfolder = override_sub
            .map(norm_path)
            .or(subfolder)
            .unwrap_or_default();

        Ok(Layout {
            subfolder,
            normalize,
            drop_unk,
            framework,
        })
    }
}

fn norm_path(p: &str) -> String {
    p.trim_start_matches("./")
        .trim_matches('/')
        .trim_end_matches('.')
        .trim_matches('/')
        .to_string()
}

pub(crate) fn read_json(path: &Path) -> Result<serde_json::Value, Error> {
    let data = fs::read(path)?;
    serde_json::from_slice(&data).map_err(|source| Error::Json {
        file: path.to_path_buf(),
        source,
    })
}

/// Turn the stored tensor into one f32 row per vocabulary id, resolving the
/// optional model2vec vocabulary-quantization indirection (`mapping`) and
/// per-token scaling (`weights`).
fn materialize(
    base: Vec<Vec<f32>>,
    mapping: Option<Vec<usize>>,
    weights: Option<Vec<f32>>,
    spec: &tokenizer::Spec,
) -> Result<Vec<Vec<f32>>, Error> {
    let n = spec.vocab.len();
    for (tok, id) in &spec.vocab {
        if *id >= n {
            return Err(Error::Invalid(format!(
                "token {tok:?} has id {id} but the vocabulary only has {n} entries; ids must be contiguous"
            )));
        }
    }
    if let Some(w) = &weights
        && w.len() != n
    {
        return Err(Error::UnsupportedTensor(format!(
            "`weights` has {} entries for a vocabulary of {n}",
            w.len()
        )));
    }
    match mapping {
        None => {
            if base.len() != n {
                return Err(Error::UnsupportedTensor(format!(
                    "embedding tensor has {} rows for a vocabulary of {n} and no `mapping` tensor",
                    base.len()
                )));
            }
            match weights {
                None => Ok(base),
                Some(w) => Ok(base
                    .into_iter()
                    .zip(w)
                    .map(|(row, s)| row.into_iter().map(|v| v * s).collect())
                    .collect()),
            }
        }
        Some(map) => {
            if map.len() != n {
                return Err(Error::UnsupportedTensor(format!(
                    "`mapping` has {} entries for a vocabulary of {n}",
                    map.len()
                )));
            }
            let mut rows = Vec::with_capacity(n);
            for (t, &idx) in map.iter().enumerate() {
                let row = base.get(idx).ok_or_else(|| {
                    Error::UnsupportedTensor(format!(
                        "`mapping` sends token {t} to row {idx}, past the {} stored rows",
                        base.len()
                    ))
                })?;
                let s = weights.as_ref().map_or(1.0, |w| w[t]);
                rows.push(row.iter().map(|v| v * s).collect());
            }
            Ok(rows)
        }
    }
}
