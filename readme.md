# ESE: Extremely Static Embeddings

`ese` turns text into dense vectors using a static embedding table compiled into
your Rust binary. It is a small, purpose-built runtime for the static embedding
model deployment pipeline described in [Much faster static embedding](https://www.flowercomputer.com/news/fast-static-embedding/).

The crate exposes two public encoding functions and two compile-time constants:

- `ese::encode_single(text)` encodes one string.
- `ese::encode(texts)` encodes many strings.
- `ese::DIMENSIONS` — the number of dimensions per vector (set by `dim-*` features).
- `ese::MODEL_NAME` — the model identifier compiled into this build.

Both functions return `f32` vectors with `ese::DIMENSIONS` elements.

## What this is

- A fast Rust crate for turning text into dense embedding vectors.
- A compiled runtime for the static embedding model
  [`sentence-transformers/static-retrieval-mrl-en-v1`](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1).
- A way to run embedding on CPU without loading a transformer runtime or model
  library at application startup.
- A useful building block for semantic search, document retrieval, ranking,
  clustering, memory systems, or local search experiments.

## What this is not

- It is not a vector database.
- It is not a search server.
- It is not a model training pipeline.
- It is not a transformer inference runtime.
- It is not yet a stable framework for plugging in arbitrary embedding models.

`ese` gives you vectors. You decide how to store them, compare them, index them,
and serve them.

## Quickstart: Rust

Add `ese` to your project:

```toml
[dependencies]
ese = { git = "https://github.com/flowercomputers/ese", features = ["rayon", "dim-512", "quant-8"] }
```

If you have this repository checked out locally, you can depend on it by path:

```toml
[dependencies]
ese = { path = "../ese", features = ["rayon", "dim-512", "quant-8"] }
```

The first build downloads the model weights and tokenizer from Hugging Face into
`target/ese-cache`. Later builds reuse the cached files. Note: `cargo clean`
wipes `target/`, including the cache. Set `ESE_CACHE_DIR` to a path outside
`target/` (e.g. `~/.cache/ese`) to persist the cache across cleans.

Encode one string:

```rust
fn main() {
    let embedding = ese::encode_single("hello world");

    println!("dimensions: {}", ese::DIMENSIONS);
    println!("first value: {}", embedding[0]);
}
```

Encode a batch:

```rust
fn main() {
    let embeddings = ese::encode([
        "potato",
        "root vegetable",
        "semantic search",
    ]);

    println!("encoded {} strings", embeddings.len());
    println!("each vector has {} dimensions", embeddings[0].len());
}
```

### Tiny semantic search example

This example embeds one query, embeds a few documents, compares them with cosine
similarity, and prints the closest matches.

```rust
fn cosine<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let a_norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let b_norm = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm * b_norm)
    }
}

fn main() {
    let query = ese::encode_single("best way to grow lavender indoors");
    let documents = [
        "Lavender needs bright light and dry soil when grown inside.",
        "Rust ownership helps prevent memory safety bugs.",
        "Tomato plants need consistent watering and warm weather.",
    ];

    let document_embeddings = ese::encode(documents);

    let mut scored: Vec<_> = documents
        .iter()
        .zip(document_embeddings.iter())
        .map(|(document, embedding)| (cosine(&query, embedding), *document))
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (score, document) in scored {
        println!("{score:.3}  {document}");
    }
}
```

For production search, you will usually precompute document embeddings, store
them somewhere, and use a vector index or nearest-neighbor library. `ese` only
handles embedding.

## Embedding as a local primitive

Because the model is static and compiled into the crate output, embedding can
be cheap enough to call like ordinary application logic. That lets a host
application construct semantic queries at runtime instead of only querying
pre-built vector indexes.

For example, a caller can rank a small in-memory working set directly (reusing
the `cosine` helper from the quickstart above):

```rust
let query = ese::encode_single("urgent technical bug or production issue");
let mut emails = vec![
    ("deploy failed", "production deploy is returning 500s"),
    ("lunch", "where should we eat today?"),
    ("invoice", "can you review the vendor invoice?"),
];

emails.sort_by(|(_, a), (_, b)| {
    let a_score = cosine(&ese::encode_single(a), &query);
    let b_score = cosine(&ese::encode_single(b), &query);
    b_score.total_cmp(&a_score)
});
```

For larger corpora, precompute and index embeddings as usual. The point is that
`ese` also makes the small, dynamic cases practical: rerank a page of search
results, score candidate memories, filter recent messages, or expose embedding
as a primitive inside an agent runtime without adding network latency, API
cost, or online availability assumptions.

## Feature flags

The default build is `dim-512`, unquantized, and single-threaded. All available
features are listed in [`Cargo.toml`](./Cargo.toml).

| Feature | What it does | Default |
| --- | --- | --- |
| `dim-1024` | Use 1024-dimensional output vectors. | No |
| `dim-768` | Use 768-dimensional output vectors. | No |
| `dim-512` | Use 512-dimensional output vectors. | Yes |
| `dim-256` | Use 256-dimensional output vectors. | No |
| `dim-128` | Use 128-dimensional output vectors. | No |
| `dim-64` | Use 64-dimensional output vectors. | No |
| `dim-32` | Use 32-dimensional output vectors. | No |
| `quant-16` | Store compiled weights as 16-bit integers. Output is still `f32`. | No |
| `quant-8` | Store compiled weights as 8-bit integers. Output is still `f32`. | No |
| `rayon` | Encode batches of 16 or more inputs in parallel. | No |

Pick one `dim-*` feature for normal use. Lower dimensions use less memory and
are cheaper to compare downstream. Higher dimensions preserve more of the
upstream model output.

Quantization changes how the compiled weight table is stored in the binary. The
public API still returns `f32` vectors.

## WebAssembly

`api-wasm/` contains WebAssembly bindings built with `wasm-pack`. The default
build is `dim-128 + quant-8` — **4.6 MB raw, 4.2 MB gzipped** — a genuine
browser payload (12× smaller than the unquantized baseline).

### Bundle size

The model is baked into the `.wasm` binary at build time. Size depends on the
`dim-*` and `quant-8` features selected:

| Variant | Raw | Gzipped | NanoBEIR nDCG@10 | Retention vs dim-512 |
|---|---:|---:|---:|---:|
| `dim-512` (f32, old default) | 60.2 MB | 55.9 MB | 0.4957 | 100% |
| `dim-512,quant-8` | 15.8 MB | 14.7 MB | ~0.4957 | ~100% |
| `dim-256,quant-8` | 8.3 MB | 7.7 MB | 0.4820 | 97.2% |
| **`dim-128,quant-8` (default)** | **4.6 MB** | **4.2 MB** | **0.4624** | **93.3%** |
| `dim-64,quant-8` | 2.7 MB | 2.4 MB | 0.4177 | 84.3% |

quant-8's additional quality cost vs f32 at the same dim is ≤ 0.1% (plan 009
`test_quantization_quality`); the size/quality tradeoff is dominated by dim
choice.

**Knee:** `dim-128,quant-8` is the recommended default — 93% retention at 13×
the size reduction. Use `dim-256,quant-8` (97% retention, 8.3 MB) if the extra
quality matters more than the extra 3.7 MB.

### Build a different variant

```console
wasm-pack build api-wasm --target web -- --no-default-features --features "dim-256,quant-8"
```

Replace `dim-256,quant-8` with any combination from the table above.

```console
wasm-pack build api-wasm --target nodejs --out-dir pkg-node
node api-wasm/crosscheck.mjs   # fidelity check: wasm vs native at cosine ≥ 0.9999
```

### Quick demo

```js
import init, { encode_single, dimensions } from './api-wasm/pkg/ese_wasm.js';
await init();

const dim = dimensions();          // 128 (default build)
const vec = encode_single("hello world");  // Float32Array(128)
```

## Quickstart: Python

The Python API is preliminary. It wraps the Rust crate with PyO3 and returns
NumPy arrays. See [`./api-py/readme.md`](./api-py/readme.md) for full Python
API documentation.

Build and install the extension into your current Python environment:

```console
cargo install maturin
cd api-py
RUSTFLAGS="-Ctarget-cpu=native" maturin develop --release
```

Use it from Python:

```python
import ese

vectors = ese.encode(["hello world", "semantic search"])

print(vectors.shape)
print(ese.DIMENSIONS)
```

The current Python wrapper builds `ese` with `rayon`, `dim-512`, and `quant-8`.

## Running tests

Run the tests (requires network access to crates.io and huggingface.co on first
run to download the default model):

```console
cargo test
```

Run the optional embedding reference check:

```console
cargo test --features tests
```

The optional test uses the tokenizer crate at build time to generate reference
outputs for a small set of inputs, then checks that `ese` produces matching
embedding directions. To verify the workspace builds without running tests,
use `cargo check --workspace`.

## Performance note

For maximum throughput on your own machine, build with:

```console
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
```

Omit this flag when building distributed binaries or wheels intended to run
on other machines — the resulting artifact may otherwise crash with SIGILL on
CPUs that do not support the instruction extensions selected at compile time.

## Running benchmarks

Run the Rust GooAQ benchmark:

```console
cargo bench --bench gooaq --features rayon
```

This downloads a subset of the GooAQ dataset into `target/gooaq` and measures
single-input, batch, and input-length-sensitive throughput.

To run the Python benchmark harness, first build a wheel:

```console
cd api-py
RUSTFLAGS="-Ctarget-cpu=native" maturin build --release
```

Then run the benchmark scripts:

```console
cd ../benches/py
uv run --with ../../target/wheels/ese_py-*.whl bench.py ours
uv run --with ../../target/wheels/ese_py-*.whl nanobeir.py ours
```

The Python harness also knows how to load models from `sentence-transformers`,
so you can compare `ours` against entries in `benches/py/models.txt`.

## Using a different model

`ese` selects the compiled model at build time via environment variables — no
source edits required. The build script derives tokenizer facts (continuation
prefix, output normalization) automatically from the model artifacts. Special
tokens (`[CLS]`/`[SEP]`) are **not** pooled by default — verified to produce a cosine of 1.0000 against the real
`SentenceTransformer.encode()` reference (specials-on was 0.9994). Set
`ESE_ADD_SPECIAL_TOKENS=1` to opt in if your model explicitly requires it.

### Environment variables

| Variable | Description |
|---|---|
| `ESE_MODEL` | `<hf-repo>@<revision>` to compile a model from Hugging Face. The `@<revision>` is **mandatory** — bare repo IDs are rejected (reproducibility is non-negotiable). |
| `ESE_MODEL_DIR` | Path to a local directory containing `model.safetensors` and `tokenizer.json`. No download, no checksum. |
| `ESE_MODEL_SHA256` | Expected SHA-256 of `model.safetensors` (strongly recommended when using `ESE_MODEL`). |
| `ESE_TOKENIZER_SHA256` | Expected SHA-256 of `tokenizer.json` (strongly recommended when using `ESE_MODEL`). |
| `ESE_MODEL_FILE` | Override the file subpath within the repo (default: `model.safetensors`). The built-in default model uses `0_StaticEmbedding/model.safetensors`. |
| `ESE_TOKENIZER_FILE` | Override the tokenizer subpath within the repo (default: `tokenizer.json`). |
| `ESE_NORMALIZE_OUTPUT` | Set to `1` to L2-normalize the output embedding vector. Required for mxbai-style models and recommended for model2vec-distilled models. |
| `ESE_SKIP_UNK` | Set to `1` to exclude unknown (`[UNK]`) tokens from the pooled mean. Matches `model2vec.StaticModel` behavior; recommended for model2vec-distilled models. |
| `ESE_ADD_SPECIAL_TOKENS` | Set to `1` to pool `[CLS]`/`[SEP]` tokens in the mean (opt-in; off by default — static-embedding references do not use them). |
| `ESE_CACHE_DIR` | Directory for cached model downloads (default: `target/ese-cache`). Set to a path outside `target/` to persist the cache across `cargo clean`. |

When `ESE_MODEL_SHA256` or `ESE_TOKENIZER_SHA256` are omitted, the build
emits a `cargo:warning` — checksums are optional for local experimentation
but required for reproducible production builds.

### Supported tokenizer families

`ese` compiles WordPiece, Unigram (SentencePiece), and BPE (ByteLevel)
vocabulary models. The build script validates the tokenizer type and
normalizer/pre-tokenizer shapes at compile time and emits a clear error for
unsupported families (WordLevel, etc.).

**WordPiece** models must use a `BertNormalizer` or compatible normalizer chain.
The default model (`sentence-transformers/static-retrieval-mrl-en-v1`) is
WordPiece.

**Unigram** models must use:
- A `Precompiled` charsmap normalizer (optionally wrapped in a `Sequence`
  with `Replace`/`Strip` steps for space collapsing and trimming).
- A pre-tokenizer that is either a bare `Metaspace` (`▁`, `always`, `split: false`)
  or a `Sequence([WhitespaceSplit, Metaspace])` (`▁`, `always`) — the latter is the
  shape used by XLM-R-family teachers (e.g. `paraphrase-multilingual-MiniLM-L12-v2`,
  `multilingual-e5-base`) after model2vec distillation.
- `byte_fallback: false`.

**BPE** models must use a `ByteLevel` pre-tokenizer (the GPT-2/RoBERTa byte
alphabet), `byte_fallback: false`, `ignore_merges: false`, and null dropout.
This covers RoBERTa-family, GPT-2-derived, and ModernBERT/GTE models, both
directly and as model2vec distillation targets (e.g.
`sentence-transformers/all-distilroberta-v1`).

WordLevel and other tokenizer families are not yet supported — see
`docs/rfc-generalization.md`.

### Worked example: NIFE-mxbai-embed-large-v1 (WordPiece)

```sh
ESE_MODEL=stephantulkens/NIFE-mxbai-embed-large-v1@970a0151f3faeddbab5c68be38d52bbc5ea719ab \
ESE_MODEL_SHA256=<sha256-of-model.safetensors> \
ESE_TOKENIZER_SHA256=<sha256-of-tokenizer.json> \
ESE_NORMALIZE_OUTPUT=1 \
cargo build --features dim-1024
```

Replace `<sha256-of-model.safetensors>` and `<sha256-of-tokenizer.json>` with
the SHA-256 digests computed from the downloaded files (`shasum -a 256 <file>`).

### Worked example: potion-multilingual-128M (Unigram)

```sh
ESE_MODEL=minishlab/potion-multilingual-128M@73908c3438cf03b6a01bcb9611d62b23d0726f08 \
ESE_MODEL_SHA256=14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397 \
ESE_TOKENIZER_SHA256=19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1 \
ESE_NORMALIZE_OUTPUT=1 \
cargo build --features dim-256
```

### Using a local artifact directory

```sh
ESE_MODEL_DIR=/path/to/my-model cargo build --features dim-256
```

The directory must contain `model.safetensors` and `tokenizer.json`. This is
the designated seam for distillation tooling: a Python step can emit a
transformed artifact directory that `ese` consumes unchanged, with no Python
in the pure-Rust consumer path. (The distillation tool ships separately.)


## How it works

At build time, `build.rs`:

1. Downloads `model.safetensors` and `tokenizer.json` if they are not already in
   `target/ese-cache`.
2. Parses the embedding table from safetensors.
3. Truncates the table to the selected `dim-*` feature.
4. Optionally quantizes the stored weights.
5. Builds a static lookup table from tokenizer string to embedding row.
6. Emits generated Rust constants and binary weight tables into Cargo's
   `OUT_DIR`.

At runtime, `ese`:

1. Normalizes input text (charsmap + space collapsing for Unigram models;
   lowercasing + accent stripping for WordPiece models).
2. Splits text into pre-tokens.
3. For WordPiece models: applies the WordPiece subword algorithm.
   For Unigram models: runs a Viterbi segmentation over the vocabulary.
4. Accumulates token vectors.
5. Returns one dense `f32` vector per input string.

The goal is to make static embeddings feel like a normal Rust function call,
without loading an ML framework in the hot path.

## Limitations and current status

- The crate is currently a preview.
- The default model is pinned to a specific commit SHA for reproducibility.
  Custom models require a pinned `@<revision>` for the same reason.
- WordPiece, Unigram (SentencePiece), and BPE (ByteLevel) vocabulary models
  are supported. WordLevel is planned (see `docs/rfc-generalization.md`).
- The Python wrapper is preliminary and has less documentation than the Rust
  crate.
- `ese` does not normalize vectors by default. If your search stack expects
  normalized vectors, set `ESE_NORMALIZE_OUTPUT=1` at build time or normalize
  them yourself before indexing or comparison.
- `ese` does not include persistence, indexing, filtering, or query serving.

If you want a complete search system, combine `ese` with your own document
store and nearest-neighbor search layer.

## Changelog

### Unreleased

This release generalizes `ese` from a single hardcoded model into a build-time
compiler for any supported static embedding model, and corrects the default
model's pooling to match the upstream reference. Entries marked **[output
change]** alter the embedding vectors a build produces.

#### Changed — action may be required

- **[output change] The default model's embeddings now match the upstream
  reference exactly.** Two corrections landed: special tokens (`[CLS]`/`[SEP]`)
  are no longer pooled — the sentence-transformers and model2vec static
  references never insert them, and pooling content tokens only reproduces
  `SentenceTransformer.encode()` at cosine 1.0000 (up from 0.9994) — and the pooled mean now uses the exact reference
  token count (earlier builds scaled by a slightly different denominator,
  shifting magnitudes). **Re-embed any stored default-model vectors** for exact
  cosine or dot-product comparisons; semantic ranking is effectively unchanged
  (similarity-matrix correlation 0.9997 to the previous behavior). Set
  `ESE_ADD_SPECIAL_TOKENS=1` to restore special-token pooling.
- **Default builds no longer force `-Ctarget-cpu=native`.** Binaries are
  portable by default (no SIGILL on CPUs older than the build machine). Opt in
  with `RUSTFLAGS="-Ctarget-cpu=native"` for maximum throughput — see
  [Performance note](#performance-note).

#### Added

- **Compile any WordPiece, Unigram (SentencePiece), or BPE (ByteLevel) model
  at build time** via `ESE_MODEL=<repo>@<revision>` or a local `ESE_MODEL_DIR`,
  with no source edits — including XLM-R / multilingual and RoBERTa-family
  models. See
  [Using a different model](#using-a-different-model).
- **Per-row weight quantization** for `quant-8` / `quant-16` — a large fidelity
  improvement on the shipped quantized wheel (worst-case quality vs the
  unquantized model went from 0.94 to 0.9999).
- **Reproducible, tamper-evident builds.** Model downloads are pinned to an
  immutable revision and verified by SHA-256 before being compiled into the
  binary.
- **New build-time environment variables:** `ESE_NORMALIZE_OUTPUT`,
  `ESE_SKIP_UNK`, `ESE_ADD_SPECIAL_TOKENS`, `ESE_CACHE_DIR`, `ESE_MODEL_SHA256`,
  `ESE_TOKENIZER_SHA256` (see [Using a different model](#using-a-different-model)).
- **Continuous integration** across an operating-system × feature matrix, with a
  quantization-aware parity gate.

#### Fixed

- The embedding parity test verifies the runtime against the compiled (possibly
  quantized) artifact, with a separate gate measuring quantized-vs-f32 fidelity.
  Quantized builds — including the shipped Python wheel configuration — were
  previously untestable.
- WordPiece whole-word fallback matches the reference: a word with an unmatched
  continuation piece becomes a single `[UNK]` (partial sub-word pieces are no
  longer kept).
