# ESE: Extremely Static Embeddings

`ese` turns text into dense vectors using a static embedding table compiled into
your Rust binary. It is a small, purpose-built runtime for the static embedding
model deployment pipeline described in [Much faster static embedding](https://www.flowercomputer.com/news/fast-static-embedding/).

The crate exposes two public encoding functions:

- `ese::encode_single(text)` encodes one string.
- `ese::encode(texts)` encodes many strings.

Both return `f32` vectors with a compile-time dimension count exposed as
`ese::DIMENSIONS`.

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
`target/ese-cache`. Later builds reuse the cached files.

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

Run the workspace tests:

```console
cargo test --workspace
```

Run the optional embedding reference check:

```console
cargo test -p ese --features tests
```

The optional test uses the tokenizer crate at build time to generate reference
outputs for a small set of inputs, then checks that `ese` produces matching
embedding directions.

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

## Using a different static embedding model

This repo does not yet expose a stable custom-model interface.

Today, using a different static embedding model means editing the two URLs near
the top of `build.rs`:

```rust
static MODEL_URL: &'static str = ".../model.safetensors";
static TOKENIZER_URL: &'static str = ".../tokenizer.json";
```

The replacement model needs to match the assumptions currently baked into the
build script and runtime:

- The model weights are stored in safetensors format.
- The tensor is shaped like `[vocab_size, dimensions]`.
- Tokenizer vocab IDs correspond to rows in the weight table.
- The tokenizer JSON has a `model.vocab` object.
- WordPiece continuation tokens use the `##` prefix.
- Special token IDs are treated as `100 = UNK`, `101 = CLS`, and `102 = SEP`.

If those assumptions are not true, you will probably need to modify `build.rs`,
`src/pretokenizer.rs`, or `src/wordpiece.rs`.

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

1. Normalizes input text.
2. Splits text into pre-tokens.
3. Applies a WordPiece-style lookup.
4. Accumulates token vectors.
5. Returns one dense `f32` vector per input string.

The goal is to make static embeddings feel like a normal Rust function call,
without loading an ML framework in the hot path.

## Limitations and current status

- The crate is currently a preview.
- The default model URLs point at Hugging Face `main`, not a pinned model commit.
- Custom model support is possible, but not yet a stable API.
- The Python wrapper is preliminary and has less documentation than the Rust
  crate.
- `ese` does not normalize vectors for you. If your search stack expects
  normalized vectors, normalize them before indexing or comparison.
- `ese` does not include persistence, indexing, filtering, or query serving.

If you want a complete search system, combine `ese` with your own document
store and nearest-neighbor search layer.
