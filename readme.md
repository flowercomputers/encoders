# ese: compiled static embeddings

`ese` is an experimental local embedding engine and benchmark harness.

Instead of loading a full transformer model at runtime, it downloads a fixed
static embedding model at build time, compiles the vocabulary and weights into
the artifact, and then produces embeddings by direct token lookup plus mean
pooling.

This repository is useful if you want to:

- benchmark this compiled approach against other embedding models
- run the Rust encoder locally without hosting a model server
- call the same encoder from Rust, C, or Python

This repository is not a polished product yet. The runtime model is fixed, the
Python and C wrappers are intentionally thin, and most of the project surface is
aimed at local benchmarking rather than broad distribution.

## What is in the repo?

- `src/`: the Rust embedding library
- `api-c/`: a minimal C wrapper around `encode_single`
- `api-py/`: a minimal Python extension built with `maturin`
- `benches/`: Rust Criterion benches plus Python comparison scripts
- `bench.sh`: the supported local benchmark entrypoint

## What happens on first run?

The first Rust build downloads the model weights and tokenizer from Hugging
Face. They are cached under `target/ese-cache`.

Benchmark runs may also download:

- GooAQ parquet shards into `target/gooaq`
- Hugging Face datasets and model weights into your normal local caches for the
  Python comparison path

If you hit Hugging Face rate limits during the Python path, set `HF_TOKEN` in
your shell before running the benchmark commands.

## Prerequisites

### Rust-only workflow

Required:

- `cargo`
- `rustc`

Validated locally with:

- `cargo 1.93.1`
- `rustc 1.93.1`

### Optional Python comparison workflow

Required:

- `python3.13`
- `uv`
- `maturin`

The Python path is optional. You do not need it for the Rust verification or
Rust Criterion benchmarks.

## Quickstart

### 1. Verify the core repo works

```sh
./bench.sh verify
```

This runs:

- `cargo test -p ese --features tests`
- `cargo build -p ese-c`
- build-only smoke checks for the Rust benchmark targets

Use this first if you just want to confirm the repo is healthy on your machine.

### 2. Run the Rust benchmarks

```sh
./bench.sh rust
```

This runs the Criterion benchmarks in two modes:

- sequential
- Rayon-enabled

Results are collected into:

- `bench_out/rust/criterion_sequential`
- `bench_out/rust/criterion_rayon`

An archive is also written to `bench_results.tar.gz`.

### 3. Run the lightweight Python smoke benchmark

```sh
./bench.sh python
```

By default this benchmarks only the local `ours` model on GooAQ. That is the
intended smoke test for the Python comparison path.

Results are collected into:

- `bench_out/python/gooaq`

### 4. Build the browser WebAssembly benchmark demo

Required once on a machine:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Then build the browser demo:

```sh
./bench.sh web
```

This writes a static browser demo to:

- `bench_out/web`

Serve it locally with any static file server, for example:

```sh
cd bench_out/web
python3 -m http.server 8000
```

The demo page gives you:

- a browser-side WASM encode benchmark
- a rendered waveform view of one embedding vector
- a cosine similarity heatmap across several sample texts

### 5. Run the lightweight NanoBEIR smoke evaluation

```sh
./bench.sh nanobeir
```

By default this evaluates only the local `ours` model.

Results are collected into:

- `bench_out/python/nanobeir`

### 6. Run the default full local harness

```sh
./bench.sh all
```

This runs:

- `verify`
- `rust`
- `python` for the smoke model
- `nanobeir` for the smoke model

It does not run the full external model sweep by default.

## Full comparison runs

The full Python model comparison sweep is intentionally opt-in.

Run every model listed in `benches/py/models.txt`:

```sh
./bench.sh python --all
./bench.sh nanobeir --all
```

You can also run a single explicit model:

```sh
./bench.sh python all-MiniLM-L6-v2
./bench.sh nanobeir static-retrieval-mrl-en-v1
```

## Supported `bench.sh` interface

```sh
./bench.sh verify
./bench.sh rust
./bench.sh web
./bench.sh python [model]
./bench.sh python --all
./bench.sh nanobeir [model]
./bench.sh nanobeir --all
./bench.sh all
```

Important defaults:

- `PYTHON_BIN=python3.13`
- `RESULTS_DIR=bench_out`
- `MODELS_FILE=benches/py/models.txt`
- `SMOKE_MODEL=ours`
- local Rust runs default to `RUSTFLAGS="-Ctarget-cpu=native"`

Useful overrides:

```sh
BENCH_TIME_SECONDS=3 ./bench.sh python
RESULTS_DIR=tmp-results ./bench.sh rust
MODELS_FILE=benches/py/models.txt ./bench.sh python --all
PYTHON_BIN=python3.13 ./bench.sh nanobeir
```

## Direct library usage

The Rust API is intentionally small:

```rust
let single = ese::encode_single("hello world");
let batch = ese::encode(["potato", "root vegetable"]);
assert_eq!(single.len(), ese::DIMENSIONS);
assert_eq!(batch[0].len(), ese::DIMENSIONS);
```

The C wrapper exports:

- `DIMENSIONS`
- `encode(const char*, float*)`

The Python extension exports:

- `encode(...)`
- `dimensions()`
- `DIMENSIONS`

The Python extension is meant to be built with `maturin`:

```sh
cd api-py
RUSTFLAGS="-Ctarget-cpu=native" maturin build --release --interpreter python3.13
```

## Known caveats

- The model is fixed in `build.rs`; this repo is not a general model runner.
- First build and first benchmark runs require network access.
- The Python path is heavier and slower to set up than the Rust path.
- `cargo build -p ese-py` is not the intended way to package the Python module;
  use `maturin build`.
- The benchmark harness is designed for local macOS/Linux use first.

## Useful manual commands

If you want the exact low-level checks without the shell harness:

```sh
cargo test -p ese --features tests
cargo build -p ese-c
cargo bench --bench bench_encode --no-run
cargo bench --bench gooaq --no-run
```
