# What `ese` claims to do — and how to falsify it

This document states what `ese` can do as a set of **concrete, testable claims**.
Each claim comes with the command to verify it and the condition that would
**falsify** it. If you can make a falsify condition true on a supported input,
the claim is wrong and we want to know.

These claims describe the **core compiler** (this branch). Claims that require
the companion distillation tooling are marked **[tooling]** and live in the
stacked tooling PR (`distill/`, `benches/py/`).

**Prerequisites:** a stable Rust toolchain. The first build downloads the
default model's weights + tokenizer from Hugging Face into `target/ese-cache`
(network required once). Numbers below were measured on an Apple M4 Pro; run
the commands on your hardware to reproduce.

---

## C1 — Compiles WordPiece, Unigram, and BPE static models with no source edits

**Verify:**
```sh
# WordPiece (BERT-family)
ESE_MODEL=stephantulkens/NIFE-mxbai-embed-large-v1@970a0151f3faeddbab5c68be38d52bbc5ea719ab \
  ESE_NORMALIZE_OUTPUT=1 cargo build --features dim-1024
# Unigram (SentencePiece)
ESE_MODEL=minishlab/potion-multilingual-128M@73908c3438cf03b6a01bcb9611d62b23d0726f08 \
  ESE_NORMALIZE_OUTPUT=1 cargo build --features dim-256
```
(BPE/ByteLevel models compile the same way via `ESE_MODEL` or `ESE_MODEL_DIR`;
a distilled RoBERTa is the reference case — see **[tooling]** C10.)

**Expect:** each builds successfully, selecting the model purely from
environment configuration.

**Falsify:** a model whose tokenizer is WordPiece, Unigram, or ByteLevel-BPE
*in the shapes the README documents as supported* fails to compile, or requires
editing source.

---

## C2 — The runtime reproduces the compiled artifact exactly (implementation fidelity)

The `tests` feature regenerates reference outputs at build time using the
upstream `tokenizers` crate as an independent oracle, then asserts the runtime
matches.

**Verify:**
```sh
cargo test --features tests
cargo test --features tests,quant-8
cargo test --features tests,quant-16
```

**Expect:** all pass. Cosine similarity to the oracle exceeds `0.999999` for f32
builds and `0.99999` for quantized builds (the exact thresholds are in
`src/test/mod.rs`).

**Falsify:** any of these fails — i.e. the runtime's tokenization or pooling
diverges from the reference for the compiled model.

---

## C3 — The default model matches the real `sentence-transformers` reference

`ese`'s default model (`static-retrieval-mrl-en-v1`) does **not** pool special
tokens, matching the reference exactly (this was a corrected divergence; see the
README Changelog).

**Verify (independent, Python):** encode a corpus with both
`SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1").encode(corpus)`
and with `ese`, then compare per-sentence cosine **and** the upper-triangular
similarity matrices `S` over the corpus (the retrieval-relevant metric).

**Expect:** per-sentence cosine `≈ 1.0000`; `mean|S_ref − S_ese| ≈ 0.00000`;
correlation of the two similarity matrices `≈ 1.0`.

**Falsify:** a measurable similarity-matrix difference, or per-sentence cosine
below `0.999`, against the live reference.

---

## C4 — Quantized builds stay faithful to the f32 model

8-bit and 16-bit builds use per-row affine quantization; a dedicated test
measures the quantized output against the unquantized ideal.

**Verify:**
```sh
cargo test --features tests,quant-8     # test_quantization_quality
cargo test --features tests,quant-16
```

**Expect:** pass. Worst-case cosine of the quantized output vs the f32 ideal is
`≥ 0.999` (quant-8) and `≥ 0.99999` (quant-16); measured quant-8 worst case is
`0.999965`.

**Falsify:** quantized quality below those thresholds on the test inputs.

---

## C5 — Builds are reproducible and tamper-evident

Model downloads are pinned to an immutable revision and verified by SHA-256
before any bytes are compiled into the binary.

**Verify:** build once (populates `target/ese-cache`), then corrupt a cached
file and rebuild:
```sh
cargo build
printf 'x' >> target/ese-cache/*/tokenizer.json   # or model.safetensors
touch build.rs && cargo build
```

**Expect:** the second build **fails** with a SHA-256 mismatch error naming the
file. The pinned revision + digests live in `build.rs`.

**Falsify:** a corrupted or substituted model artifact compiles silently, or two
clean builds of the same configuration embed different weights.

---

## C6 — Unsupported tokenizers are rejected at build time, never silently mishandled

**Verify:** point `ESE_MODEL` / `ESE_MODEL_DIR` at a model whose tokenizer is an
unsupported family or shape (e.g. a cased WordPiece model with
`lowercase: false`, a BPE model with a non-ByteLevel pre-tokenizer, or a
WordLevel/GloVe tokenizer) and run `cargo build`.

**Expect:** the build **panics with a clear message naming the unsupported
component** and pointing at the roadmap — it does not produce a binary.

**Falsify:** any tokenizer shape the README does not list as supported compiles
and emits embeddings (i.e. silent divergence rather than a loud error).

---

## C7 — Runs as WebAssembly with output identical to native, in a small bundle

**Verify:**
```sh
wasm-pack build api-wasm --target nodejs --out-dir pkg-node
node api-wasm/crosscheck.mjs
```

**Expect:** per-sentence cosine `≥ 0.9999` between the wasm build and the native
binary built with the same features (observed: `1.0000`). The default bundle
(`dim-128`, `quant-8`) is `~4.6 MB` raw / `~4.2 MB` gzipped, retaining `93.3%`
of the full-dimension NanoBEIR score.

**Falsify:** wasm output diverges from native, or the default bundle is
materially larger than claimed.

---

## C8 — Inference is a pure function over baked data — no model runtime, no ML framework

**Verify:** inspect the runtime dependency graph (`cargo tree -p ese`) and the
encode path. `encode_single` is text normalization + a perfect-hash lookup +
arithmetic over weights baked into `.rodata` via `include_bytes!`.

**Expect:** no model-load step at runtime and no transformer/ONNX/`tokenizers`
runtime dependency on the encode path. Runtime deps are Unicode tables (plus
`spm_precompiled` only on the Unigram path); the embedding table is compiled
into the binary.

**Falsify:** a runtime model-load, file read of weights at encode time, or an
ML-inference framework on the `encode`/`encode_single` path.

---

## C9 — Throughput far exceeds CPU transformer baselines (measure on your hardware)

**Verify:**
```sh
cargo bench --bench gooaq --features rayon   # single, batch, length-sensitive throughput
cargo bench                                  # head-to-head vs model2vec-rs (bench_encode.rs)
```

**Expect:** single-input latency in the low microseconds; batched throughput
orders of magnitude above a CPU transformer, and competitive with or faster than
`model2vec-rs`. (The deployment runtime is the same fast path for every compiled
model — independent of tokenizer family.)

**Falsify:** on equivalent hardware, throughput does not exceed the cited
baselines.

---

## C10 — **[tooling]** Any Hugging Face transformer can be distilled to an `ese` model, and contrastive post-training lifts retrieval quality several-fold

Requires the companion tooling (`distill/`, `benches/py/`).

**Verify:**
```sh
cd distill
uv run python distill.py sentence-transformers/all-MiniLM-L6-v2 --out out/minilm        # raw distill
uv run python tokenlearn_train.py --teacher sentence-transformers/all-MiniLM-L6-v2 \
  --corpus gooaq --n-texts 100000 --out ../target/distilled/tokenlearn-minilm           # + Tokenlearn
# then measure both with the NanoBEIR harness in benches/py
```

**Expect:** raw model2vec distillation retains `~15–19%` of the teacher's
NanoBEIR retrieval; Tokenlearn post-training lifts a distilled MiniLM from
`0.104 → ~0.40` NanoBEIR (≈`70%` retention) in ~95 s of CPU. The lift is **not**
a training-corpus artifact: it holds ~`4×` on out-of-domain SciFact/NFCorpus
(`0.083 → 0.344`). The distilled artifact compiles through the core compiler and
matches `model2vec`'s own output at cosine `1.0`.

**Falsify:** no measurable NanoBEIR lift from Tokenlearn, the lift vanishing on
out-of-domain datasets, or the compiled `ese` output diverging from `model2vec`.

---

## Scope and non-claims (what we do *not* claim)

These are honest limits, stated so the claims above aren't read as more than
they are:

- **Not transformer-equivalent on order/negation-sensitive queries.** `ese` is a
  bag-of-tokens mean-pool; it discards word order by construction
  ("A recommends B" ≈ "B recommends A"). The default model retains roughly
  three-quarters of a small transformer's NanoBEIR — strong for a static model,
  not equal to a transformer.
- **Raw distillation is low-quality without post-training** (~17% retention,
  per C10). The quality claim is about the *Tokenlearn* path, not raw distill.
- **Tokenizer coverage is bounded:** WordPiece, Unigram (SentencePiece), and
  ByteLevel-BPE in documented shapes. Cased models, `byte_fallback` vocabs,
  non-ByteLevel BPE pre-tokenizers, WordLevel, and exotic normalizer chains are
  **rejected at build time** (C6), not supported.
- **Quantization is lossy** (C4 bounds the loss; it does not claim losslessness).
- **No claim of bit-identical output across architectures** beyond the
  wasm-vs-native fidelity in C7; floating-point results are within the stated
  cosine tolerances, not bit-exact.

If a claim here is wrong, or a verification command doesn't reproduce, that's a
bug — please open an issue with the command and the output.
