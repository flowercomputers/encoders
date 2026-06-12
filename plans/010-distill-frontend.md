# Plan 010: Distillation front-end — turn ANY HF embedding model into ese artifacts (layer C)

> **Executor instructions**: This plan is executed by TWO sequential
> sub-agents with a reviewer gate between them. Stage 1 builds the tool;
> Stage 2 proves it on a model basket. Each stage's GATES are the
> definition of done — every gate must be green with evidence (command +
> output) in the report. SKIP updating `plans/README.md`.
>
> **Drift check (run first)**: HEAD must be `c574b5b` or a descendant that
> has not touched `build.rs`/`src/` since.

## Status

- **Priority**: P1
- **Effort**: L (two stages)
- **Risk**: MED (new Python surface; no changes to the Rust runtime)
- **Depends on**: plans 006 (ESE_MODEL_DIR seam) and 008 (Unigram runtime);
  design in docs/rfc-generalization.md §layer-C
- **Category**: direction (layer C)
- **Planned at**: commit `c574b5b`, 2026-06-12

## Why this matters

ese now compiles any WordPiece or Unigram **static** model. The remaining
gap to "any model" is transformers: bge, e5, arctic-embed, MiniLM — the
models users actually start from. model2vec already solves the math
(distill a static token-embedding table from any sentence-transformers
teacher); its output format is ese's input format. This plan productizes
the bridge: one small Python tool that distills, **validates and repairs**
the artifacts (model2vec's own exporter has shipped tokenizers whose
post_processor ids contradict the vocab — the exact class ese now rejects),
and emits a directory `ESE_MODEL_DIR` consumes. End state:

```
uv run distill.py <any-hf-teacher> --out out/<slug>
ESE_MODEL_DIR=$PWD/out/<slug> ESE_NORMALIZE_OUTPUT=1 cargo test --features tests,dim-256
```

## Current state (facts)

- `ESE_MODEL_DIR` (plan 006) accepts a local dir with `model.safetensors`
  + `tokenizer.json`; no checksums needed; the parity oracle regenerates
  for whatever model is compiled. ese accepts WordPiece (BERT-uncased
  normalizer shapes) and Unigram (Metaspace `always`/`split:false`,
  normalizer chain of charsmap/single-char-pads/two whitespace
  regexes/strip) — see `guard_tokenizer_family` + `check_normalizer` in
  build.rs for the authoritative accepted shapes. Unsupported shapes fail
  the build with clear errors (that is correct behavior, not a bug).
- ese's specials cross-check (plan 007) panics when
  `post_processor.special_tokens[*].ids` contradict vocab name ids — the
  known model2vec-compaction bug (potion-base-8M: stored teacher ids
  101/102 = tokens `×`/`ß` in the compacted vocab).
- model2vec (Python): `from model2vec.distill import distill;
  m = distill(model_name=..., pca_dims=256); m.save_pretrained(dir)` —
  published model2vec repos contain exactly one F32 tensor named
  `embeddings` of shape `[vocab, dims]` plus a tokenizer.json, so
  `save_pretrained` output is expected to be ese-compatible modulo the
  tokenizer-consistency repair. Reference encode behavior (measured in
  plan 005): mean-pool ALL ids the tokenizer returns, then L2-normalize →
  always pair with `ESE_NORMALIZE_OUTPUT=1`.
- Python tooling convention in-repo: uv projects (`benches/py`,
  `spikes/greedy-fidelity`). Teachers download from HF (hundreds of MB) —
  distilled outputs must NOT be committed.
- Useful local sentences/checks: `spikes/greedy-fidelity/measure_fidelity.py`
  has a multilingual sentence list to crib from.

## Architecture (decided)

- New top-level `distill/` uv project: `pyproject.toml` (deps: model2vec,
  sentence-transformers, torch, safetensors, numpy, tokenizers), `distill.py`
  (CLI), `README.md`, `.gitignore` (ignore `out/`, caches).
- `distill.py` does, in order: (1) `model2vec.distill` with `--dims`
  (default 256) and the library's default weighting (record the actual
  applied settings in the output); (2) `save_pretrained` to a temp
  location, then copy/rename into `--out` as exactly `model.safetensors` +
  `tokenizer.json` (+ a small `DISTILL_INFO.md` recording teacher, rev,
  dims, model2vec version, settings); (3) **validate**: single F32 tensor,
  shape matches tokenizer vocab size, tokenizer family is WordPiece or
  Unigram, specials-id consistency (post_processor stored ids vs vocab name
  ids); (4) **repair** where safe: if post_processor special ids mismatch
  but the named tokens exist in the vocab, rewrite the stored ids to the
  vocab's ids and print what was repaired; anything unrepairable → exit
  non-zero with a clear message naming the issue (mirroring ese's
  vocabulary of errors).
- A tiny committed Rust example `examples/embed_stdin.rs` (~25 lines):
  reads UTF-8 lines from stdin, prints one embedding per line as
  space-separated f32 — the cross-language comparison harness (`cargo run
  --example embed_stdin`). Plain code; no deps.
- No CI distillation job (teachers are too heavy); the tool is documented
  as a local/offline step. CI coverage of distilled-model compilation is
  already represented by the ESE_MODEL_DIR code path exercised in tests.
- No changes to `build.rs` or `src/` runtime (the example file is
  additive).

## Stage 1 — build the tool (sub-agent 1)

In scope: `distill/` (create), `examples/embed_stdin.rs` (create),
`readme.md` (one new section "Distilling a transformer into an ese model").
Out of scope: `build.rs`, `src/`, CI, `api-py/`, basket testing beyond the
single gate model.

GATES (all must be green, evidence in report):

- **G1 — distill runs**: `cd distill && uv run python distill.py
  BAAI/bge-base-en-v1.5 --out out/bge-base` exits 0; `out/bge-base/`
  contains `model.safetensors`, `tokenizer.json`, `DISTILL_INFO.md`.
- **G2 — artifact shape**: a python check (part of distill.py's validate
  step, also runnable standalone via `--check out/bge-base`) confirms:
  exactly one tensor, name `embeddings`, dtype F32, shape `[vocab_size,
  256]` consistent with tokenizer vocab count; tokenizer family WordPiece
  or Unigram; specials-id consistency holds.
- **G3 — export fidelity (python-side oracle)**: encode 20 fixed sentences
  with `model2vec.StaticModel.from_pretrained(out/bge-base)` vs a manual
  mean-pool+L2 computed directly from the exported safetensors + tokenizer
  — cosine ≥ 0.999999 per sentence. (Proves the exported files ARE the
  model, before Rust enters the picture.)
- **G4 — ese end-to-end on the gate model**:
  `ESE_MODEL_DIR=$PWD/distill/out/bge-base ESE_NORMALIZE_OUTPUT=1 cargo
  test --features tests,dim-256` → all pass.
- **G5 — cross-language harness exists**: `cargo run --example
  embed_stdin` (default build) accepts `hello world\n` on stdin and prints
  512 f32s on one line; `cargo test` (default) still green; no warnings.
- **G6 — hygiene**: `git status` shows no untracked large files;
  `distill/out/` is gitignored; `distill/README.md` documents usage, the
  repair behavior, and the supported-family constraints with a pointer to
  build.rs guards.

## Stage 2 — the model basket (sub-agent 2, after reviewer approves Stage 1)

In scope: running the Stage-1 tool; a results table committed at
`distill/BASKET_RESULTS.md`; NO code changes except bug fixes to
`distill.py` explicitly justified in the report.

The basket (chosen to span families + the user-report comparator; "random"
in spirit but each entry earns its slot):

1. `BAAI/bge-base-en-v1.5` — WordPiece uncased (Stage 1's gate model; rerun)
2. `Snowflake/snowflake-arctic-embed-xs` — WordPiece; the transformer the
   external user report benchmarked ese against (87% finding) — narrative
   acceptance: "the comparator is now compilable"
3. `sentence-transformers/all-MiniLM-L6-v2` — WordPiece uncased, the most
   downloaded sentence-transformer
4. One SentencePiece/XLM-R-family teacher, e.g.
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` —
   exercises the Unigram path end-to-end through distillation
5. One EXPECTED-REJECT: a BPE-tokenizer teacher (e.g. a gte/modernbert
   variant) — the gate is that the failure is CLEAN and NAMED (tool or
   build error stating BPE unsupported), never silent wrongness

GATES per basket model 1–4 (evidence: a row in BASKET_RESULTS.md +
command outputs in the report):

- **B1 — distill + validate** exits 0 (repairs, if any, printed and noted).
- **B2 — ese parity**: `ESE_MODEL_DIR=... ESE_NORMALIZE_OUTPUT=1 cargo
  test --features tests,dim-256` → all pass (the build-time oracle
  regenerates per model — this is the same tight gate every supported
  model passes).
- **B3 — cross-language oracle**: 20 fixed sentences (English + CJK +
  code; reuse the spike's list) through `cargo run --example embed_stdin`
  (compiled for this model) vs `model2vec.StaticModel` encode in Python →
  cosine ≥ 0.999 per sentence (f32 cross-implementation budget; record the
  minimum).
- **B4 — semantic sanity**: 3 fixed triplets (anchor, similar, dissimilar)
  — cosine(anchor, similar) > cosine(anchor, dissimilar) for all 3, using
  ese outputs.

GATES for basket model 5: **B5** — the rejection message captured
verbatim; it must name the unsupported tokenizer type.

Final basket gate: **B6** — after all runs, the standard repo matrix is
still green (`cargo test`, `cargo test --features tests`,
`tests,quant-8` — run as separate commands) and `git status` is clean of
artifacts.

## STOP conditions (both stages)

- model2vec's `save_pretrained` output does NOT match the expected
  single-tensor/tokenizer.json shape — report the actual file listing and
  tensor names; do not improvise a converter without reporting first.
- The G3/B3 oracle comparison cannot reach threshold and the gap is not a
  fixable export bug — report per-sentence numbers; do not loosen gates.
- A basket model's distilled tokenizer needs repairs beyond the
  specials-id rewrite (e.g. structural changes) — record it as a clean
  failure with the error, move on; 3 of basket models 1–4 fully green is
  the minimum bar for the stage (4/4 is the target; a documented miss on
  the multilingual entry is acceptable if its rejection is clean).
- torch/model2vec installation fails in the environment — report versions
  attempted.

## Maintenance notes

- The tool's validate/repair step is the canonical place to absorb future
  model2vec exporter quirks — keep its error vocabulary aligned with
  build.rs guards so users see consistent messages.
- `examples/embed_stdin.rs` doubles as the harness for any future
  cross-implementation checks (wasm emitter, packs) — keep it dependency-free.
- Follow-up candidates after this lands: custom-vocab distillation flags
  (model2vec supports it; interacts with the specials repair), NanoBEIR
  quality table per distilled model (benches/py harness exists), and the
  layer-E passes operating on distilled artifacts.
