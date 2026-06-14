# Plan 013: NanoBEIR quality-tier table — measure what we can now compile

> **Executor instructions**: Follow step by step; the GATES are the contract
> (command + output evidence each). Honor STOP conditions. SKIP updating
> `plans/README.md`. This is a measurement plan — no Rust runtime changes.
>
> **Drift check (run first)**: HEAD `8d9ede2` or a descendant.

## Status

- **Priority**: P1 (foundational — informs Phase 3)
- **Effort**: M (mostly compute)
- **Risk**: LOW (additive Python; no core changes)
- **Depends on**: plans 008/010/011/012 (the models to measure)
- **Category**: direction / measurement
- **Planned at**: `8d9ede2`, 2026-06-14

## Why this matters

ese can now compile ~8 models (default + NIFE + potion-multilingual +
distilled bge/arctic/MiniLM/e5/paraphrase-multilingual) with **zero
published quality numbers**. The original blog leaned on NanoBEIR; we owe a
speed/quality tier table that (a) is the public follow-up to the post,
(b) gives users the data to pick a model, and (c) tells *us* where the
quality gaps are so Phase 3 (vocab surgery) targets the right thing. The
harness already exists (`benches/py/nanobeir.py`, `model_loader.py`).

## Current state

- `benches/py/nanobeir.py`: runs `sentence_transformers.evaluation.NanoBEIREvaluator`
  on a model loaded by `model_loader.py::load_model(name)`. `load_model("ours")`
  returns `OurSentenceTransformerWrapper` (wraps the ese Python wheel — ONE
  model per wheel); other names load real `SentenceTransformer` models.
- `examples/embed_stdin.rs`: pipes a corpus through the compiled ese binary
  (any `ESE_MODEL`/`ESE_MODEL_DIR` + flags) → one embedding per line.
- `spikes/specials-parity/check.py` and `distill/crosscheck.py` already
  contain faithful Python re-implementations of ese's pooling (WordPiece +
  Unigram, with `ESE_NORMALIZE_OUTPUT`/`ESE_SKIP_UNK`/`ESE_ADD_SPECIAL_TOKENS`
  semantics), proven to match the compiled binary at cosine 1.0.
- Model caches in `target/ese-cache/`: default, NIFE, potion-multilingual,
  potion-base-8M. Distilled artifacts must be (re)generated with `distill/`.

## The approach (decided)

Quality and speed come from different places:

- **Quality (NanoBEIR)**: build a Python `EseEncoder` (a
  sentence-transformers-compatible `.encode()` wrapper) that loads an
  artifact dir's `model.safetensors` + `tokenizer.json` and reproduces
  ese's exact pooling — REUSE the pooling code already in
  `distill/crosscheck.py` / `spikes/specials-parity/check.py`, do not
  re-derive. **Validate it once** against the compiled binary
  (`embed_stdin`) at cosine ≥ 0.999, then trust it for all models (proven
  identical → same NanoBEIR score as the binary, far faster than building N
  wheels). Run `NanoBEIREvaluator` on it per model.
- **Speed (QPS)**: comes from the *compiled binary* (the differentiator).
  Measure via `examples/embed_stdin.rs` timing or `benches/py/bench.py`
  on ≥2 models (the default + one large-vocab multilingual). All models
  share the same PHF/baked-weights fast path, so 2–3 points characterize
  the curve; note vocab size + dims per model.
- **Baselines**: for distilled models, also score the *teacher* with real
  `SentenceTransformer` (retention % = ese score / teacher score). Teacher
  NanoBEIR on CPU is slow — time-box to the small NanoBEIR set; if a teacher
  is too slow, record published numbers from its model card and mark the row
  "card".

## Model set (each a table row)

| key | how to get artifacts |
|---|---|
| static-retrieval-mrl-en-v1 (default) | built-in (the blog's model) |
| NIFE-mxbai | `ESE_MODEL=...@970a015...` (dim-1024, normalize) |
| potion-multilingual-128M | `ESE_MODEL=...@73908c3...` (dim-256, normalize) |
| distilled bge-base-en-v1.5 | `distill/distill.py BAAI/bge-base-en-v1.5` |
| distilled arctic-embed-xs | `distill/distill.py Snowflake/snowflake-arctic-embed-xs` |
| distilled all-MiniLM-L6-v2 | `distill/distill.py sentence-transformers/all-MiniLM-L6-v2` |
| distilled multilingual-e5-base | `distill/distill.py intfloat/multilingual-e5-base` |

(potion-base-8M optional bonus now that plan 012 unblocked it.)

## GATES (evidence = command + output)

- **G1 — EseEncoder validated**: a one-off check shows the Python
  `EseEncoder` for the default model matches `embed_stdin` (dim-1024 or
  dim-512) at per-sentence cosine ≥ 0.999 on a 10-sentence probe. Without
  this, the quality numbers are unfounded.
- **G2 — NanoBEIR per model**: `NanoBEIREvaluator` runs for every row in the
  model set; record the headline metric (mean NanoBEIR nDCG@10) per model.
- **G3 — retention vs teacher**: for ≥3 distilled models, the teacher's
  NanoBEIR (measured or card) and the retention % (ese/teacher) recorded.
- **G4 — speed**: QPS (single + batched) for ≥2 ese models measured from the
  compiled binary; methodology + hardware noted (it is NOT an M4 — note the
  machine; the blog's absolute numbers are M4-specific).
- **G5 — table committed**: `benches/py/QUALITY_TABLE.md` with columns
  `model | family | dims | vocab | NanoBEIR (ese) | NanoBEIR (teacher/ref) |
  retention % | QPS single | QPS batch | notes`, plus a short prose summary
  of where ese sits on the speed/quality frontier and which models retain
  best. `git status` clean of artifacts (distilled outputs + datasets
  gitignored).

## STOP conditions

- G1 fails (EseEncoder ≠ binary) — the pooling reimplementation is wrong;
  report the diverging inputs, do not publish numbers built on it.
- NanoBEIR datasets cannot download (network) — record which models were
  measured, mark the rest pending, still commit a partial table.
- A teacher cannot load (trust_remote_code / arch) — use its model-card
  number and mark "card"; do not block the table on it.

## Scope

**In scope**: `benches/py/` (new `ese_encoder.py`, edits to
`model_loader.py`/`nanobeir.py` if needed, `QUALITY_TABLE.md`), `.gitignore`
entries for datasets/artifacts.
**Out of scope**: any Rust change; `build.rs`; the distill tool's logic
(only run it); CI.

## Maintenance notes

- The `EseEncoder` becomes the reusable quality-eval harness for Phase 3 —
  every vocab-surgery pass re-runs this table to prove a quality gain.
- Keep absolute QPS numbers hardware-tagged; for the public table, relative
  (vs MiniLM on the same machine) is the honest comparison.
- Reviewer: confirm G1 actually ran (the whole table's validity rests on
  it) and that retention % uses the same NanoBEIR config for ese and teacher.
