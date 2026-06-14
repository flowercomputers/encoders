# Plan 017: Tokenlearn-quality distillation — close the 0.11 → 0.40 NanoBEIR gap

> **Executor instructions**: Follow step by step; GATES are the contract
> (command + output evidence each). This is research-shaped — honor the STOP
> conditions strictly (compute can run away). SKIP updating
> `plans/README.md`. A measured PARTIAL lift that proves the lever + ships
> the integration is success; reference-tier quality is the stretch.
>
> **Drift check (run first)**: HEAD `32363d6` or a descendant.

## Status

- **Priority**: P1 (the single highest-leverage quality move)
- **Effort**: L (training; compute-bound)
- **Risk**: HIGH (research + heavy compute; STOP conditions matter)
- **Depends on**: 010 (distill tool), 012 (potion-base-8M unblocked), 013
  (NanoBEIR harness — the measurement gate)
- **Category**: direction (RFC layer C — distillation quality)
- **Planned at**: `32363d6`, 2026-06-14

## Why this matters

Plan 013 measured the truth: ese compiles faithfully and fast, but **raw
`model2vec.distill()` retains only 15–19% of the teacher's retrieval
signal** (NanoBEIR 0.10 vs 0.43+ for trained static models). The high-quality
static models (static-retrieval-mrl 0.4957, potion 0.37) were
**contrastively post-trained**, not raw-distilled — minishlab's recipe is
**Tokenlearn** (https://github.com/MinishLab/tokenlearn): after distillation,
featurize a corpus with the teacher and train the static table to reproduce
the teacher's mean-pooled sentence embeddings. This plan brings that recipe
into ese's `distill/` so *any* teacher becomes a high-quality static model —
the difference between "ese compiles any model fast" (done) and "ese turns
any model into a *hyper-performant* model" (the stated aim).

## Current state

- `distill/distill.py`: raw `model2vec.distill()` → artifacts. Validated,
  but the output scores ~0.10 NanoBEIR (plan 013).
- `benches/py/` (plan 013): `ese_encoder.py` (faithful pooling, validated ==
  compiled binary at cosine 1.0) + `run_nanobeir_ese.py` (NanoBEIR via
  `NanoBEIREvaluator`) + `QUALITY_TABLE.md`. **This is the measurement gate
  — reuse it; do not reinvent.**
- `distill/crosscheck.py`: ese-vs-model2vec parity (cosine + S-matrix).
- potion-base-8M (`minishlab/potion-base-8M`, a Tokenlearn'd English model,
  256-dim, WordPiece) is now compilable (plan 012 unblocked it) but was NOT
  in plan 013's table.
- Model caches in `target/ese-cache/`; potion-base-8M cache present.

## GATES (evidence = command + output)

### G1 — proof of ceiling (cheap; do FIRST, de-risks everything)

Compile potion-base-8M (already Tokenlearn'd) and measure it on the plan-013
NanoBEIR harness:
`ESE_MODEL=minishlab/potion-base-8M@bf8b056651a2c21b8d2565580b8569da283cab23 ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 cargo build --release --example embed_stdin --features dim-256`,
then evaluate via `EseEncoder` (config: dim-256, normalize, skip_unk).
**Expected ≥ 0.35** (vs raw-distilled 0.10). This proves (a) Tokenlearn-tier
quality is reachable, (b) ese compiles it faithfully. If potion-base-8M
scores ~0.10 like the raw distills, STOP — the harness or flags are wrong,
and the whole premise needs rechecking. Record the score.

### G2 — Tokenlearn is usable (tractability gate)

Investigate the actual `tokenlearn` / `model2vec` training API in this
environment (`pip show tokenlearn model2vec`; read its README / entry points;
`tokenlearn --help` or the Python API). Record: the exact featurize→train
call sequence, the corpus format it expects, and whether it runs CPU-only.
If `tokenlearn` cannot be installed or is fundamentally GPU-only / infeasible
on this machine in a bounded time, STOP and report — then deliver the
documented recipe + the G1 proof as the outcome (do not burn hours fighting
the toolchain).

### G3 — demonstrate the lift (the deliverable)

Pick ONE small, CPU-fast teacher (recommend `sentence-transformers/all-MiniLM-L6-v2`
— plan 013 measured its raw distill at 0.104, a clean baseline to beat).
Featurize a **scoped** corpus (a tractable subset — e.g. 50k–200k texts from
a readily available HF dataset such as `sentence-transformers/gooaq`,
`wikimedia/wikipedia` sample, or `allenai/c4` streamed; pick what loads
fast, record the corpus + size), run Tokenlearn training, then:
1. Save the trained static model to an `ESE_MODEL_DIR` artifact.
2. Compile through ese (`ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 ... dim-256`),
   confirm parity (`cargo test --features tests,dim-256` on that dir) and
   crosscheck vs the trained model2vec (cosine ≥ 0.999).
3. Measure NanoBEIR via the plan-013 harness.

**Gate: NanoBEIR ≥ 0.25** (more than 2× the 0.104 raw baseline), **target
≥ 0.35**. Honest framing: a compute-limited single run may not reach
reference 0.45 — a clear, measured lift that proves the lever and ships the
integration is the bar. Record raw-vs-tokenlearn'd side by side.

### G4 — integrate + document

Add a Tokenlearn path to `distill/` (e.g. `distill.py --train <corpus>` or a
`distill/tokenlearn.py` companion) so the recipe is repeatable, with a
`distill/README.md` section and the corpus/compute caveats. Append the G1
(potion-base-8M) and G3 (tokenlearn'd MiniLM, raw vs trained) rows to
`benches/py/QUALITY_TABLE.md` with a prose note that Tokenlearn is the
quality lever.

### Z — hygiene

`cargo test` (default) still green; `git status` clean of artifacts
(corpora, featurized caches, trained models all gitignored under
`target/`/`distill/out`); the Rust crate is UNCHANGED (this is a
Python/tooling + measurement plan — `git diff src/ build.rs` empty).

## STOP conditions

- **G1 potion-base-8M scores ~0.10** (not ≥0.35) — the premise/harness is
  wrong; report and stop before any training.
- **Tokenlearn uninstallable / GPU-only / no CPU path in bounded time** —
  deliver the documented recipe + G1 proof; mark G3 BLOCKED with the
  toolchain reason. Do NOT spin for hours.
- **Featurization/training exceeds a reasonable wall-clock budget**
  (say > ~2–3 h for the scoped corpus) — cut the corpus further and report
  the partial result; a smaller-but-real lift is the deliverable, not a
  perfect score.
- **The trained model fails ese parity/crosscheck** (cosine < 0.999) — the
  trained artifact's tokenizer/format diverged from what ese compiles;
  report the artifact shape (it may need the same padding/specials repairs
  the distill tool already does).
- Any need to change the Rust crate (`src/`/`build.rs`) — out of scope;
  Tokenlearn produces the same `(tokenizer, table, pooling)` triple ese
  already compiles. If it doesn't, STOP and report what's different.

## Scope

**In scope**: `distill/` (tokenlearn integration + README), `benches/py/`
(QUALITY_TABLE.md update; reuse the harness), `.gitignore` entries.
**Out of scope**: the Rust crate (`src/`, `build.rs`); vocab surgery
(Track B / later); contrastive-from-scratch (A2 / plan 018); CI.

## Maintenance notes

- This is Track A1 of Phase 3 (plan 016). If Tokenlearn underdelivers on
  CPU, A2 (contrastive-from-scratch via sentence-transformers StaticEmbedding,
  the static-retrieval-mrl recipe) is the heavier fallback — its own plan.
- The corpus is the quality knob: bigger/cleaner corpus → higher score.
  Document what was used so a GPU re-run can push toward reference tier.
- Reviewer: confirm G1 (potion-base-8M ≥ 0.35) independently — it's the
  proof the whole approach is sound — and that G3's lift is measured on the
  same NanoBEIR config as the 0.104 baseline (apples to apples).
