# Plan 005: Generalization roadmap — from one hardcoded model to "compile any embedding model" (design doc + spike)

> **Executor instructions**: This is a *design/spike plan*, not a
> build-everything plan. The deliverables are (1) a written RFC at
> `docs/rfc-generalization.md` and (2) one running spike with a
> machine-checkable verification. Follow the steps; honor STOP conditions.
> When done, update the status row in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 5155c13..HEAD -- build.rs src/`
> Expect changes from plans 001–004. The architecture facts below (what is
> hardcoded where) should still hold; if tokenization is no longer hardcoded
> BERT-WordPiece, this roadmap is already underway — STOP and reconcile with
> whoever did that.

## Status

- **Priority**: P1 (read first; execute spike after plan 002)
- **Effort**: M for RFC + spike (the roadmap it plans is L, multi-stage)
- **Risk**: LOW (read-mostly; spike is additive)
- **Depends on**: plans/002-ci-and-quant-aware-parity.md (parity gate for the spike)
- **Category**: direction
- **Planned at**: commit `5155c13`, 2026-06-12

## Why this matters

ese today is a compiler with exactly one compilable program: it ingests the
artifact pair (`model.safetensors` + `tokenizer.json`) of
`sentence-transformers/static-retrieval-mrl-en-v1` and emits a zero-runtime
static embedding library. The maintainers' stated direction (readme-rewrite
branch: *"It is not **yet** a stable framework for plugging in arbitrary
embedding models"*), the public positioning
(https://www.flowercomputer.com/news/enzyme-garden/ invites external
developers in), an external user report benchmarking ese against transformer
alternatives and wanting quality tiers, and an in-repo attempt at a second
model (`codex/nife-mxbai-support`) all point the same way: the
single-model pipeline should become a general system for optimizing *any*
embedding model. This document fixes the architecture for that path so the
individual implementation plans (starting with plan 006) compose instead of
accreting one-off hacks.

## Current state — what is hardcoded where

The pipeline has three stages; generalization = making each stage
config-driven instead of model-specific:

```
[HF artifacts] --build.rs--> [PHF-packed weights + codegen'd constants] --src/--> [encode()]
```

1. **Ingestion (`build.rs`)** — hardcoded:
   - one model URL pair (plan 003 pins it; plan 006 makes it selectable)
   - vocab parsing assumes `tokenizer.json` `model.vocab` is a
     `{token: id}` map — true for **WordPiece** only. **Unigram** stores
     `vocab: [[token, score], ...]` (and needs the scores at runtime);
     **BPE** stores `vocab` + a `merges` list.
   - special tokens by ID (plan 004 moves to by-name; still assumes
     `[CLS]`/`[SEP]` exist and are added)
2. **Runtime tokenization (`src/pretokenizer.rs`)** — hardcodes the
   BERT-uncased normalizer: lowercase (line ~26), NFD + strip combining
   marks, CJK char isolation, whitespace/punctuation splitting. But every
   model's `tokenizer.json` *declares* its normalizer and pre-tokenizer
   (e.g. `{"type": "BertNormalizer", "lowercase": true, ...}` or
   `{"type": "Metaspace", ...}`). A cased model compiled through today's
   pipeline would be silently lowercased — wrong embeddings, no error.
3. **Runtime token matching (`src/wordpiece.rs`)** — hardcodes WordPiece
   greedy longest-match with `"##"` continuation prefix (line 32) and
   `MAX_WORD_LEN=100`. Unigram needs Viterbi over token scores; BPE needs
   merge-rule application.

Also relevant:
- `benches/bench_encode.rs` already loads `minishlab/potion-base-8M` via the
  `model2vec-rs` dev-dependency — i.e. the repo already contains a
  **reference implementation (oracle) for Unigram-family static models**.
- `benches/py/models.txt` benchmarks potion/M2V/GloVe/static-retrieval
  models — the comparison set IS the demand list for compilable models.
- The Python wheel (`api-py/`) hardcodes one model+quant config per build.

## The four layers (B → A → C → D)

Ordered by dependency, not by value — each is independently shippable.

### Layer B — Manifest-driven model ingestion (plan 006, ready to execute)

Replace "the model is a constant in build.rs" with "the model is build
configuration": `ESE_MODEL=<hf-repo>@<revision>` (+ checksums) or a local
artifact directory, with the current model as the pinned default. Derive
dims/special-tokens/flags from the artifacts; **fail the build with a clear
error** when the model needs a tokenizer family the runtime doesn't have
yet. Absorbs the NIFE branch (its `ModelProfile`, per-model cache keys,
optional output normalization) without its fatal flaw (one Cargo feature
per model — features must be additive in Cargo's model; per-model features
collide under feature unification and explode combinatorially).

Scope guard: with only WordPiece supported, layer B alone already covers any
BERT-vocab static model (e.g. NIFE-mxbai distillations).

### Layer A — Tokenizer-general compilation

Make build.rs read the `normalizer`, `pre_tokenizer`, and `model` sections
of `tokenizer.json` and emit a matching runtime configuration
(codegen'd constants/tables, same zero-runtime philosophy):

- **Normalizer**: map the declared chain (BertNormalizer fields, Lowercase,
  NFD/NFC/NFKC, StripAccents, Replace, Prepend...) onto a small set of
  compile-time flags consumed by `pretokenizer.rs`. Unsupported component →
  build error naming it.
- **Pre-tokenizer**: Whitespace/BertPreTokenizer (today's behavior),
  Metaspace (SentencePiece-style, for Unigram models), ByteLevel (BPE).
- **Model**:
  - WordPiece — today's `wordpiece.rs` (parameterize the `##` prefix and
    max word length from the JSON instead of hardcoding).
  - **Unigram — the high-value add**: vocab with log-prob scores packed into
    the existing PHF (score alongside the weight row), Viterbi
    longest-segmentation at runtime. This unlocks the entire
    model2vec/potion family — the models this repo already benchmarks
    against.
  - BPE — defer unless a concrete model demands it (merge tables are a
    different runtime shape; byte-level pretok touches normalization too).

Verification oracle: `model2vec-rs` (already a dev-dep) — encode the same
inputs through ese-compiled potion vs model2vec-rs at f32, require cosine
≥ 0.999999 AND relative L2 < 1e-4 (same gates as plans 002/004).

### Layer C — Distillation front-end ("any model" literally)

ese's input format is model2vec's *output* format. A small companion tool —
recommended: `distill/` Python project managed like `benches/py` (uv,
pyproject) — that wraps `model2vec.distill` so that:

```
ese-distill <any-hf-embedding-model> --out artifacts/   # teacher → static artifacts
ESE_MODEL_DIR=artifacts cargo build --release           # artifacts → compiled library
```

turns *any* HF embedding model (transformer or static) into an ese build.
Quality gates reuse the existing harness: `benches/py/nanobeir.py` for
NanoBEIR scores, published per-model in a quality/speed tier table. This is
also the honest answer to the external user report (static ≈ 87% of a small
transformer's retrieval quality): better teachers and tuned distillation
move that number; tiers let users choose.

NIFE-mxbai (a third party hand-distilling mxbai into ese-consumable form)
is evidence this workflow is wanted; layer C productizes it.

### Layer D — Runtime-loadable model packs (distribution unlock)

Compile-time baking stays the flagship. But one pip wheel / one wasm bundle
per model contradicts "any model" for Python/JS consumers. Define
`.esepack` = the already-emitted artifacts (`weights.bin`, `seeds.bin`,
`verify.bin` + a constants header) in one file; add a feature-gated runtime
loader. Two costs to design around (RFC must address):
- alignment: `src/lookup.rs` transmutes `include_bytes!` data to
  `&[Slot; TABLE_SIZE]` via `#[repr(C, align(8))]` statics — a runtime
  loader must copy into an aligned allocation (or use an aligned mmap) and
  bounds-check the header.
- types: `DIMENSIONS` is `const` and the API returns `[f32; DIMENSIONS]`;
  runtime packs imply `Vec<f32>`/slices — keep the static API untouched and
  add a separate `ese::dynamic` module/type so existing consumers see zero
  change.

### Sequencing and what NOT to do

```
001-004 (hygiene, CI, pinning)
  → B (plan 006)            [S/M — covers all WordPiece static models]
  → A (Unigram first)       [L — covers potion/model2vec family]
  → C (distill front-end)   [M — covers ANY teacher model]
  → D (runtime packs)       [M — distribution; parallel with C]
```

Non-goals (state them in the RFC so they stay decided): no transformer
inference at runtime (that's the moat — ese stays a static-table engine);
no training framework; no vector DB. Quality limits of static embeddings
(word-order/negation blindness, per the external report) are inherent —
the answer is tiers + distillation quality, not architecture changes.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Parity gate | `cargo test --features tests` | all pass |
| Oracle present | `grep -n "model2vec_rs" benches/bench_encode.rs` | match |
| Inspect a Unigram tokenizer | `python3 -c "import json,urllib.request; t=json.load(urllib.request.urlopen('https://huggingface.co/minishlab/potion-base-8M/resolve/main/tokenizer.json')); print(t['model']['type'], t['normalizer'], t['pre_tokenizer'])[:1]"` (adapt freely) | prints `Unigram` + declared components |

## Scope

**In scope**:
- `docs/rfc-generalization.md` (create — the RFC)
- `benches/` or a new `spikes/` dir for the Step 3 spike (throwaway,
  clearly marked, not wired into default builds)
- `plans/README.md` (status row)

**Out of scope** (do NOT do as part of this plan):
- Any change to `src/` or `build.rs` behavior — implementation belongs to
  plan 006 and successors.
- Building layer A/C/D themselves.

## Git workflow

- Branch: `advisor/005-generalization-rfc`
- Commit style: short lowercase imperative.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Catalog the target models' tokenizer configs

For each of: `minishlab/potion-base-8M`, `minishlab/potion-multilingual-128M`,
`minishlab/M2V_base_output`, `sentence-transformers/static-retrieval-mrl-en-v1`,
`stephantulkens/NIFE-mxbai-embed-large-v1` — download `tokenizer.json`
(HF `resolve/main` URLs) and record: `model.type`, normalizer chain,
pre-tokenizer chain, special/added tokens, vocab representation, and for
Unigram whether scores are present. Produce the support matrix table for the
RFC (which layer-A components each model requires).

**Verify**: the RFC draft contains a table with ≥ 5 rows and columns
(model, tokenizer type, normalizer, pre-tokenizer, blocked-on).

### Step 2: Write the RFC

`docs/rfc-generalization.md` covering: the four layers as specified above
(copy/adapt freely), the support matrix from Step 1, the build-config
interface decision (env vars vs manifest file — recommend: env vars
`ESE_MODEL`, `ESE_MODEL_DIR`, `ESE_MODEL_SHA256`,... with a documented
default; record why per-model Cargo features were rejected: Cargo features
must be additive, per-model features collide under unification), the
`.esepack`/alignment notes for layer D, the oracle strategy
(tokenizers crate for WordPiece parity, model2vec-rs for Unigram parity,
NanoBEIR for quality), and open questions (BPE demand, multilingual
normalization coverage, who hosts distilled artifacts).

**Verify**: `test -f docs/rfc-generalization.md` and it contains sections
matching: `grep -c "^## " docs/rfc-generalization.md` → ≥ 6.

### Step 3: Spike — prove the Unigram oracle path

Goal: de-risk layer A's hardest claim ("we can match model2vec's Unigram
encoding"). Write a throwaway Rust test or example (e.g.
`benches/spike_unigram_oracle.rs` as a `[[bench]]`-less dev target or an
`examples/` file): load `minishlab/potion-base-8M` via `model2vec-rs` AND
via the `tokenizers` crate (both already in dev/build deps), encode ~10
fixed sentences with (a) model2vec-rs `encode()` and (b) your own
mean-pool over `tokenizers`-produced token IDs against the model's
embedding matrix — confirm (a)≡(b) within float noise (cosine ≥ 0.999999).
This proves the runtime semantics ese must replicate are "tokenize +
mean-pool" with no hidden steps (or discovers the hidden steps — e.g.
model2vec applies an output norm — which is precisely what the spike is
for; document whatever you find in the RFC).

**Verify**: `cargo test --features tests <spike test name>` (or `cargo run
--example spike_unigram_oracle`) → prints per-sentence cosine, all
≥ 0.999999, exit 0. Paste the output into the RFC appendix.

### Step 4: File the follow-up plan stubs

Append to `plans/README.md` under "Dependency notes" the planned successor
plans with one-line scopes: `007-unigram-runtime (layer A)`,
`008-distill-frontend (layer C)`, `009-runtime-model-packs (layer D)` —
status `NOT YET PLANNED`, so the next `/improve` run reconciles instead of
re-auditing.

**Verify**: `grep -c "NOT YET PLANNED" plans/README.md` → ≥ 3.

## Test plan

The spike (Step 3) is the test: a machine-checkable demonstration that the
Unigram oracle path works. No production tests change.

## Done criteria

- [ ] `docs/rfc-generalization.md` exists with support matrix + 4 layers +
      non-goals + open questions
- [ ] Spike runs and its cosine output is ≥ 0.999999 on all inputs (or the
      RFC documents exactly which extra steps model2vec applies, with the
      spike updated to match and then passing)
- [ ] `cargo test --features tests` still exits 0 (nothing production broke)
- [ ] `plans/README.md` updated (status row + successor plan stubs)

## STOP conditions

Stop and report back (do not improvise) if:

- The spike's (a) vs (b) comparison cannot be brought above 0.999 cosine
  even after accounting for documented model2vec post-processing — the
  Unigram path has unknown semantics and layer A's estimate is wrong;
  report the discrepancy data.
- `model2vec-rs` cannot load potion-base-8M in this environment (network/
  API drift) — note versions and stop the spike, finish the RFC.
- You find existing generalization work in the repo beyond the NIFE branch
  (drift check) — reconcile, don't duplicate.

## Maintenance notes

- The RFC is the decision record: future plans (006+) must stay consistent
  with it or update it explicitly in the same change.
- Re-run the Step 1 catalog when adding a model family — tokenizer.json
  schemas vary more than HF docs suggest.
- Reviewer of the RFC: the key judgment calls are (1) env-var interface
  vs manifest file, (2) Unigram-before-BPE ordering, (3) keeping per-model
  Cargo features rejected. Each has its rationale inline; challenge there.
