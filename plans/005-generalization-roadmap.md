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
>
> **Revision note (2026-06-12, v2)**: rewritten after a design conversation
> from the cofounder (Cursor agent transcript) introduced the IR-triple
> framing, the greedy-ification lever, and skeletoken. Tokenizer-type facts
> below were verified empirically against HF on 2026-06-12 — notably,
> `potion-base-8M` is **WordPiece**, not Unigram (the v1 plan had this
> wrong).

## Status

- **Priority**: P1 (read first; execute spike after plan 002)
- **Effort**: M for RFC + spike (the roadmap it plans is L, multi-stage)
- **Risk**: LOW (read-mostly; spike is additive)
- **Depends on**: plans/002-ci-and-quant-aware-parity.md (parity gate for the spike)
- **Category**: direction
- **Planned at**: commit `5155c13`, 2026-06-12 (revised same day)

## Why this matters

ese today is a compiler with exactly one compilable program: it ingests the
artifact pair (`model.safetensors` + `tokenizer.json`) of
`sentence-transformers/static-retrieval-mrl-en-v1` and emits a zero-runtime
static embedding library. The maintainers' stated direction (readme:
*"does not yet expose a stable custom-model interface"*), the public
positioning (https://www.flowercomputer.com/news/enzyme-garden/ invites
external developers in), an external user report wanting quality tiers, an
in-repo attempt at a second model (`codex/nife-mxbai-support`), and a
cofounder design exploration all point the same way. This document fixes the
architecture so the implementation plans compose instead of accreting
one-off hacks.

**The unifying frame** (from the cofounder design conversation, adopted
here): every static embedding model — however it was trained (distilled
from a teacher à la model2vec/NIFE, contrastively trained from scratch à la
the HF static-embeddings recipe, or co-occurrence-based like GloVe) — is the
same deployable artifact: a **`(TokenizerSpec, EmbeddingMatrix, PoolingSpec)`
triple**. Training provenance is upstream and invisible at deployment. ese
v2 is an AOT compiler whose IR is that triple: frontend loads it, optional
optimization passes transform it, the backend emits a self-contained
artifact, and a golden-vector verifier enforces fidelity. ese's existing
`build.rs` is already a single-target version of this compiler.

## Current state — what is hardcoded where

```
[HF artifacts] --build.rs--> [PHF-packed weights + codegen'd constants] --src/--> [encode()]
```

1. **Ingestion (`build.rs`)** — one model URL pair (plan 003 pins it; plan
   006 makes it selectable); vocab parsing assumes `tokenizer.json`
   `model.vocab` is a `{token: id}` map (true for WordPiece/WordLevel;
   Unigram stores `[[token, score], ...]`; BPE adds a `merges` list);
   special tokens hardcoded (plan 004 moves to by-name).
2. **Runtime tokenization (`src/pretokenizer.rs`)** — hardcodes the
   BERT-uncased normalizer (lowercase, NFD + strip marks, CJK isolation)
   and whitespace/punct pre-tokenization.
3. **Runtime token matching (`src/wordpiece.rs`)** — greedy longest-match
   WordPiece with hardcoded `"##"` prefix. Pooling: CLS + mean + SEP
   (magnitude semantics fixed by plan 004).

Also relevant:
- `model2vec-rs` is already a dev-dependency (`benches/bench_encode.rs`
  loads `minishlab/potion-base-8M`) — a built-in oracle for model2vec-family
  models.
- The `feature = "tests"` golden harness (build-time reference outputs via
  the `tokenizers` crate) is the seed of the compiler's verifier.
- The Python wheel (`api-py/`) hardcodes one model+quant config per build.

## Tokenizer census (verified against HF, 2026-06-12)

The HF `tokenizer.json` format is the de-facto standard; targeting it covers
essentially the whole field. Within it there are exactly **four `model`
algorithms**, and only the `model` needs a real algorithm at runtime —
normalizers/pre-tokenizers are composable filters, post-processors just add
special tokens (for mean pooling: "two more rows in the bag").

| `model.type` | algorithm | verified examples | runtime cost |
|---|---|---|---|
| WordLevel | dict lookup | GloVe exports | trivial |
| WordPiece | greedy longest-prefix + `##` | static-retrieval-mrl-en-v1, **potion-base-8M**, **M2V_base_output**, NIFE-mxbai (BERT-family teachers) | what ese has today |
| BPE | ordered merge-rank application | roberta/gpt-derived, modernbert embedders | merges table + rank engine |
| Unigram | Viterbi over scored lattice | **potion-multilingual-128M** (Metaspace), **multilingual-e5-base** (Precompiled normalizer) | scores + DP |

Pre-tokenizer dialects that matter: char-level+`##` (BERT), **ByteLevel**
(`Ġ`, byte→unicode alphabet; BPE world), **Metaspace** (`▁`; SentencePiece
world). Hard case to know about: `Precompiled` normalizers (SentencePiece
charsmap, e.g. e5 family) are effectively unreimplementable at runtime —
they must be folded at compile time or the model rejected.

Verified consequence for sequencing: **plan 006 (layer B) alone already
unlocks potion-base-8M, M2V_base_output, and NIFE-mxbai** — all WordPiece
with BERT normalization. The Unigram bucket (multilingual potion, e5/xlm-r)
is what layer A is for.

**Key relaxation** (cofounder conversation, adopted): static pooling is an
order-independent mean, so the runtime only needs the *multiset* of token
ids, not the exact sequence. This licenses compile-time tokenizer
simplification that would be unacceptable for a transformer.

## The architecture: 5-stage compiler over the IR triple

- **Frontend**: weight loader (safetensors → matrix; ese has this) +
  tokenizer loader (`tokenizer.json` → typed, validated `TokenizerSpec`).
- **IR**: `(TokenizerSpec, EmbeddingMatrix, PoolingSpec)`. `PoolingSpec` =
  `mean | weighted(zipf/idf) | sif` + how specials count (plan 004 fixed
  the denominator semantics — the IR must carry them explicitly).
- **Optimization passes (IR → IR)** — the differentiated-value layer, all
  cheap because the model is a row table: vocab **prune** (drop rows absent
  from a target corpus), **extend** (add domain words/phrases, init by
  mean-pooling their subword rows), **transplant** (swap tokenizers,
  re-derive rows over the new tokens' decompositions — exact under mean
  pooling), **reweight** (bake Zipf/IDF into rows), **decase/fold
  normalization** (bake lowercasing into vocab keys, shrink the runtime
  normalizer), **quantize** (upgrade global affine → per-dimension scales;
  optionally binary+Hamming for retrieval) + MRL truncation.
- **Backend**: PHF builder (keep CHD), tokenizer codegen (see the design
  question below), emitters — Rust crate (today), **wasm (headline target:
  CPU-only, small blob, browser/edge/lambda)**, C header, and an mmap-able
  flat pack (`bake` vs `load` are opposite deployment shapes; offer both —
  layer D).
- **Verifier**: every compiled artifact must reproduce reference
  `SentenceTransformer.encode()` within cosine ≥ threshold (and, post plan
  004, raw-vector magnitude parity for unquantized builds). Without this
  contract, "compile any model" is unshippable — tokenizer divergence is
  invisible until retrieval quality silently tanks. Document the one known
  probabilistic edge: PHF OOV lookups are rejected by a 64-bit verify hash,
  so false positives are astronomically rare but nonzero.

### The load-bearing design question: one matcher or four?

Two candidate architectures for tokenizer generality; **the RFC must decide,
and the spike (Step 3) provides the data**:

1. **Per-family runtimes**: implement Unigram Viterbi (and later BPE
   merge-rank) in Rust alongside the existing WordPiece matcher. Faithful;
   more runtime code, more codegen paths, Precompiled normalizers still
   unsolved.
2. **Greedy-ification (compile-time transform)**: use
   [skeletoken](https://github.com/stephantul/skeletoken) (Python; Pydantic
   models of `tokenizer.json`; verified API: `decase_vocabulary()`,
   `make_model_greedy()`, vocab edits with merge-table repair; round-trips
   through `tokenizers`) to convert BPE/Unigram models into greedy
   longest-match form at compile time. The runtime then needs exactly ONE
   matcher — ese's existing one — plus ~2 extra pre-tokenizer dialects
   (ByteLevel alphabet map, Metaspace). Pay a tokenization-fidelity cost at
   compile time; the verifier measures exactly how much cosine it costs.
   Couples naturally with the transplant/extend passes (re-deriving rows
   after changing segmentation recovers fidelity).

If (2)'s fidelity cost is acceptable, layer A shrinks from "implement two
tokenizer algorithms + dialects" to "implement two pre-tokenizer dialects +
a canonicalization step". Caveats the RFC must address honestly:

- Greedy-ifying Unigram changes the token *multiset*, not just order — the
  cosine cost is an empirical, per-model, per-language question.
  Multilingual models (the main reason Unigram matters) may be hit hardest
  on non-whitespace scripts; the spike must measure non-English inputs.
- skeletoken introduces Python into the model-compilation path. Design
  constraint: the Python phase runs **once per model**, emitting a
  canonical spec consumable by `build.rs` — pure-Rust consumers never run
  Python for already-canonicalized models (the default model ships
  pre-canonicalized; `ESE_MODEL_DIR` from plan 006 is the natural seam).
- skeletoken's tokenizer hashing (the cache-key/equivalence idea) is
  **roadmap-only, not implemented** (verified 2026-06-12) — don't make the
  artifact cache depend on it.

### Layers and sequencing

```
001-004 (hygiene, CI, pinning)
  → B (plan 006: manifest ingestion)   [M — already unlocks potion-base-8M, M2V, NIFE-mxbai]
  → A (tokenizer generality)           [size depends on the matcher decision above]
  → E (optimization passes: prune/extend/transplant/reweight/quant)  [M, incremental, high product value]
  → C (distillation front-end: wrap model2vec/pynife so ANY teacher → triple)  [M]
  → D (emitters: wasm headline, C header, mmap packs)                [M]
```

C and D are parallelizable after B; E can start as soon as the IR boundary
exists (each pass is independent). Non-goals (state in the RFC so they stay
decided): no transformer inference at runtime; no training framework; no
vector DB. Static-embedding quality limits (word-order/negation blindness)
are inherent — the answer is tiers, distillation quality, and vocab surgery
(phrase rows attack fragmentation directly), not architecture changes.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Parity gate | `cargo test --features tests` | all pass |
| Oracle present | `grep -n "model2vec_rs" benches/bench_encode.rs` | match |
| Inspect a tokenizer | `curl -sL https://huggingface.co/<repo>/resolve/main/tokenizer.json \| python3 -c "import json,sys; t=json.load(sys.stdin); print(t['model']['type'], (t.get('normalizer') or {}).get('type'), (t.get('pre_tokenizer') or {}).get('type'))"` | prints the three types |

## Scope

**In scope**:
- `docs/rfc-generalization.md` (create — the RFC)
- A throwaway spike under `examples/` or `benches/` (clearly marked, not
  wired into default builds)
- `plans/README.md` (status row)

**Out of scope** (do NOT do as part of this plan):
- Any change to `src/` or `build.rs` behavior — implementation belongs to
  plan 006 and successors.
- Building layers A/C/D/E themselves.
- Adding skeletoken or any Python tooling to the build — spike-side
  experimentation only.

## Git workflow

- Branch: work on `advisor/improvements` (the project's integration branch)
  or a child of it.
- Commit style: short lowercase imperative.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Complete the tokenizer support matrix

Extend the verified census above to the full benchmark set
(`benches/py/models.txt`) plus NIFE-mxbai: for each model record
`model.type`, normalizer chain, pre-tokenizer chain, post-processor /
special tokens, vocab representation, and (Unigram) score presence. Already
verified 2026-06-12: potion-base-8M (WordPiece/BertNormalizer),
potion-multilingual-128M (Unigram/Sequence/Metaspace), M2V_base_output
(WordPiece/BertNormalizer), static-retrieval-mrl-en-v1 (WordPiece),
multilingual-e5-base (Unigram/Precompiled/Sequence). Produce the
support-matrix table for the RFC with a "blocked on" column (which layer or
pass each model needs).

**Verify**: the RFC draft contains the table with ≥ 8 rows.

### Step 2: Write the RFC

`docs/rfc-generalization.md` covering: the IR triple and 5-stage
architecture (adapt freely from this plan), the support matrix, the
**matcher decision** (per-family runtimes vs greedy-ification) presented
with the spike data from Step 3 and a recommendation, the two-phase
compile / canonical-spec boundary (Python skeletoken phase ↔ Rust backend;
constraint: no Python in the consumer path), the build-config interface
(env vars per plan 006; record why per-model Cargo features were rejected:
features must be additive, per-model features collide under unification),
the optimization-pass layer E with the four vocab-surgery passes, emitter
priorities (wasm headline; mmap pack alignment notes: `src/lookup.rs`
transmutes to `#[repr(C, align(8))]` statics, a runtime loader must copy
into aligned memory and bounds-check; keep the static API and add a
separate dynamic module), the verifier contract (cosine + magnitude gates,
PHF OOV note), and open questions (BPE demand timeline, Precompiled
normalizer handling, who hosts canonicalized/distilled artifacts).

**Verify**: `test -f docs/rfc-generalization.md`;
`grep -c "^## " docs/rfc-generalization.md` → ≥ 7.

### Step 3: Spike — measure the greedy-ification fidelity cost on a real Unigram model

This replaces v1's "Unigram oracle" spike and answers the load-bearing
question with data. Target model: **`minishlab/potion-multilingual-128M`**
(static, Unigram, Metaspace — verified). Procedure (Python, throwaway, may
live in `benches/py/` style with uv; do not add deps to the Rust crate):

1. Baseline: encode a fixed multilingual input set (≥200 sentences spanning
   English + at least 3 non-whitespace/non-Latin-script languages — e.g.
   zh, ja, th — plus some code/identifiers) with the reference
   (`model2vec` Python or `sentence-transformers` loading the model).
2. Greedy-ify: load the model's `tokenizer.json` with skeletoken, apply
   `make_model_greedy()` (plus `decase_vocabulary()` if applicable),
   re-serialize, and encode the same inputs by: tokenize with the
   transformed tokenizer (`tokenizers` lib) → mean-pool rows from the
   model's embedding matrix (mirroring the reference pooling incl. any
   output normalization the reference applies — discover and document).
3. Report per-language cosine distributions (min/p5/median) between
   baseline and greedy-ified encodings.

Decision thresholds for the RFC recommendation: median ≥ 0.99 and p5 ≥ 0.97
per language → greedy-ification is viable as the default path (with
per-family runtimes as a fallback for high-fidelity needs); below that →
layer A implements Unigram Viterbi natively and greedy-ification becomes an
optional optimization pass. Either way the numbers go in the RFC appendix.

**Verify**: the spike script runs end-to-end and emits the per-language
table; numbers pasted into the RFC appendix.

### Step 4: File the follow-up plan stubs

Append to `plans/README.md` under "Dependency notes" the successor plans
with one-line scopes reflecting the RFC's decision:
`007-tokenizer-generality (layer A — approach per RFC §matcher-decision)`,
`008-optimization-passes (layer E)`, `009-distill-frontend (layer C)`,
`010-emitters-wasm-and-packs (layer D)` — status `NOT YET PLANNED`.

**Verify**: `grep -c "NOT YET PLANNED" plans/README.md` → ≥ 4.

## Test plan

The spike (Step 3) is the test: a machine-checkable measurement of the
riskiest assumption. No production tests change.

## Done criteria

- [ ] `docs/rfc-generalization.md` exists: IR triple, support matrix (≥8
      models), matcher decision w/ spike data, two-phase boundary, layer E
      passes, emitter priorities, verifier contract, non-goals, open
      questions
- [ ] Spike ran on potion-multilingual-128M with ≥200 multilingual inputs;
      per-language cosine table in the RFC appendix
- [ ] `cargo test --features tests` still exits 0 (nothing production broke)
- [ ] `plans/README.md` updated (status row + 4 successor stubs)

## STOP conditions

Stop and report back (do not improvise) if:

- skeletoken cannot round-trip potion-multilingual-128M's tokenizer (parse
  or re-serialization failure) — report the error verbatim; the two-phase
  architecture loses its frontend and the RFC must say so.
- The reference encoding path (step 3.1) doesn't reproduce published model
  behavior (sanity-check a couple of similarity pairs) — the baseline is
  wrong; fix before measuring, or report.
- You find existing generalization work in the repo beyond the NIFE branch
  (drift check) — reconcile, don't duplicate.
- Network/HF access unavailable — finish the RFC with the matrix marked
  partial; skip the spike and mark this plan BLOCKED on the spike.

## Maintenance notes

- The RFC is the decision record: plans 006+ must stay consistent with it
  or update it explicitly in the same change.
- The matcher decision should be revisited if skeletoken ships tokenizer
  hashing or a Rust port emerges.
- The support matrix goes stale — re-run the census command when adding a
  model family.
- Reviewer of the RFC: the key judgment calls are (1) greedy-ification vs
  per-family runtimes (data-driven via spike), (2) Python-phase boundary
  (no Python for consumers), (3) keeping per-model Cargo features rejected,
  (4) layer E pass ordering. Challenge there.
