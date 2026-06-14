# Plan 016: Phase 3 — close the distillation-quality gap (reframed by plan 013 data)

> **Status: DESIGN plan, now UNBLOCKED (plan 013 DONE).** Reframed
> 2026-06-14 by the NanoBEIR measurements. Execution splits into gated
> sub-plans (017+); this file fixes the priority and the targets.

## Status

- **Priority**: P1 (the "most performant" axis — and now the clear #1 gap)
- **Effort**: L (research-shaped; multiple tracks)
- **Risk**: MED–HIGH (training/quality work, not mechanical)
- **Depends on**: 013 (the quality map — DONE), 010 (distill harness)
- **Category**: direction (RFC layer C quality + layer E)
- **Planned at**: `8d9ede2`, reframed at `cd0ff9c`, 2026-06-14

## What plan 013 measured (and why it reframes this plan)

NanoBEIR mean nDCG@10, M4 Pro, via the validated `EseEncoder`
(`benches/py/QUALITY_TABLE.md`):

| model | NanoBEIR | note |
|---|---|---|
| static-retrieval-mrl-en-v1 (default) | **0.4957** | matches its model card → harness validated |
| NIFE-mxbai (distilled+tuned) | 0.4685 | 108k vocab |
| potion-multilingual-128M | 0.3699 | multilingual, 500k vocab |
| **raw-distilled bge / arctic / MiniLM / e5** | **0.096–0.111** | **15–19% of teacher retention** |
| (teachers bge / arctic / MiniLM) | 0.56–0.63 | transformer reference |

**The finding**: ese compiles faithfully (default 0.4957 ≈ card; distilled
models ≡ model2vec at cosine 1.0). But **raw `model2vec.distill()` — what
our `distill/` tool does today — retains only ~17% of the teacher's
retrieval signal.** The high-quality static models (static-retrieval-mrl,
potion) are not raw distillations; they were **contrastively
trained/tuned** (Tom Aarsen's recipe; minishlab's Tokenlearn).

**Consequence for priority**: vocab surgery (prune/extend/reweight) is
*polish* — it cannot move a 0.10 model to 0.43. The order-of-magnitude lever
is **distillation quality**. This plan now LEADS with that.

## Track A (primary) — distillation quality: close 0.11 → ~0.43

The goal: make `distill/` produce models in the static-retrieval-mrl /
potion quality tier, not raw-PCA tier. Two known recipes, both producing a
`(tokenizer, table, pooling)` triple ese already compiles:

- **A1 — model2vec Tokenlearn / contrastive post-training**: after the
  initial distill, contrastively fine-tune the static table on a corpus
  (model2vec ships this; minishlab's potion models use it). Likely the
  cheapest path to the tier jump. Sub-plan gate: a Tokenlearn-tuned
  distill of one teacher scores ≥ 0.35 NanoBEIR (vs 0.11 raw), compiles
  through ese unchanged, crosscheck 1.0.
- **A2 — contrastive-from-scratch** (the HF static-embeddings recipe, the
  RFC's third provenance): train a StaticEmbedding with sentence-transformers
  on a retrieval dataset, then compile. This is how static-retrieval-mrl
  itself was made (0.4957). Heavier; the reference-quality ceiling.

Sub-plan 017 = A1 (highest leverage, cheapest). 018 = A2 (if A1 underdelivers).
Both reuse plan 013's `EseEncoder` NanoBEIR harness as the gate — no quality
claim without a measured score.

## Track B (secondary) — vocab surgery (polish + the wasm-size lever)

Worth doing AFTER Track A lands a quality baseline worth polishing, and
independently valuable for **binary/wasm size** (plan 015 found the default
wasm is 60 MB, weights-dominated):

- **B1 — prune** (S): drop rows absent from a target corpus → smaller PHF +
  binary, lossless in-domain. *This is the direct fix for the 60 MB wasm
  bundle* — prune to a deployment vocab. Gate: same in-domain NanoBEIR,
  smaller table + `.wasm`.
- **B2 — extend** (M): phrase/bigram rows vs word-order blindness. Gate:
  NanoBEIR lift on multiword-entity queries.
- **B3 — reweight** (M): Zipf/IDF in pooling, reusing the per-row scalar
  from plan 009. Gate: NanoBEIR lift vs naive mean at equal speed.

## Design constraints (carried from the prior 12 plans)

- Each transform is opt-in, never changing default behavior silently
  (plan-012 lesson).
- Every quality claim re-runs plan 013's NanoBEIR table — no pass ships on
  vibes; the measured-gain bar is explicit.
- Track A is where the strategic payoff is: it is the difference between
  "ese compiles any model fast" (true today) and "ese turns any model into
  a *hyper-performant* model" (the stated aim — currently only true for
  already-well-trained inputs).

## Next action

Write sub-plan **017 (Track A1 — Tokenlearn/contrastive post-training in
`distill/`)** as the next gated execution plan, targeting a measured jump
from 0.11 → ≥ 0.35 NanoBEIR on one teacher. Track B follows or parallels for
the wasm-size win.

## Open questions (for the maintainer)

- Corpus for A1/B1: general (ship broad quality) vs domain-specific (let
  users tune)? A product decision.
- Is reference-tier quality (A2, ~0.50) a goal, or is "good enough +
  tiny+fast" (A1, ~0.35) the product? Determines A1-vs-A2 investment.
