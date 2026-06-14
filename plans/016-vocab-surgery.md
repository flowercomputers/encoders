# Plan 016: Vocab surgery + reweighting (layer E) — the lever past the static ceiling

> **Status: BLOCKED on plan 013 (NanoBEIR table).** This is a design/spike
> plan; its concrete targets come from where 013 shows ese loses quality.
> Do NOT dispatch execution until 013 is DONE and the quality gaps are known.

## Status

- **Priority**: P1 (the "most performant" axis of the stated aim)
- **Effort**: L (multi-pass; each pass independent)
- **Risk**: MED (changes compiled weights / pooling)
- **Depends on**: 013 (quality baseline + gap analysis), 010 (distill harness)
- **Category**: direction (RFC layer E)
- **Planned at**: `8d9ede2`, 2026-06-14

## Why this matters

Speed is solved and class-leading; **quality** is the open frontier. Static
embeddings are structurally word-order/negation-blind (~"87% of a transformer"
per the user report). Because ese is literally a row table, vocab operations
are cheap and mostly training-free — this is the lever to push the quality
number up without abandoning the static paradigm. Sequenced after 013 so each
pass is justified by a measured gap and verified by a measured gain on the
same NanoBEIR harness.

## The passes (each its own sub-plan once 013 lands)

- **E-prune** (S): drop vocab rows absent from a target corpus → smaller PHF
  + binary, lossless in-domain. Pure win for embedded/edge deploys. Gate: a
  pruned model scores identically on in-domain NanoBEIR with a smaller table.
- **E-extend** (M): add domain whole-words / bigrams / phrases as new rows,
  initialized by mean-pooling their subword decomposition (free) or one-shot
  teacher distillation (NIFE-style). Directly attacks WordPiece/Unigram
  fragmentation and the word-order weakness (a precomputed "new york" row
  beats averaging "new" + "york"). Gate: measurable NanoBEIR lift on queries
  with multiword entities, no regression elsewhere.
- **E-reweight** (M): bake Zipf/IDF token weights into the pooling (a
  per-row scalar consumed in `accumulate`/`pool_and_mean`) so common tokens
  contribute less than rare ones — model2vec does this post-hoc; ese can bake
  it. Gate: NanoBEIR lift vs naive mean at equal speed.
- **E-binary** (M, optional): 1-bit/dim + Hamming distance as a retrieval
  index tier, stacking on MRL truncation. Gate: retrieval recall vs storage
  tradeoff curve.

## Design constraints (decided now, to keep later sub-plans on rails)

- Each pass is an **optional, opt-in transform** (env flag or
  `ESE_MODEL_DIR` artifact-side step), NEVER changing default behavior
  silently — the plan-012 lesson.
- Reweighting reuses the per-row params machinery from plan 009 (a row
  scalar already flows through `accumulate`); do not invent a parallel one.
- Every pass re-runs plan 013's `EseEncoder` NanoBEIR table and must show a
  gain (or honest neutrality) — no pass ships on vibes.
- Vocab-extend rows interact with the PHF construction and the specials/UNK
  logic — extend is the riskiest; spike it first.

## Next action (when unblocked)

After 013, read its `QUALITY_TABLE.md`, pick the 1–2 passes whose target gap
is largest, and split each into its own gated execution plan (017+) modeled
on the prior plans. Update this file with the chosen order and the measured
baselines they must beat.

## STOP / open questions

- Where does the quality actually leak (short queries? multiword entities?
  multilingual?) — 013 answers this; without it, pass selection is guesswork.
- Corpus dependence: prune/extend need a target corpus — is the use case
  general (ship a broad vocab) or domain-specific (let users prune)? Product
  decision to surface to the maintainer.
