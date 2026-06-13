# Plan 011: Unigram Sequence pre-tokenizer (XLM-R family) + quant threshold tighten

> **Executor instructions**: Follow this plan step by step; gates are the
> contract (command + output evidence each). Honor STOP conditions. SKIP
> updating `plans/README.md`. Items A (pre-tokenizer) and C (threshold) are
> independent; item B (specials) is discovered-scope inside A's acceptance.
>
> **Drift check (run first)**: HEAD must be `784921b` or a descendant that
> has not touched `build.rs`/`src/` since.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (new pre-tokenizer path; existing Metaspace path must be inert)
- **Depends on**: 008 (Unigram runtime), 009 (per-row quant), 010 (distill tool)
- **Planned at**: commit `784921b`, 2026-06-13

## Why this matters

The plan-010 basket's one miss: distilled XLM-R-family teachers
(`paraphrase-multilingual-MiniLM-L12-v2`, and the e5 family) are cleanly
rejected because their tokenizers use a `Sequence` pre-tokenizer, while
ese's Unigram runtime accepts only bare `Metaspace{always, split:false}`.
Supporting this one shape unlocks the multilingual half of distillation.
Plus one recorded follow-up: the quant parity threshold (0.9999) is far
looser than the measured minima (≥ 0.9999977) — tighten it.

## Current state (verified 2026-06-13)

- Teacher tokenizers for BOTH `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  and `intfloat/multilingual-e5-base` (fetched from HF):
  - `model.type: Unigram`
  - `pre_tokenizer: Sequence([WhitespaceSplit, Metaspace{replacement: "▁", add_prefix_space: true}])`
    — note the LEGACY Metaspace serialization (`add_prefix_space`, no
    `prepend_scheme`/`split` fields). How tokenizers-0.21.4 deserializes
    legacy `add_prefix_space` and what `split` defaults to is fact #1 to
    extract from the vendored source
    (`~/.cargo/registry/src/*/tokenizers-0.21.4/src/pre_tokenizers/metaspace.rs`
    + `whitespace.rs` for WhitespaceSplit). Mirror exactly.
  - normalizer: bare `Precompiled` (top-level, not a Sequence) — the 008
    guard/runtime may already accept this (charsmap with all other op
    flags false); verify, don't assume.
  - `post_processor: TemplateProcessing` — XLM-style; the distilled
    artifact's template likely references `<s>`/`</s>` (NOT `[CLS]`/`[SEP]`)
    and the vocab is COMPACTED (basket measured 249,999 entries), so the
    stored ids may be stale (the known ×/ß hazard class). What model2vec's
    distilled tokenizer actually contains is fact #2 — inspect the
    artifact you distill, do not assume.
- Basket failure message (distill/BASKET_RESULTS.md, Model 4): build panic
  "has pre_tokenizer type 'Sequence'; ese's Unigram runtime only supports
  Metaspace pre-tokenizers".
- `src/unigram.rs::apply_metaspace` implements bare Metaspace
  (split:false → whole string, one Viterbi pass). With
  WhitespaceSplit+Metaspace, each whitespace-separated word becomes its
  own "▁word" chunk Viterbi-segmented independently — likely cheaper per
  call, but semantics must come from the vendored source, not this guess.
- model2vec reference behavior: mean-pool all ids its tokenizer returns
  EXCEPT unk (build with `ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1`).
- quant parity threshold: `src/test/mod.rs` uses 0.9999 for quant builds;
  measured minima are ≥ 0.9999977 (plan 002) and the per-row oracle
  (plan 009) tracks the runtime even tighter.
- Model caches in the main checkout (`target/ese-cache/`) survive: default,
  NIFE, potion-multilingual, potion-base-8M. Teachers are in the user HF
  cache. The crosscheck harness: `distill/crosscheck.py` +
  `examples/embed_stdin.rs`.

## Item A — Sequence([WhitespaceSplit, Metaspace]) for Unigram builds

- Step 1: extract the semantics facts from the vendored tokenizers source
  (legacy `add_prefix_space` deserialization; WhitespaceSplit's exact
  whitespace definition; how Sequence chains pre-tokenizers — each stage
  splits the OUTPUT chunks of the previous). Record with file:line refs in
  the report AND as a comment block where implemented.
- Step 2: build.rs — extend the Unigram pre-tokenizer guard to accept
  exactly this Sequence shape (and the existing bare Metaspace); emit a
  const for the pre-tokenizer mode (e.g. `METASPACE_SPLIT_WHITESPACE: bool`).
  Anything else still rejected by name.
- Step 3: src/unigram.rs — when the const is set: after normalize(), split
  on whitespace per fact #1's definition, apply metaspace per word, run
  `viterbi_segment` per word, pool across all words' rows. Keep
  the existing path byte-identical when the const is false. KISS: one
  `for word in ...` loop, no abstraction.

## Item B — specials, discovered scope

After distilling the acceptance model, inspect its tokenizer: does the
post_processor's single template insert tokens? If NO (model2vec stripped
it, like potion-multilingual): nothing to do. If YES: extend
`derive_add_special_tokens`/the cross-check to handle the template's
ACTUAL token names generically (the inserted prefix/suffix tokens by name,
id cross-checked against the vocab — reuse the existing machinery, do not
hardcode `<s>`). If that requires pooling specials in the UNIGRAM runtime
(currently WordPiece-only machinery): STOP and report the artifact's
post_processor JSON before implementing — that is a scope decision for the
reviewer. (The crosscheck gate would expose any silent mismatch either
way.)

## Item C — tighten quant parity threshold

`src/test/mod.rs`: quant threshold 0.9999 → 0.99999, comment updated with
the measured minima justifying it.

## GATES

- **A1**: semantics facts recorded (file:line) + in-code comment.
- **A2 — acceptance**: `cd distill && uv run python distill.py
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --out
  out/paraphrase-ml` exits 0, then
  `ESE_MODEL_DIR=$PWD/distill/out/paraphrase-ml ESE_NORMALIZE_OUTPUT=1
  ESE_SKIP_UNK=1 cargo test --features tests,dim-256` → ALL pass.
- **A3 — crosscheck**: `distill/crosscheck.py` on that artifact → min
  cosine ≥ 0.999 over the 20 sentences (must include CJK + code), 3/3
  triplets, UNK incidence recorded.
- **A4 — existing paths inert**: potion-multilingual acceptance (bare
  Metaspace) green; the five standard feature combos green; NIFE green;
  bge-distilled + quant-8 + SKIP_UNK composition green. Run each as a
  separate command (zsh).
- **A5 — rejection still honest**: a pre-tokenizer Sequence containing
  anything OTHER than [WhitespaceSplit, Metaspace] is still rejected
  naming the component (build a tiny /tmp fixture by editing a copy of the
  acceptance artifact's tokenizer.json).
- **A6 — stretch, non-blocking**: distill `intfloat/multilingual-e5-base`
  and run A2/A3 on it. If it fails, record the exact error and move on —
  do NOT debug beyond one look.
- **B1**: the distilled artifact's post_processor JSON quoted in the
  report + which branch of Item B applied.
- **C1**: quant-8 and quant-16 parity green at 0.99999; the commit shows
  the threshold + comment change only.
- **Z**: zero warnings (`cargo check`, plus quant-8 variant); `cargo check
  --workspace` green; `git status` clean of artifacts.

## STOP conditions

- Item B requires Unigram-runtime specials pooling (see Item B) — report
  the post_processor JSON and stop that item.
- A3 minimum lands below 0.99 — per-sentence values + segmentation diff
  for the worst sentence, then stop (likely a pre-tokenizer semantics
  mismatch; do not loosen the gate).
- The legacy-Metaspace deserialization semantics differ from what the
  basket artifact actually needs (e.g. distilled tokenizer uses the NEW
  serialization) — report both shapes; support what the artifact contains.
- Any A4 regression — the new path leaked into existing paths.

## Scope

**In scope**: `build.rs`, `src/unigram.rs`, `src/test/mod.rs` (threshold),
`src/lookup.rs` (only if a const needs plumbing), `readme.md` (one line on
the newly supported shape), `distill/BASKET_RESULTS.md` (append a
model-4-revisited row + e5 stretch row).
**Out of scope**: WordPiece paths, `src/pretokenizer.rs`, BPE, CI changes,
`api-py/`, vocab surgery.

## Maintenance notes

- After this, the "supported tokenizer shapes" story is: WordPiece
  (BERT-uncased) + Unigram (bare Metaspace | WhitespaceSplit+Metaspace).
  The guard lists remain the single source of truth.
- If A6 (e5) passes, update the support matrix claim in the RFC in a
  follow-up — e5-family distillation was the original multilingual goal.
- Reviewer: verify A1's facts against the vendored source yourself;
  scrutinize the per-word pooling count semantics (denominator = total
  tokens across words, matching the oracle).
