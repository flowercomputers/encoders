# Plan 012: Specials off by default — match the real static-embedding reference

> **Executor instructions**: Follow step by step; gates are the contract
> (command + output evidence each). Honor STOP conditions. SKIP updating
> `plans/README.md`. Items A and B are independent.
>
> **Drift check (run first)**: HEAD must be `17781dc` or a descendant that
> has not touched `build.rs`/`src/` since.

## Status

- **Priority**: P1 (correctness — changes default model output)
- **Effort**: M (mostly deletion; one behavior flip + a harness check)
- **Risk**: MED — **changes ese's default embedding output** (specials no
  longer pooled). Cosine-direction change is small (S-matrix corr 0.9997)
  but real; stored default-model vectors should be re-embedded for exactness.
- **Depends on**: 010 (distill/crosscheck harness)
- **Planned at**: commit `17781dc`, 2026-06-13

## Why this matters

A verification spike (`spikes/specials-parity/`, prompted by model2vec
maintainer feedback) proved that the real `sentence-transformers`
`StaticEmbedding` reference for `static-retrieval-mrl-en-v1` does **not**
pool special tokens: mean pooling WITHOUT `[CLS]`/`[SEP]` reproduces
`SentenceTransformer.encode()` at cosine **1.0000**, while ese's current
WITH-specials pooling lands at **0.9994** (S-matrix geometry error 0.00141
vs 0.00000). ese has pooled CLS+SEP since the initial commit; the parity
suite never caught it because the build-time oracle assumed
`add_special_tokens=true` and the runtime faithfully matched that wrong
assumption. The maintainer states the whole static-embedding category
(model2vec AND sentence-transformers) never inserts specials. This plan
makes specials-off the default, which simultaneously: (1) fixes the default
model, (2) **dissolves the potion-base-8M rejection** — with specials never
pooled, the inconsistent post-processor ids are irrelevant, so that model
(and its class) compiles correctly, and (3) deletes a subsystem (CLS/SEP
accumulation, the specials-id cross-check) that was solving a non-problem.

## Current state (at `17781dc`)

- `build.rs::derive_add_special_tokens` reads `post_processor`: BertProcessing
  → true; TemplateProcessing referencing CLS/SEP → true (+ a cross-check
  panic on id mismatch via `cross_check_special_token_ids`); else false.
  The default model has BertProcessing → derives **true** today.
- `build.rs` emits `ADD_SPECIAL_TOKENS` const; reads (currently no)
  `ESE_ADD_SPECIAL_TOKENS` env.
- `src/lib.rs::encode_single` (WordPiece path): when `ADD_SPECIAL_TOKENS`,
  accumulates `CLS` first, inits `token_count = 2`, accumulates `SEP` after
  the loop, divides by `n+2`. When false: no specials, `token_count = 0`,
  divide by `n` (guarded).
- `src/unigram.rs`: never pools specials (Unigram models already
  specials-free); unaffected.
- `generate_testdata` oracle uses `tokenizer.encode(input, add_special_tokens)`
  with the derived flag — so flipping the flag regenerates the oracle to
  match, keeping parity green at the NEW (correct) behavior.
- `cross_check_special_token_ids` panics potion-base-8M
  (`bf8b056651a2c21b8d2565580b8569da283cab23`): post-processor ids 101/102
  vs vocab 2/3.
- `distill/crosscheck.py` compares ese vs model2vec via per-sentence cosine.
- Models that must keep passing: default (static-retrieval-mrl), NIFE-mxbai,
  potion-multilingual, basket distills (bge/arctic/MiniLM/paraphrase/e5).
  All except the default already derive/run specials-OFF, so only the
  default model's output changes.

## Item A — specials-off default + opt-in, remove the dead subsystem

1. `build.rs`: replace `derive_add_special_tokens` with an env read:
   `ADD_SPECIAL_TOKENS = (env ESE_ADD_SPECIAL_TOKENS == "1")`, default false;
   `cargo:rerun-if-env-changed=ESE_ADD_SPECIAL_TOKENS`. Document at the emit
   site WHY (static-embedding references do not pool specials; see
   spikes/specials-parity). When true AND a post-processor names CLS/SEP,
   keep the existing cross-check (it is only meaningful when specials are
   actually pooled). When the opt-in is set but the vocab lacks CLS/SEP →
   the existing panic in `main()` still applies.
2. Delete `derive_add_special_tokens` and (unless reused by the opt-in
   path) `cross_check_special_token_ids`. If the opt-in path keeps the
   cross-check, retain only that function and call it solely under the
   opt-in. No dead code, no `#[allow(dead_code)]` bandaids.
3. `src/lib.rs`: the `ADD_SPECIAL_TOKENS` branch in `encode_single` stays
   (now false by default → CLS/SEP code is compiled-out by const). One doc
   line: specials are not pooled by default; `ESE_ADD_SPECIAL_TOKENS=1`
   opts in (rarely needed — static references do not use them).
4. `readme.md` + `docs/rfc-generalization.md`: correct any claim that ese
   derives specials from the post-processor; state the specials-off default
   and the spike finding. Keep it to the minimum honest edit.

## Item B — adopt the S-matrix method as the harness quality check

`distill/crosscheck.py`: in addition to per-sentence cosine, compute the
upper-triangular similarity matrix S over the sentence set for both ese and
model2vec outputs and report `mean|S_ese − S_m2v|` and their correlation
(the maintainer's retrieval-relevant metric). Keep the per-sentence cosine
too. Plain numpy, ~15 lines (mirror `spikes/specials-parity/check.py`'s
`upper_tri_sim`).

## GATES (evidence = command + output)

- **A1 — default model now matches the real reference**: re-run
  `spikes/specials-parity/check.py` after rebuilding the default at the new
  behavior — but simpler and self-contained: `cargo test --features tests`
  passes (oracle regenerated specials-off; runtime matches), AND a one-off
  check shows the default build's `encode_single("hello world")` now equals
  the specials-OFF numpy pooling (the spike already proved that == the ST
  reference). Record the spike's post-fix S-matrix delta (expect ~0.00000
  for `mean|S_ref − S_ese|`).
- **A2 — potion-base-8M now COMPILES** (previously rejected):
  `ESE_MODEL=minishlab/potion-base-8M@bf8b056651a2c21b8d2565580b8569da283cab23
  ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 cargo test --features tests,dim-256`
  → all pass. Then crosscheck it against model2vec (it now should match,
  since both pool content-tokens-only): record min cosine + S-matrix delta.
- **A3 — opt-in still works**: `ESE_ADD_SPECIAL_TOKENS=1 cargo test
  --features tests` → passes (oracle + runtime both specials-on, parity
  holds); confirms the path is not broken, just non-default.
- **A4 — no regressions**: NIFE acceptance, potion-multilingual acceptance,
  the five standard feature combos, and one distilled-model build
  (bge + quant-8 + SKIP_UNK) — all green. Run separately (zsh).
- **A5 — dead code gone**: `grep -c "derive_add_special_tokens" build.rs`
  → 0; if the cross-check was removed, `grep -c "cross_check_special_token_ids"
  build.rs` → 0; `cargo check` + `cargo check --features quant-8` → zero
  warnings.
- **B1 — harness S-matrix check**: `distill/crosscheck.py` on any cached
  artifact prints the S-matrix delta + correlation alongside per-sentence
  cosine.
- **Z**: `cargo check --workspace` green; `git status` clean of artifacts.

## STOP conditions

- A1's post-fix spike shows `mean|S_ref − S_ese|` is NOT ~0 — the flip did
  not actually remove the specials contribution; report the numbers.
- potion-base-8M (A2) fails parity or crosscheck after the flip for a reason
  other than specials — capture per-sentence values; do not force it.
- Any A4 regression — the change leaked beyond the default/potion behavior.
- Removing the cross-check would also remove a check the opt-in path needs —
  keep it scoped under the opt-in instead of deleting.

## Scope

**In scope**: `build.rs`, `src/lib.rs` (doc + the existing const branch),
`src/test/mod.rs` (only if an input/threshold needs adjusting — unlikely),
`readme.md`, `docs/rfc-generalization.md`, `distill/crosscheck.py`,
`plans/README.md` (rejected→compiles note for potion in the rejected list).
**Out of scope**: `src/unigram.rs` (already specials-free),
`src/wordpiece.rs` (UNK logic unchanged), `src/pretokenizer.rs`, quant
scheme, CI, `api-py/`.

## Maintenance notes

- This **changes the default model's embedding output** (specials no longer
  pooled). Anyone with stored default-model vectors compared via dot product
  or exact cosine should re-embed; retrieval ranking barely moves (corr
  0.9997). Call this out in release notes — it is the correctness counterpart
  to plan 004's magnitude change.
- After this, `ADD_SPECIAL_TOKENS` is an explicit opt-in, not a derivation —
  the post-processor is a tokenization detail, not a pooling instruction.
- potion-base-8M moves from the "rejected" list to "compiles"; update the
  index's rejected section and the model2vec issue thread can be closed as
  "ese now handles it correctly by not pooling specials".
- Reviewer: confirm A1's post-fix S-matrix delta is ~0 against the real ST
  reference (not just internal parity), and that the opt-in path (A3) still
  cross-checks ids when specials are actually requested.
