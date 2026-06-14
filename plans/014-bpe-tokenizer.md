# Plan 014: BPE tokenizer support — the last major tokenizer family

> **Executor instructions**: Follow step by step; GATES are the contract
> (command + output evidence each). Honor STOP conditions. SKIP updating
> `plans/README.md`.
>
> **Drift check (run first)**: HEAD `8d9ede2` or a descendant that has not
> touched `build.rs`/`src/` since. NOTE: plan 015 (wasm) may run in parallel
> in a separate worktree — your branch is independent; the reviewer merges
> sequentially and resolves any `src/lib.rs` dispatch overlap.

## Status

- **Priority**: P1 (the substantial gap to universal compilation)
- **Effort**: L (third runtime tokenizer algorithm)
- **Risk**: MED (new path; WordPiece + Unigram paths must stay inert)
- **Depends on**: 008 (Unigram established the multi-family pattern), 010
  (distill tool produces the acceptance artifact), 012 (specials-off)
- **Category**: direction (layer A completion)
- **Planned at**: `8d9ede2`, 2026-06-14

## Why this matters

ese compiles WordPiece and Unigram. **BPE** is the one substantial family
left — it covers RoBERTa, GPT-2-derived, ModernBERT, GTE, and most
LLM-derived embedders, both for direct compilation and as distillation
targets. The plan-010 basket rejected `gte-modernbert-base` precisely here.
After this plan, a BPE-tokenizer model distilled by model2vec (which
preserves the teacher's tokenizer) compiles through the existing
`ESE_MODEL`/`ESE_MODEL_DIR` interface, and the compiler covers ~every
`tokenizer.json` `model.type` in the wild (only WordLevel/GloVe, a trivial
dict-lookup, would remain — out of scope here).

## Current state

- `build.rs::guard_tokenizer_family` (~line 962): panics on
  `model.type != "WordPiece" && != "Unigram"`, naming BPE/WordLevel as
  unsupported.
- `build.rs` builds a PHF over the tokenizer vocab (token string → row
  index). For Unigram it also emits `scores.bin`; for WordPiece a dummy.
  The PHF construction is family-agnostic (it just needs token→row pairs).
- `src/lookup.rs`: `Slot { verify, [row_min,row_scale for quant], weight }`;
  `lookup_slot(key)`. A model-kind dispatch already exists
  (`IS_UNIGRAM` const → `src/unigram.rs`, else WordPiece in `src/lib.rs`).
- `src/wordpiece.rs`, `src/unigram.rs`: the two existing runtime tokenizers
  (greedy longest-match; Viterbi). Both end in mean-pool of slot rows with
  the `ESE_NORMALIZE_OUTPUT`/`ESE_SKIP_UNK` flags.
- The model-fact derivation, env interface, per-row quant, oracle
  (`generate_testdata` via the `tokenizers` crate), and parity/quality
  tests are all family-agnostic and reused.
- The authoritative semantics are the vendored `tokenizers` crate (built
  under `--features tests`): read
  `~/.cargo/registry/src/*/tokenizers-0.21.4/src/models/bpe/` (`model.rs`,
  `word.rs`) and `pre_tokenizers/byte_level.rs` BEFORE writing the runtime.

## Acceptance model

Direct BPE static models are rare; the real path is **distill a
BPE-tokenizer transformer** (model2vec keeps the teacher tokenizer → a
static BPE model) then compile. Use
`sentence-transformers/all-distilroberta-v1` (RoBERTa → ByteLevel BPE;
loads cleanly, no `trust_remote_code`). Confirm its `tokenizer.json`
`model.type == "BPE"` and `pre_tokenizer` is `ByteLevel` first.

## Architecture (decided — do not redesign)

- **Model-kind**: extend the dispatch (e.g. an emitted
  `TOKENIZER_KIND` / `IS_BPE` const) so `encode_single` routes BPE to a new
  `src/bpe.rs`. WordPiece and Unigram paths stay byte-for-byte identical.
- **Merges as data**: BPE needs the ordered merge list. Emit `merges.bin`
  (or a codegen'd table) mapping a merged-pair key → rank, packed so the
  runtime can look up "is (A,B) a merge, and at what rank". KISS: a second
  PHF over the merge pairs (key = `A + "\u{0000}" + B` or the byte form the
  vendored code uses) → rank u32; or, if simpler and verified equivalent,
  store rank alongside each vocab token. Pick the shape that mirrors the
  vendored `tokenizers` BPE merge application exactly — that fidelity is
  the whole game (the oracle uses the real crate).
- **ByteLevel pre-tokenizer** (`src/bpe.rs`): map raw UTF-8 bytes to the
  ByteLevel unicode alphabet (the 256-byte → printable-char table), handle
  the `Ġ` space marker and `add_prefix_space`, and split per ByteLevel's
  regex/whitespace rule. Extract the exact alphabet + regex + prefix
  semantics from `pre_tokenizers/byte_level.rs` (fact #1, recorded in
  report + a comment block). Assume `dropout: None` (verify; reject
  non-null dropout at build).
- **Runtime**: ByteLevel pre-tokenize → for each chunk, apply BPE merges by
  ascending rank until no merge applies (the vendored `Word` algorithm) →
  look up each resulting token's slot → mean-pool with the existing flags.
  Keep it plain: explicit loops, named functions, comments stating the
  semantics facts. UNK handling per the vendored source (BPE may emit
  `<unk>` or byte-fallback — record which and mirror it; ByteLevel BPE
  typically has no UNK because every byte is representable).
- **Guards**: `guard_tokenizer_family` accepts `model.type == "BPE"` when
  the pre-tokenizer is the supported ByteLevel shape and dropout is null;
  anything else still rejected by name.

## GATES (evidence = command + output)

- **G1 — semantics facts**: the BPE merge-application rule, the ByteLevel
  alphabet + prefix/regex rule, and the UNK/byte-fallback behavior recorded
  with file:line from the vendored source, AND as a comment block atop
  `src/bpe.rs`.
- **G2 — acceptance**: distill the teacher
  (`cd distill && uv run python distill.py sentence-transformers/all-distilroberta-v1 --out out/distilroberta`),
  then `ESE_MODEL_DIR=$PWD/distill/out/distilroberta ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 cargo test --features tests,dim-256`
  → all pass (parity vs the tokenizers-crate oracle, on the BPE path).
- **G3 — crosscheck**: `distill/crosscheck.py` on that artifact → per-sentence
  cosine ≥ 0.999 AND S-matrix delta ≤ 0.001 vs `model2vec.StaticModel`
  (English + code sentences). Record minima.
- **G4 — existing paths inert**: the five standard feature combos, NIFE,
  potion-multilingual, and the bge (WordPiece) + quant-8 composition — all
  green (run separately; zsh). `git diff src/wordpiece.rs src/unigram.rs`
  shows no behavioral change (const-dispatch only, if any).
- **G5 — rejection still honest**: a BPE model with an unsupported
  pre-tokenizer (or non-null dropout) is rejected naming the component
  (build a /tmp fixture). And a genuinely unsupported family (WordLevel)
  still rejected.
- **Z**: zero warnings (`cargo check`, `--features quant-8`); `cargo check
  --workspace` green; `git status` clean of artifacts.

## STOP conditions

- The vendored BPE involves machinery beyond ordered-merge + ByteLevel
  (e.g. dropout sampling on encode, a cache that changes results) — report
  before implementing.
- G2/G3 cannot reach threshold after two debugging rounds — report
  per-sentence similarities and the diverging tokenization for the worst
  sentence; do NOT loosen the gate.
- `all-distilroberta-v1`'s tokenizer is not ByteLevel BPE as expected —
  report its actual shape; pick another ByteLevel-BPE ST teacher and note it.
- Any G4 regression — the BPE path leaked into WordPiece/Unigram.

## Scope

**In scope**: `build.rs`, `src/bpe.rs` (create), `src/lookup.rs` (merge
table plumbing if needed), `src/lib.rs` (dispatch + doc), `Cargo.toml`/lock
(only if a tiny ByteLevel-alphabet helper dep is unavoidable — prefer
hand-rolling the 256-entry table), `.github/workflows/ci.yml` (one
distilroberta acceptance job), `readme.md` + `docs/rfc-generalization.md`
(BPE now supported).
**Out of scope**: WordLevel/GloVe; cased-model normalizers; quant scheme;
`src/wordpiece.rs`/`src/unigram.rs` behavior; `api-py`.

## Maintenance notes

- After this, supported families = WordPiece + Unigram + BPE. The guard
  lists are the single source of truth; keep them honest (plan 007 standard).
- The merge-table representation is the one novel artifact — document its
  layout next to `scores.bin`/`charsmap.bin` in `src/lookup.rs`.
- Reviewer: verify G1's facts against the vendored source yourself, and that
  the merge application order in `src/bpe.rs` matches `Word::merge_all`
  rank semantics exactly (off-by-one in rank = silent mis-tokenization).
