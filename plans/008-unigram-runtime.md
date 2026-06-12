# Plan 008: Native Unigram support — compile SentencePiece-family static models (layer A, part 1)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving on.
> If anything in "STOP conditions" occurs, stop and report. SKIP updating
> `plans/README.md` — the reviewer maintains the index. Before reporting,
> audit every claim against an actual tool result.
>
> **Drift check (run first)**: HEAD must be `e1476ff` or a descendant that
> has not touched `build.rs` or `src/` since.

## Status

- **Priority**: P1
- **Effort**: L (new runtime algorithm + build-side packing)
- **Risk**: MED (new code path; existing WordPiece path must be untouched)
- **Depends on**: plans 001–007 (all DONE); design decided in
  docs/rfc-generalization.md §4 (spike verdict: native Viterbi, NOT
  greedy-ification)
- **Category**: direction (layer A)
- **Planned at**: commit `e1476ff`, 2026-06-12

## Why this matters

ese currently compiles WordPiece models only. The entire
SentencePiece/Unigram bucket — `potion-multilingual-128M`, the e5/xlm-r
family — is rejected at build time. The RFC's spike measured that
greedy-ifying Unigram models loses too much fidelity (p5 cosine 0.73–0.91),
so the decided path is a native Viterbi segmenter. After this plan, any
Unigram static model whose normalizer chain fits the shapes below compiles
via the existing `ESE_MODEL` interface and passes the same parity gate as
WordPiece models. KISS bar applies (maintainer requirement): plain code,
no general regex engine, no generic normalizer framework — exactly the ops
the target models declare, compiled as data.

## Current state (verified facts, 2026-06-12)

Acceptance target `minishlab/potion-multilingual-128M` at revision
`73908c3438cf03b6a01bcb9611d62b23d0726f08`:

- `model.type` = `"Unigram"`; vocab = array of `[token, score]` pairs,
  **500,353 entries**, log-prob scores (f64 in JSON, store as f32);
  `unk_id: 1` (`[UNK]`), `byte_fallback: false`, no `fuse_unk` key.
  Max token length **186 chars**.
- safetensors: single tensor `embeddings`, F32, shape `[500353, 256]`
  → build with `dim-256`.
- Normalizer chain (Sequence, flattened in order):
  1. `Precompiled` (SentencePiece charsmap; base64 in tokenizer.json,
     ~316 KB encoded)
  2. `Replace{Regex: " {2,}" → " "}`
  3. 33 × `Replace{String: "<single punctuation char>" → " <char> "}`
     (every pattern is exactly one char; content is space-char-space)
  4. `Replace{Regex: "\\s+" → " "}`
  5. `Strip{strip_left: true, strip_right: true}`
- Pre-tokenizer: `Metaspace{replacement: "▁" (U+2581), prepend_scheme:
  "always", split: false}` — i.e. the whole normalized string becomes ONE
  chunk: spaces→▁, prepend ▁ if missing, then the Unigram model segments
  the entire chunk in one pass.
- Post-processor: TemplateProcessing inserting NOTHING (single = `$A`
  only) → `derive_add_special_tokens` correctly yields `false` already.
- model2vec reference behavior (measured in the plan-005 spike): mean-pool
  all token ids the tokenizer returns, then L2-normalize → build with
  `ESE_NORMALIZE_OUTPUT=1`.

Repo state at `e1476ff`:

- `build.rs::guard_tokenizer_family` panics on `model.type != "WordPiece"`.
- `build.rs` vocab parsing assumes `model.vocab` is a `{token: id}` object
  (`.as_object()`), true only for WordPiece.
- `src/lookup.rs`: PHF lookup returns `Option<&'static Param>`; slots are
  `{verify: u64, weight: Param}`; parallel `seeds.bin`/`verify.bin` arrays.
- `src/pretokenizer.rs`: hardcoded BERT-uncased normalization — used by
  the WordPiece path ONLY; do not modify it.
- `src/lib.rs::encode_single`: WordPiece pipeline.
- The parity oracle (`generate_testdata`) uses the `tokenizers` crate and
  therefore already handles Unigram correctly — it is the reference your
  runtime must match.
- Runtime deps today: `unicode-normalization`, `unicode-general-category`,
  optional `rayon`. Build-deps: `serde_json`, `minreq`, `sha2`, optional
  `tokenizers`.

The authoritative semantics for Viterbi/UNK/tie-breaking are the
`tokenizers` crate's own implementation. It is vendored locally once the
`tests` feature has built: find it with
`ls ~/.cargo/registry/src/*/tokenizers-*/src/models/unigram/` — read
`model.rs`, `lattice.rs`, and `trie.rs` BEFORE writing the runtime. Mirror:
how the lattice is populated (char positions, vocab matches), the Viterbi
recurrence and its tie-breaking, the unk handling (note the constant
`K_UNK_PENALTY` and how `unk_score` is computed from the min score), and
`fuse_unk`'s default when absent from JSON. Also read
`normalizers/precompiled.rs` + the `spm_precompiled` crate usage for the
charsmap call pattern, and `pre_tokenizers/metaspace.rs` for
`prepend_scheme: always, split: false` semantics.

## Architecture (decided — do not redesign)

- **Model-kind branch at build time, const-dispatch at runtime.** build.rs
  emits `pub const TOKENIZER_KIND: TokenizerKind = ...;` (a tiny two-variant
  enum in a shared include, or simply `pub const IS_UNIGRAM: bool`).
  `encode_single` branches on the const; the compiler eliminates the dead
  arm. The WordPiece path stays byte-for-byte identical in behavior.
- **Scores as a parallel array**, NOT a slot-layout change:
  `scores.bin` = `[f32; TABLE_SIZE]` indexed by PHF slot (0.0 for empty
  slots), emitted only for Unigram models (empty/absent for WordPiece;
  emit a 4-byte dummy and gate inclusion on the const so `include_bytes!`
  always has a file). `src/lookup.rs` gains
  `pub fn lookup_with_score(key: &str) -> Option<(&'static Param, f32)>`
  (or an internal slot-index lookup both functions share) — keep the
  existing `lookup()` signature unchanged.
- **Normalizer ops compiled as data** in `model_constants.rs` +
  `OUT_DIR` blobs:
  - charsmap bytes: base64-decode in build.rs (hand-roll the ~20-line
    decoder or add `base64` to BUILD-deps only — either fine), write
    `charsmap.bin`, runtime `include_bytes!`s it and feeds
    `spm_precompiled::Precompiled`.
  - punctuation pad table: build verifies every String-replace pattern is
    EXACTLY one char and content == format!(" {} ", pattern); then emits
    `pub const PAD_CHARS: &[char] = &[...]` — runtime applies it as a
    single char-walk pass. If any String replace does not fit that shape →
    build error (STOP shape, see below).
  - whitespace regexes: build accepts ONLY the exact pattern strings
    `" {2,}"` and `"\\s+"` (any other Regex replace → build error) and
    emits two bools; runtime implements both as one "collapse whitespace
    runs to a single space" pass at the right points (note: a
    collapse-spaces pass followed later by collapse-all-whitespace is
    order-sensitive only if the pad table inserts spaces in between —
    apply ops strictly in chain order; the simple loop for `" {2,}"`
    collapses space-runs, the one for `"\\s+"` collapses any-whitespace
    runs).
  - strip flags: two bools → trim.
  - Metaspace: `pub const METASPACE: char = '▁';` + prepend flag. Reject
    `split: true` or `prepend_scheme != "always"` for v1 (build error
    naming the unsupported config).
  - `pub const MAX_TOKEN_CHARS: usize = <max token char count in vocab>;`
- **New runtime module `src/unigram.rs`**: normalize (chain above) →
  metaspace → Viterbi segment → accumulate rows (+ scores from the
  parallel array drive segmentation only; pooling is over weight rows) →
  UNK rows per the tokenizers semantics → mean by token count →
  optional L2 normalize (existing path). Hot-loop note, not cleverness:
  the PHF hash is a per-byte fold, so for a fixed start position extend
  the hash incrementally as the candidate end grows (apply the final
  `h ^ (h >> 32)` mix per candidate) instead of rehashing each substring
  from scratch. Bound candidates by `MAX_TOKEN_CHARS`.
- **Runtime dependency**: add `spm_precompiled = "0.1"` (HF, Apache-2.0,
  single-purpose). Before committing to it run `cargo tree -p ese` and
  confirm it brings no heavy transitive deps (more than ~3 small crates →
  STOP and report the tree). It must be an unconditional dependency
  (model selection is env-driven; a cargo feature would reintroduce the
  mismatch hazard) — for WordPiece builds the const-dispatch makes its
  code dead and the linker drops it.
- **Guards** (`guard_tokenizer_family`): `model.type` may now be
  `"WordPiece"` (existing checks unchanged) or `"Unigram"` (new checks:
  `byte_fallback` must be false/absent, pre-tokenizer must be the
  supported Metaspace shape, normalizer chain must consist only of the op
  shapes listed above — anything else panics naming the component).
  Everything else still rejected naming the type.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| WordPiece regression (all) | `cargo test`; `cargo test --features tests`; `cargo test --features tests,quant-8`; `cargo test --features tests,quant-16`; `cargo test --features tests,rayon,dim-256` | all pass, run as separate commands (zsh does not word-split `${f:+...}` in loops) |
| NIFE regression | the exact command in plans/007 table | all pass |
| Unigram acceptance | `ESE_MODEL=minishlab/potion-multilingual-128M@73908c3438cf03b6a01bcb9611d62b23d0726f08 ESE_MODEL_SHA256=<computed> ESE_TOKENIZER_SHA256=<computed> ESE_NORMALIZE_OUTPUT=1 cargo test --features tests,dim-256` | all pass (parity vs tokenizers-crate oracle) |
| Unsupported-family negative | build a tiny local BPE fixture (Step 6) and `ESE_MODEL_DIR=... cargo check` | clear error naming BPE |
| Workspace | `cargo check --workspace` | exit 0 |

## Scope

**In scope**: `build.rs`, `src/lookup.rs`, `src/lib.rs`, `src/unigram.rs`
(create), `Cargo.toml`/`Cargo.lock` (the `spm_precompiled` dep + optional
`base64` build-dep), `.github/workflows/ci.yml` (one
`potion-multilingual-acceptance` job mirroring the nife job),
`readme.md` (update the WordPiece-only constraint wording).

**Out of scope**: `src/pretokenizer.rs` and `src/wordpiece.rs` (WordPiece
path frozen); BPE/WordLevel support; cased-model support; quantization
changes; greedy-ification; `api-py`; the spike directory.

## Steps

### Step 1: Read the oracle's source

After `cargo build --features tests` (compiles the `tokenizers` crate),
read the vendored Unigram implementation (paths in Current state). Record
in your report: the UNK penalty constant and unk-score formula, the
fuse_unk default for missing JSON key, and the Viterbi tie-breaking rule
(which candidate wins on equal score). These three facts determine parity.

**Verify**: the three facts are written down with file:line references
into your report AND as a comment block at the top of `src/unigram.rs`.

### Step 2: Build-side Unigram path in `build.rs`

Branch on `model.type` after the (updated) guard: parse the vocab array
into `(token, id=index, score)`; the PHF construction is reused unchanged
(entries are token→row-index pairs exactly as for WordPiece); emit
`scores.bin` parallel to the slots; decode and emit `charsmap.bin`; derive
and emit the normalizer-op constants, Metaspace consts, `MAX_TOKEN_CHARS`,
`UNK` row from `unk_id` (Unigram models have no CLS/SEP —
`ADD_SPECIAL_TOKENS` derivation already handles this), and the
`TOKENIZER_KIND`/`IS_UNIGRAM` const. WordPiece builds must emit identical
constants to before plus the new const(s) and a dummy `scores.bin`/
`charsmap.bin`.

**Verify**: all five WordPiece regression commands pass (byte-identical
behavior); `ESE_MODEL=...potion-multilingual... cargo build --features dim-256`
completes (runtime not implemented yet — gate only the build script paths;
if lib compilation fails at this stage that is expected, note it and
continue to Step 3).

### Step 3: Runtime — `src/unigram.rs` + lookup extension

Implement per the Architecture section. `encode_single` gains the const
branch:

```rust
if lookup::IS_UNIGRAM {
    return unigram::encode_single(text /* , &mut buffers as needed */);
}
```

(or an equivalent shape that keeps the WordPiece body untouched below it).
Keep `unigram.rs` plain: one normalize function applying the compiled op
chain in order, one metaspace function, one viterbi function with the DP
table as two flat `Vec`s (best_score, best_backpointer), one pooling loop.
Comment the Viterbi with the three facts from Step 1.

**Verify**: the Unigram acceptance command passes — parity ≥ 0.999999
(f32) against the tokenizers-crate oracle on all 11 test inputs, including
CJK and empty-string cases. If specific inputs fail, debug by printing the
oracle's token ids from `generate_testdata` (temporary `cargo:warning`
lines) vs your segmentation — do not loosen thresholds.

### Step 4: Regression sweep

Run every command in the table: five WordPiece combos, NIFE, workspace.

**Verify**: all pass; `git diff src/pretokenizer.rs src/wordpiece.rs` → empty.

### Step 5: Guards, CI, readme

Update `guard_tokenizer_family` + its doc comment (Unigram accepted shapes
listed); add the `potion-multilingual-acceptance` CI job (ubuntu, env vars
incl. both SHA-256s you computed, `--features tests,dim-256`, cache key
`hf-model-potion-ml-73908c3`); readme: "WordPiece models only" →
"WordPiece and Unigram (SentencePiece) models" with the supported-shape
caveats, and update the worked-examples section with the potion command.

**Verify**: YAML parses; `grep -c "Unigram" readme.md` ≥ 1.

### Step 6: Negative fixture — BPE still rejected clearly

Create a throwaway local fixture (in `/tmp`, not the repo): a minimal
`tokenizer.json` with `model.type: "BPE"` (vocab `{"a":0,"b":1}`, merges
`[]`) plus a tiny valid single-tensor safetensors (2×4 F32 — write with a
10-line python script). `ESE_MODEL_DIR=/tmp/<fixture> cargo check` →
build error naming BPE and pointing at the RFC.

**Verify**: the error message captured verbatim in your report.

## Done criteria

- [ ] Unigram acceptance command passes (potion-multilingual compiles +
      parity green via env config only)
- [ ] All five WordPiece feature combos + NIFE acceptance still pass
- [ ] `src/pretokenizer.rs` and `src/wordpiece.rs` unchanged (`git diff` empty)
- [ ] BPE fixture rejected with a clear message
- [ ] `cargo tree` shows `spm_precompiled` with an acceptably small
      footprint (recorded in report)
- [ ] CI has the new acceptance job; YAML parses
- [ ] `cargo check --workspace` exits 0
- [ ] No compiler warnings introduced (`cargo check 2>&1 | grep warning` clean)
- [ ] No files outside scope modified

## STOP conditions

- The tokenizers crate's Unigram semantics involve machinery beyond
  scores+Viterbi+unk-penalty (e.g. n-best sampling on the encode path, a
  cache layer that changes results) — report what you found before
  implementing.
- `spm_precompiled` pulls heavy transitive deps (see Architecture) or its
  API cannot consume the decoded charsmap bytes directly.
- Any normalizer op in the target model's chain does not fit the compiled
  shapes (single-char pad replaces, the two exact whitespace regexes,
  strip) — the shape list is wrong; report the op verbatim.
- Parity fails on specific inputs after two debugging rounds — report the
  per-input similarities and the diverging segmentations; do not ship a
  loosened threshold.
- Any WordPiece regression command fails — the shared-path change is
  wrong; the WordPiece path must be provably untouched.

## Maintenance notes

- This is layer A part 1. Part 2 candidates (separate plans): cased
  models, BPE (needs merges table + ByteLevel pretok), `split: true`
  Metaspace, byte_fallback vocabs. The guard's accepted-shapes lists are
  the single source of truth for "what ese compiles" — keep them honest
  (plan 007's standard).
- multilingual-e5 ALSO needs Unigram + Precompiled — it should compile
  after this plan IF its other chain ops fit the shapes; worth a bonus
  check but it is a transformer repo (multi-tensor safetensors) so it will
  fail at parse_safetensors — that is correct behavior (it needs
  distillation first, layer C).
- The incremental-hash Viterbi inner loop is the performance-relevant
  part; if encode throughput for Unigram models matters later, benchmark
  against `model2vec-rs` (already a dev-dep) before optimizing further.
- Reviewer: scrutinize Step 1's three facts against the vendored source
  yourself, and the charsmap byte-equality (decoded bytes' SHA vs a
  reference decode).
