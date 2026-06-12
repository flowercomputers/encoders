# Plan 006: Manifest-driven model ingestion — select the compiled model with env config instead of editing build.rs

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 5155c13..HEAD -- build.rs src/lib.rs src/test/mod.rs`
> Plans 001–004 are expected to have landed (pinned URLs + checksums,
> special tokens by name, quant-aware thresholds). This plan builds directly
> on those. If plan 003 or 004's changes are absent from `build.rs`, STOP —
> execute those first.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (touches build-script codegen consumed by all builds)
- **Depends on**: plans/003-pin-and-checksum-model-downloads.md,
  plans/004-small-correctness-batch.md; design context in
  plans/005-generalization-roadmap.md (layer B)
- **Category**: direction
- **Planned at**: commit `5155c13`, 2026-06-12

## Why this matters

Today, compiling a different embedding model means editing URL constants in
`build.rs`. The repo's one attempt at a second model
(`origin/codex/nife-mxbai-support`) did it with a per-model Cargo feature —
a dead end, because Cargo features must be additive: two crates in one build
graph enabling different `model-*` features would unify and fight, and the
feature list grows with every model. This plan makes the model a **build
configuration**: `ESE_MODEL`/`ESE_MODEL_DIR` env vars with pinned-revision
checksums, while the current model stays the zero-config default. After
this lands, any WordPiece-vocab static embedding model on Hugging Face (or
any local artifact pair, e.g. produced by a future distillation tool)
compiles without source edits — and models the runtime can't yet tokenize
fail the build with a clear message instead of producing silently wrong
embeddings.

## Current state

(Line numbers from `5155c13`; plans 001/003/004 will have shifted them —
match on content.)

- `build.rs` top: pinned `MODEL_URL`/`TOKENIZER_URL` constants + SHA-256
  constants (shape post-plan-003):

  ```rust
  static MODEL_URL: &'static str = "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/<REV>/0_StaticEmbedding/model.safetensors";
  static TOKENIZER_URL: &'static str = "...";
  static MODEL_SHA256: &str = "...";
  static TOKENIZER_SHA256: &str = "...";
  ```

- `build.rs` flow: `shared_model_dir()` → `download_if_missing(path, url, sha)`
  → `parse_safetensors()` (handles F16/BF16/F32/F64, truncates dims via
  `trunc_dims()`) → vocab from `tokenizer.json` `model.vocab` as a
  `{token: id}` object → special tokens by name (post-plan-004) → PHF
  construction → emits `weights.bin`, `seeds.bin`, `verify.bin`,
  `model_constants.rs` (DIMENSIONS, TABLE_*, UNK/CLS/SEP, QUANT_*) →
  optional `generate_testdata()` using the `tokenizers` crate as oracle.
- `src/lib.rs::encode_single` unconditionally adds CLS and SEP and counts
  them in the mean (post-plan-004): semantics `(CLS + SEP + Σ)/(n+2)`.
- `src/pretokenizer.rs` hardcodes BERT-**uncased** normalization (lowercase,
  NFD, strip marks, CJK isolation). This plan does NOT change that — it
  **guards** it (Step 3): models whose tokenizer config doesn't match get a
  build error. Full normalizer generality is layer A
  (see `plans/005-generalization-roadmap.md`).
- Reference branch: `git show origin/codex/nife-mxbai-support:build.rs` —
  contains the `ModelProfile` struct, per-model `cache_key` subdirectory,
  `add_special_tokens`/`normalize_output` flags, and an output-normalization
  step in `lib.rs`. This plan reuses those ideas with env-var selection
  instead of Cargo features.
- Useful fact: `tokenizer.json` declares everything needed for the guard:
  `model.type` (`"WordPiece"` / `"Unigram"` / `"BPE"`),
  `normalizer` (e.g. `{"type":"BertNormalizer","lowercase":true,...}`),
  `model.continuing_subword_prefix` (the `"##"`), and a `post_processor`
  that names CLS/SEP if the model adds them.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Default build + parity | `cargo test --features tests` | all pass |
| Quantized parity | `cargo test --features tests,quant-8` | all pass |
| Alt-model build (Step 5) | `ESE_MODEL=... ESE_MODEL_SHA256=... ESE_TOKENIZER_SHA256=... cargo test --features tests,dim-1024` | all pass |
| Negative test (Step 6) | `ESE_MODEL=minishlab/potion-base-8M@main cargo check 2>&1 \| grep -i unigram` | clear build error naming Unigram |
| Resolve a HF revision | `curl -s https://huggingface.co/api/models/<repo> \| python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"` | 40-char sha |

## Scope

**In scope** (the only files you should modify):
- `build.rs`
- `src/lib.rs` (optional output-normalization + `MODEL_NAME` const)
- `src/test/mod.rs` (only if assertions need the model name in messages)
- `readme.md` ("Using a different model" section)
- `plans/README.md` (status row)

**Out of scope** (do NOT touch):
- `src/pretokenizer.rs`, `src/wordpiece.rs` — no tokenizer-family work
  (layer A, future plan).
- `Cargo.toml` features — do NOT add per-model features; do NOT remove the
  existing `dim-*`/`quant-*` features (deprecating them is a later
  decision).
- `api-py/` — wheel-side model selection comes with layer D.
- The PHF construction and binary emit format — unchanged.

## Git workflow

- Branch: `advisor/006-manifest-model-ingestion`
- Commit per step, short lowercase imperative (e.g. `select model via
  ese_model env var`, `guard unsupported tokenizer families at build time`).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Introduce a resolved-model struct and env-var resolution

In `build.rs`, add:

```rust
struct ResolvedModel {
    /// display name, e.g. "sentence-transformers/static-retrieval-mrl-en-v1@<rev>"
    name: String,
    /// cache subdirectory key (sanitized name: '/' and '@' -> '-')
    cache_key: String,
    /// where artifacts come from
    source: ModelSource,
}
enum ModelSource {
    Remote { model_url: String, tokenizer_url: String,
             model_sha256: Option<String>, tokenizer_sha256: Option<String> },
    LocalDir(PathBuf),
}
```

Resolution order in a `resolve_model()` fn:
1. `ESE_MODEL_DIR` set → `LocalDir` (must contain `model.safetensors` +
   `tokenizer.json`; no download, no checksum required).
2. `ESE_MODEL` set → must be `<hf-repo>@<revision>` (reject bare repo ids
   with a panic explaining that a pinned revision is required —
   reproducibility is non-negotiable, see plan 003). URLs:
   `https://huggingface.co/<repo>/resolve/<rev>/model.safetensors` and
   `.../tokenizer.json`. Optional `ESE_MODEL_SHA256` /
   `ESE_TOKENIZER_SHA256` env vars feed the existing verification from plan
   003; when unset, print `cargo:warning=` that integrity is unverified for
   a custom model. **Subpath escape hatch**: `ESE_MODEL_FILE` /
   `ESE_TOKENIZER_FILE` may override the two file subpaths (the default
   model needs `0_StaticEmbedding/model.safetensors`).
3. Neither set → the built-in default: current pinned constants + their
   checksums (exactly today's behavior).

Emit `cargo:rerun-if-env-changed=` for every env var read
(`ESE_MODEL`, `ESE_MODEL_DIR`, `ESE_MODEL_SHA256`, `ESE_TOKENIZER_SHA256`,
`ESE_MODEL_FILE`, `ESE_TOKENIZER_FILE`).

Cache layout: `shared_model_dir(...)` gains the `cache_key` subdirectory
(port from the NIFE branch) so models don't overwrite each other.

**Verify**: `cargo test --features tests` → all pass (default path,
behavior unchanged); `ls <cache-root>` shows a per-model subdirectory.

### Step 2: Derive model facts from the artifacts instead of constants

Replace assumptions with derivation, all in `build.rs`:

- `DIMENSIONS`: already derived from the safetensors shape + `trunc_dims()`
  — keep, but emit a build error if the safetensors has more than one
  non-metadata tensor (ambiguous file) instead of silently picking the
  first key.
- Special tokens: post-plan-004 they're looked up by `"[UNK]"`/`"[CLS]"`/
  `"[SEP]"` names. Generalize: read the tokenizer's `post_processor`
  section. If it references CLS/SEP (TemplateProcessing/BertProcessing) →
  `ADD_SPECIAL_TOKENS = true` and the named tokens must exist (panic
  otherwise). If there is no post_processor or it adds no special tokens →
  `ADD_SPECIAL_TOKENS = false` and CLS/SEP are emitted as zero rows. UNK:
  from `model.unk_token` if present, else `"[UNK]"`, panic if missing.
- WordPiece continuation prefix: read
  `model.continuing_subword_prefix` (default `"##"`) and emit it as
  `pub const CONTINUING_PREFIX: &str = ...` in `model_constants.rs`; change
  `src/wordpiece.rs` to use `lookup::CONTINUING_PREFIX` instead of the
  literal `"##"` (two `push_str` call sites — this is a constant swap, not
  tokenizer-family work). Same for `MAX_WORD_LEN` from
  `model.max_input_chars_per_word` (default 100).
- Output normalization: env `ESE_NORMALIZE_OUTPUT=1` → emit
  `NORMALIZE_OUTPUT: bool` and add the L2-normalize step at the end of
  `encode_single` (port the `normalize()` fn from
  `git show origin/codex/nife-mxbai-support:src/lib.rs`). Reference
  testdata generation must apply the same flag (port from the branch's
  `generate_testdata`).
- Emit `pub const MODEL_NAME: &str = "<name>"` and re-export it from
  `src/lib.rs` (port from the branch, including its doctest).
- `ADD_SPECIAL_TOKENS` in the runtime: gate the CLS/SEP accumulation in
  `encode_single` on `lookup::ADD_SPECIAL_TOKENS` (port from the branch —
  but keep plan 004's counting semantics: when true, token_count starts
  at 2; when false, it starts at 0 and the `token_count != 0` guard
  returns; mirror exactly what `generate_testdata`'s oracle computes with
  `tokenizer.encode(input, add_special_tokens)`).

**Verify**: `cargo test --features tests` and
`cargo test --features tests,quant-8` → all pass (the default model derives
to exactly the previous constants: ADD_SPECIAL_TOKENS=true, prefix `##`,
MAX_WORD_LEN=100). `grep -n '"##"' src/wordpiece.rs` → 0 matches.

### Step 3: Guard unsupported tokenizer families and normalizers

In `build.rs`, after loading `tokenizer.json`:

- `model.type != "WordPiece"` → `panic!` with:
  `model '<name>' uses a '<type>' tokenizer; ese currently compiles WordPiece models only (Unigram support is planned — see docs/rfc-generalization.md)`.
- Normalizer check: accept `BertNormalizer` with `lowercase: true` (and
  null/absent `strip_accents`, which BERT treats as "follow lowercase"), or
  a `Sequence` equivalent, or normalizers the hardcoded runtime already
  matches. Anything else (e.g. `lowercase: false` — a **cased** model) →
  panic naming the unsupported component. Err on the side of rejecting:
  a false build error costs minutes; silently-wrong embeddings cost trust.
  Keep the accepted-shapes list in one function with a comment pointing at
  `src/pretokenizer.rs` as the thing it mirrors.

**Verify**: `ESE_MODEL=minishlab/potion-base-8M@main cargo check 2>&1 | grep -i "unigram"`
→ the clear error message (resolve `main` to a sha first if Step 1's
pinning rejects bare `@main`; in that case use the resolved revision — the
point is the Unigram rejection, exercise it however the interface allows).

### Step 4: Document it

Add a "Using a different model" section to `readme.md`: the env vars, the
pinned-revision requirement, checksum vars, `ESE_MODEL_DIR` for local
artifacts, the WordPiece-only constraint (with one sentence on the roadmap),
and one full worked example (the Step 5 model).

**Verify**: `grep -c "ESE_MODEL" readme.md` → ≥ 2.

### Step 5: Acceptance — compile a second real model via env config only

Reproduce what the NIFE branch did, with zero source edits:

1. Resolve revision: the branch pinned
   `stephantulkens/NIFE-mxbai-embed-large-v1` at
   `970a0151f3faeddbab5c68be38d52bbc5ea719ab` — use that.
2. Download both files at that revision, compute SHA-256s (`shasum -a 256`).
3. Run:

   ```sh
   ESE_MODEL=stephantulkens/NIFE-mxbai-embed-large-v1@970a0151f3faeddbab5c68be38d52bbc5ea719ab \
   ESE_MODEL_SHA256=<digest> ESE_TOKENIZER_SHA256=<digest> \
   ESE_NORMALIZE_OUTPUT=1 \
   cargo test --features tests,dim-1024
   ```

   Expectations: this model's tokenizer has no CLS/SEP post-processing
   (the branch set `add_special_tokens: false`), is WordPiece, 1024-dim.
   The build-time oracle (`generate_testdata`) recomputes reference outputs
   for THIS model, so the parity test is meaningful automatically.

**Verify**: the command above → all tests pass. Then re-run the default
build (`cargo test --features tests`) → still passes (cache isolation
works; no cross-contamination).

### Step 6: Negative-path checks

- Unpinned model rejected: `ESE_MODEL=foo/bar cargo check` → build error
  demanding `@<revision>`.
- Missing local artifacts: `ESE_MODEL_DIR=/tmp/empty cargo check` → clear
  error listing the two required filenames.

**Verify**: both commands fail with the described messages (capture them in
your report).

## Test plan

- The existing parity machinery is intrinsically multi-model (oracle
  regenerates per compiled model) — Steps 5/6 are the new coverage, run
  manually and recorded in the report.
- Add one CI job to `.github/workflows/ci.yml` building the Step 5 model
  (env vars in the job; cache key must include the model name+rev — see
  plan 002's maintenance note). If CI does not exist yet (plan 002
  unexecuted), skip and note it.
- Full local gate: `cargo test && cargo test --features tests && cargo test --features tests,quant-8` → all pass.

## Done criteria

- [ ] Default build unchanged: `cargo test --features tests` and
      `cargo test --features tests,quant-8` exit 0 with no env vars set
- [ ] Step 5's NIFE-mxbai command exits 0 (second model compiles and passes
      parity via env config only)
- [ ] `ESE_MODEL` with a Unigram model fails the build naming the tokenizer
      type (Step 3 verify)
- [ ] Unpinned `ESE_MODEL=foo/bar` is rejected (Step 6)
- [ ] `grep -n '"##"' src/wordpiece.rs` → 0 (prefix now model-derived)
- [ ] `grep -c "ESE_MODEL" readme.md` → ≥ 2
- [ ] No per-model Cargo feature added: `grep -c "model-" Cargo.toml` → 0
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Plans 003/004 are not in `build.rs` (drift check) — execute those first.
- Step 5's parity test fails for NIFE-mxbai after the flags are derived
  correctly — the model needs runtime behavior beyond
  `ADD_SPECIAL_TOKENS`/`NORMALIZE_OUTPUT` (report the failing inputs and
  similarities; the branch's profile may have encoded an assumption this
  plan missed).
- The default model's derivation produces different constants than before
  (e.g. post_processor parsing concludes ADD_SPECIAL_TOKENS=false) — the
  derivation logic is wrong; do not "fix" it by hardcoding.
- You need to modify `src/pretokenizer.rs` beyond reading new constants —
  that's layer A scope; stop and flag.

## Maintenance notes

- This establishes the model-selection interface (`ESE_MODEL[@rev]`,
  `ESE_MODEL_DIR`, checksum vars). Layer A (Unigram) and layer C
  (distillation) plug into it: a distilled local model is just
  `ESE_MODEL_DIR=...`. Don't add a second interface.
- The Step 3 normalizer guard mirrors `src/pretokenizer.rs` by hand — when
  layer A makes normalization config-driven, delete the guard in the same
  change.
- After this lands, `origin/codex/nife-mxbai-support` is fully superseded —
  the operator can delete the branch (its readme commit landed via plan 001).
- Reviewer: scrutinize Step 2's ADD_SPECIAL_TOKENS derivation (the
  post_processor JSON shapes vary: TemplateProcessing vs BertProcessing) and
  the cache-key sanitization (no `/` in directory names).
