# Plan 020: Extract a compiler IR and dedupe build.rs so new model families stop being monolith surgery

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 8a4616c..HEAD -- build.rs`
> This plan restructures `build.rs`. If it changed since this plan was written
> — **especially if plan 018 landed** (it adds validation guards you must
> preserve) — read the current `build.rs` fully and re-map the step targets
> before editing. A drift here is expected if 014 merged; reconcile, don't
> assume the line numbers below.
>
> **Step 0 (dispatch only)**: agent worktrees are branched from `master`, NOT
> from `advisor/improvements`. Before anything else run:
> `git reset --hard advisor/improvements`. Safe — the worktree branch is fresh.

## Status

- **Priority**: P2
- **Effort**: L
- **Risk**: MED — large refactor of build-time code. Mitigated entirely by the
  golden-vector parity gate (`--features tests`) and the two CI acceptance
  models: this is a **behavior-preserving** refactor, so those gates must stay
  green at every step, byte-for-byte.
- **Depends on**: plan 018 (its build-time guards must be carried into the new
  IR frontend, not dropped). If 014 has not landed, you may still do this plan,
  but coordinate so 014's checks land in the refactored structure.
- **Category**: tech-debt / architecture
- **Planned at**: commit `8a4616c`, 2026-06-13

## Why this matters

The RFC (`docs/rfc-generalization.md` §1–2) frames `ese` as a clean five-stage
compiler over an IR triple `(TokenizerSpec, EmbeddingMatrix, PoolingSpec)`. The
**code does not match this** — and that mismatch is the structural reason the
generalization roadmap has stalled at two tokenizer families and zero of the
eight optimization passes (E1–E8) the RFC describes.

Concretely, in `build.rs` (1761 lines) the WordPiece and Unigram paths are two
parallel inline implementations that have copy-pasted:

- the **PHF builder** (`build.rs:328-383` ≈ `build.rs:658-708`, identical
  algorithm),
- the **weights.bin emitter** (`build.rs:402-450` ≈ `build.rs:723-760`,
  near-identical per-row-quant slot packing),
- the **model_constants.rs codegen** (`build.rs:502-559` and `build.rs:829`+),
  which emit the same ~30 consts with different values.

There is **no `TokenizerSpec` or IR struct** — model facts live as ~25 loose
local variables threaded through `main`/`build_unigram` by hand. Adding a third
family (cased WordPiece, WordLevel/GloVe, BPE) today means a third copy of all
three blocks. Adding an optimization pass is impossible because there is no IR
for a pass to transform.

This plan introduces the missing seams **without changing any output**: one PHF
builder, one slot emitter, one codegen function over a populated IR struct. After
it lands, the next family is "populate the IR differently," not "fork the
monolith," and the E-pass layer has a data structure to operate on.

**This plan adds no new capability and changes no embedding.** Its success
criterion is that everything produces byte-identical artifacts and every parity
gate stays green — while the line count of duplicated logic drops sharply.

## Current state

`build.rs` structure:
- `resolve_model()` (`build.rs:47`) → `ResolvedModel { name, cache_key, source }`.
- `main()` (`build.rs:156`) → resolves, downloads, `parse_safetensors`, guards
  the tokenizer family, then **branches** on `model_type`:
  `if model_type == "Unigram" { build_unigram(...); return; }` (`build.rs:215`),
  else the WordPiece path runs inline in `main` (`build.rs:235-577`).
- `build_unigram()` (`build.rs:585`) — the parallel Unigram path.
- Shared helpers below: `phf_hash`, `row_quant_params`, `quantize_f32_to_bytes`,
  `format_quantized_param`, `dtype`, `dtype_sz`, `parse_safetensors`,
  `download_if_missing`, `guard_tokenizer_family`, `check_normalizer`,
  `guard_unigram_shapes`, `collect_unigram_norm_ops`, `cross_check_special_token_ids`,
  `special_token_id`, `generate_testdata`.

**The duplicated PHF builder.** WordPiece, `build.rs:322-383`:

```rust
    let vocab_size = entries.len();
    let table_size = vocab_size.next_power_of_two();
    let table_mask = table_size - 1;
    let num_buckets = vocab_size;
    let mut buckets: Vec<Vec<usize>> = vec![vec![]; num_buckets];
    for (i, (token, _)) in entries.iter().enumerate() {
        let b = (phf_hash(token.as_bytes(), 0) as usize) % num_buckets;
        buckets[b].push(i);
    }
    let mut bucket_order: Vec<usize> = (0..num_buckets).collect();
    bucket_order.sort_by(|a, b| buckets[*b].len().cmp(&buckets[*a].len()));
    let mut seeds: Vec<u32> = vec![0; num_buckets];
    let mut verify: Vec<u64> = vec![0; table_size];
    let mut slot_to_weight: Vec<usize> = vec![0; table_size];
    let mut occupied: Vec<bool> = vec![false; table_size];
    for &bi in &bucket_order { /* seed search ... */ }
```

Unigram, `build.rs:658-708`, is the same code (note one cosmetic difference: the
WordPiece version's stuck-assert message includes the entry count; the Unigram
one doesn't — `build.rs:355-359` vs `build.rs:686`). Both produce
`(seeds, verify, slot_to_weight, occupied, table_size, table_mask, num_buckets)`.

**The duplicated slot emitter.** WordPiece `build.rs:423-450`, Unigram
`build.rs:738-760` — both loop `0..table_size`, write `verify[slot]`, pick the row
(`weights[slot_to_weight[slot]]` or a zero row), and in quant builds write
`(row_min, row_scale)` then the quantized weights via `row_quant_params` +
`quantize_f32_to_bytes`. The only differences are local variable names
(`quant_max` vs `quant_max_uni`, `slot_size` vs `slot_size_uni`).

**The duplicated codegen.** WordPiece `build.rs:503-559`, Unigram
`build.rs:829`+ — both `fs::write(out.join("model_constants.rs"), format!(...))`
emitting `DIMENSIONS`, `Param`, `TABLE_SIZE`, `TABLE_MASK`, `NUM_BUCKETS`,
`SLOT_SIZE`, `UNK`/`CLS`/`SEP` + their `_MIN`/`_SCALE`, `MAX_WORD_LEN`,
`CONTINUING_PREFIX`, `ADD_SPECIAL_TOKENS`, `NORMALIZE_OUTPUT`, `SKIP_UNK`,
`DTYPE_SIZE`, `QUANT_MIN`/`QUANT_SCALE`, `MODEL_NAME`, and the Unigram block
(`IS_UNIGRAM`, `MIN_SCORE`, `MAX_TOKEN_CHARS`, `UNK_ID`, `PAD_CHARS`,
`COLLAPSE_*`, `STRIP_*`, `METASPACE_*`). The WordPiece block hardcodes the
Unigram consts to inert defaults (`build.rs:540-554`); the Unigram block sets them
to real values.

Repo conventions:
- Zero-dependency build script (no new crates). Hand-rolled is deliberate.
- The emitted `model_constants.rs` must remain **exactly** what `src/lookup.rs`
  (the `Slot` struct, `Param`, the consts) expects — the runtime is out of scope
  and must not need a single edit.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Drift check | `git diff --stat 8a4616c..HEAD -- build.rs` | empty, or reconcile if 018 landed |
| Workspace builds | `cargo check --workspace` | exit 0 |
| Default tests | `cargo test` | all pass |
| Parity (WordPiece default) | `cargo test --features tests` | all pass |
| Parity quant-8 | `cargo test --features tests,quant-8` | all pass |
| Parity quant-16 | `cargo test --features tests,quant-16` | all pass |
| WordPiece acceptance | `ESE_MODEL=stephantulkens/NIFE-mxbai-embed-large-v1@970a0151f3faeddbab5c68be38d52bbc5ea719ab ESE_NORMALIZE_OUTPUT=1 cargo test --features tests,dim-1024` | all pass |
| Unigram acceptance | potion-multilingual job from `.github/workflows/ci.yml` (dim-256) | all pass |
| Byte-identical artifact check | see step 1 (compare `weights.bin`/`seeds.bin`/`model_constants.rs` before vs after) | identical |

## Suggested executor toolkit

- Before starting, read `.github/workflows/ci.yml` to get the exact pinned SHAs
  and feature sets for the two acceptance models — they are your regression net.

## Scope

**In scope** (the only file you should modify):
- `build.rs`

**Out of scope** (do NOT touch — if you think you need to, STOP):
- Anything under `src/`. The runtime, `src/lookup.rs`'s `Slot`/`Param`/consts
  contract, and the emitted-constant names/types must be unchanged.
- The slot binary layout, quantization math, PHF hash function, normalizer
  derivation, or special-token logic. You are **moving** code, not changing what
  it computes.
- `distill/`, benches, docs (beyond a comment).
- Do NOT "improve" the algorithms while refactoring (no "while I'm here" changes
  to quantization, hashing, etc.). Behavior-preserving only.

## Git workflow

- Work on `advisor/improvements`.
- **Commit after each step** (each is independently green) — small commits make
  a parity regression bisectable. Lowercase imperative messages, e.g.
  `extract build_phf helper; dedupe wordpiece/unigram phf`,
  `extract emit_slots_bin shared by both paths`,
  `introduce ModelIr struct; single codegen emitter`.
- Do NOT push or open a PR unless instructed.

## Steps

> Order is deliberate: dedupe the **mechanical** blocks first (PHF, slot
> emitter) where byte-identical output is easy to assert, then introduce the IR
> struct and unify codegen last. Each step keeps the build green.

### Step 0: Capture a golden artifact baseline

Before changing anything, build the default model with tests and snapshot the
generated artifacts so you can prove byte-identity after each step:

```sh
cargo build --features tests
# locate OUT_DIR (printed in build output or under target/debug/build/ese-*/out/)
cp "$(find target -path '*/out/model_constants.rs' | head -1)" /tmp/golden_model_constants.rs
cp "$(find target -path '*/out/weights.bin' | head -1)" /tmp/golden_weights.bin
cp "$(find target -path '*/out/seeds.bin' | head -1)" /tmp/golden_seeds.bin
```

Repeat for `--features tests,quant-8` into a separate `/tmp/golden_*_q8` set
(the quant build exercises the per-row-param slot packing).

**Verify**: the golden files exist and are non-empty.

### Step 1: Extract one `build_phf` helper; use it in both paths

Introduce a struct and function near the other helpers:

```rust
struct PhfTable {
    seeds: Vec<u32>,
    verify: Vec<u64>,
    slot_to_weight: Vec<usize>,
    occupied: Vec<bool>,
    table_size: usize,
    table_mask: usize,
    num_buckets: usize,
}

/// Build a CHD/PHF over the vocab keys. Behavior-identical to the two former
/// inline copies (WordPiece build.rs:322-383 and Unigram build.rs:658-708).
fn build_phf(entries: &[(String, usize)]) -> PhfTable { /* moved verbatim */ }
```

Replace both inline blocks with `let table = build_phf(&entries);` and update the
downstream references (`table.seeds`, `table.verify`, etc.). Keep the stuck-assert
message that names the entry count (the WordPiece variant) — it is strictly more
informative; using it in both is an acceptable, non-behavioral message change.

**Verify** (byte-identity is the gate):
- `cargo build --features tests`, then `diff` the new `weights.bin`/`seeds.bin`
  against `/tmp/golden_weights.bin`/`/tmp/golden_seeds.bin` → **no differences**.
- Same diff for the quant-8 golden set after `cargo build --features tests,quant-8`.
- `cargo test --features tests` and `cargo test --features tests,quant-8` → pass.

### Step 2: Extract one slot-emitter helper

Move the slot-packing loop (WordPiece `build.rs:423-450`, Unigram
`build.rs:738-760`) into:

```rust
/// Pack the PHF slots into weights.bin bytes. Behavior-identical to the two
/// former inline copies. Handles f32 and quant-8/quant-16 layouts.
fn build_weights_bin(table: &PhfTable, weights: &[Vec<f32>], zero_row: &[f32],
                     dims: usize, quant_max: f32) -> Vec<u8> { /* moved verbatim */ }
```

Have both paths call it and then `fs::write(out.join("weights.bin"), &bytes)`.
Keep the `slot_size`/`SLOT_SIZE` computation consistent (it already matches
between paths).

**Verify**: same byte-identity diffs as step 1 (f32 + quant-8) → no differences;
parity tests pass.

### Step 3: Introduce a `ModelIr` struct holding every codegen field

Define a struct that captures **all** the values both codegen blocks emit:

```rust
struct ModelIr {
    // matrix / table
    dims: usize,
    dtype: &'static str,          // "f32" | "u8" | "u16"
    dtype_sz: usize,
    table: PhfTable,
    // specials
    unk_str: String, cls_str: String, sep_str: String,
    unk_min: f32, unk_scale: f32, cls_min: f32, cls_scale: f32, sep_min: f32, sep_scale: f32,
    // wordpiece facts
    max_word_len: usize, continuing_prefix: String, add_special_tokens: bool,
    unk_id: usize,
    // global compat params
    global_min: f32, quant_scale: f32,
    // shared flags
    normalize_output: bool, skip_unk: bool,
    model_name: String,
    // unigram facts (inert defaults for WordPiece)
    is_unigram: bool, min_score: f32, max_token_chars: usize,
    pad_chars: Vec<char>, collapse_spaces: bool, collapse_whitespace: bool,
    strip_left: bool, strip_right: bool,
    metaspace_prepend: bool, metaspace_split_whitespace: bool,
}
```

Populate it in the WordPiece path with the values currently passed to `format!`
(`build.rs:503-557`), using the inert Unigram defaults exactly as the WordPiece
codegen does today (`is_unigram: false`, `min_score: 0.0`, `max_token_chars: 0`,
empty `pad_chars`, all `collapse_*`/`strip_*` false, `metaspace_*` false).

Do **not** wire codegen to it yet — just build the struct alongside the existing
`format!` and assert nothing changed (the existing emit still runs).

**Verify**: `cargo build --features tests` still emits the golden
`model_constants.rs` (unchanged path) → `diff` against golden → no differences.

### Step 4: Single `emit_model_constants(out, &ir)` function

Move the `format!` template (it is the same template in both paths, with the
WordPiece one hardcoding Unigram defaults) into:

```rust
fn emit_model_constants(out: &Path, ir: &ModelIr) {
    fs::write(out.join("model_constants.rs"), format!(/* the existing template */)).unwrap();
}
```

Switch the WordPiece path to call it. Then populate a `ModelIr` in the Unigram
path too (with real Unigram values from `build_unigram`) and switch it to the
same function. Delete both old inline `format!` blocks.

**Critical**: the emitted text must be byte-identical to the originals for both
families. The template has subtle bits — `METASPACE_CHAR: char = '\u{2581}'`,
`#[allow(dead_code)]` attributes, the `:?` debug-formatting of floats. Reproduce
them exactly. The golden diff is your proof.

**Verify**:
- WordPiece default: `model_constants.rs` diff vs `/tmp/golden_model_constants.rs`
  → **no differences** (f32 and quant-8).
- Unigram: build the potion-multilingual acceptance model with `--features tests`
  and confirm `cargo test --features tests` passes (snapshot its
  `model_constants.rs` before this step if you want a Unigram golden too — recommended).
- All four parity commands pass; both acceptance models pass.

### Step 5: Collapse the `main`/`build_unigram` skeletons to the shared shape

With PHF, slot emit, and codegen unified, the two paths now differ only in
**frontend** work: WordPiece parses a `vocab` object + derives WordPiece facts;
Unigram parses a `[token, score]` array + walks the normalizer chain. Tighten the
two functions so they read as: *frontend → build IR → (build_phf, build_weights_bin,
emit_model_constants, emit scores/charsmap) → testdata*. Do not over-engineer a
trait; a shared tail sequence operating on `ModelIr` is enough. Leave the
family-specific frontend code in place.

If plan 018 landed, ensure its validation guards now live in the **frontend /
IR-construction** region (e.g. the finite scan and vocab/row check as part of
building the `EmbeddingMatrix`/IR), per 014's maintenance note — relocate, don't
duplicate.

**Verify**: full gate — all four parity commands + both acceptance models pass;
golden diffs for default (f32 + quant-8) still clean.

### Step 6: Confirm the duplication is actually gone

The point of the plan. Confirm the PHF algorithm, slot loop, and codegen template
each appear **once**:

```sh
grep -c "best_path\|next_power_of_two" build.rs   # PHF table-size logic: expect 1 occurrence of next_power_of_two
grep -c "model_constants.rs" build.rs              # one write site
```

(Exact counts depend on phrasing; the intent is one builder, one emitter, one
codegen.) Note the before/after `build.rs` line count in the PR.

**Verify**: `wc -l build.rs` is meaningfully smaller than 1761; no logic block is
duplicated between the two paths.

## Test plan

This is a behavior-preserving refactor; the test plan is **byte-identity +
parity**, not new behavior:

- **Golden artifact diffs** (steps 0–4): `model_constants.rs`, `weights.bin`,
  `seeds.bin` must be byte-identical before/after for the default model (f32 and
  quant-8). This is the strongest possible proof for a refactor.
- **Parity gates**: `cargo test --features tests`,
  `cargo test --features tests,quant-8`, `cargo test --features tests,quant-16`.
- **Family coverage**: both CI acceptance models (NIFE-mxbai WordPiece dim-1024;
  potion-multilingual Unigram dim-256) build and pass `--features tests`.
- No new runtime tests (runtime unchanged). `cargo test` (default) still passes.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `cargo check --workspace` exits 0
- [ ] Default-model `weights.bin`, `seeds.bin`, `model_constants.rs` are
      byte-identical to the step-0 golden snapshots (f32 and quant-8)
- [ ] `cargo test --features tests`, `--features tests,quant-8`, `--features tests,quant-16` all pass
- [ ] Both acceptance models (NIFE-mxbai, potion-multilingual) build and pass `--features tests`
- [ ] The PHF builder, slot emitter, and `model_constants.rs` codegen each exist
      in exactly one place (no WordPiece/Unigram copies)
- [ ] `wc -l build.rs` is smaller than the pre-refactor 1761 lines
- [ ] If plan 018 landed, its guards are present in the refactored frontend (not lost)
- [ ] `git status` shows only `build.rs` modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- A golden artifact diff shows **any** difference after a step that is supposed to
  be byte-identical. Do not "accept" a diff — it means the refactor changed
  output. Revert the step and report what diverged.
- Any parity gate or acceptance model regresses. Same rule: behavior-preserving
  means green stays green.
- You discover the two paths' "duplicated" blocks are not actually identical in
  some load-bearing way (a real algorithmic difference, not just variable names).
  Report the difference — do not paper over it by picking one arbitrarily.
- The refactor pulls you toward editing `src/` or changing the emitted-constant
  contract. That is out of scope; stop and report.
- `build.rs` drifted from the excerpts (likely if plan 018 merged) and you cannot
  confidently map the steps onto the current structure — re-read fully and report
  your remapping before proceeding.

## Maintenance notes

- This plan is the prerequisite for cheap family additions (cased WordPiece,
  WordLevel/GloVe, BPE) and for the RFC's E1–E8 optimization passes — each becomes
  "populate / transform `ModelIr`" instead of forking the monolith. Reference this
  plan when those land.
- A reviewer should treat the golden diffs as the primary evidence: a clean
  refactor of a code generator is provable, and this PR should prove it.
- Intentional non-goal: this plan does not build the full RFC IR triple
  (`TokenizerSpec` with a typed normalizer chain, `PoolingSpec` enum). It extracts
  the *backend* seams (PHF/emit/codegen) and a flat `ModelIr`. A richer typed
  `TokenizerSpec` frontend is a natural follow-up once a third family motivates it
  — note that as deferred so the next contributor knows the IR is intentionally
  minimal for now.
