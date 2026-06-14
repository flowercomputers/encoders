# Plan 018: Make a wrong model fail the build loudly instead of compiling to silently-wrong embeddings

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 8a4616c..HEAD -- build.rs readme.md`
> If `build.rs` changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it as a
> STOP condition.
>
> **Step 0 (dispatch only)**: agent worktrees are branched from `master`, NOT
> from `advisor/improvements`. Before anything else run:
> `git reset --hard advisor/improvements`. Safe — the worktree branch is fresh.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW–MED (adds build-time validation; the only behavior change is that
  some currently-"successful" wrong builds will now fail loudly — that is the goal)
- **Depends on**: none. (Should land **before** plan 020, which restructures
  `build.rs`; 016 must carry these guards into the new frontend.)
- **Category**: bug / correctness
- **Planned at**: commit `8a4616c`, 2026-06-13

## Why this matters

`ese`'s whole value proposition is now "point `ESE_MODEL` at a model and get a
hyper-fast compiled embedder." But the compiler has **no safety net**: a user can
aim it at a model it doesn't actually support correctly and get a binary that
**compiles successfully and emits plausible-but-wrong vectors**. Three compounding
gaps (all verified in `build.rs` at `8a4616c`):

1. **Dimension truncation silently assumes MRL ordering.**
   `build.rs:1396` does `let dims = full_dims.min(trunc_dims())` and keeps the
   first N columns of every row, unconditionally, with no warning. Correct for a
   Matryoshka (MRL) model like the built-in default; **silently wrong** for the
   common non-MRL model — the user picks `dim-256`, gets well-formed vectors with
   degraded geometry, and ships them.

2. **No table-integrity check on any build.** There is no NaN/inf scan, no
   all-zero-row check, and no comparison of the weight tensor's row count against
   the vocabulary size. A corrupt download (checksums are optional) or a
   mismatched tokenizer/weights pair compiles into the binary. The WordPiece path
   even *silently drops* vocab entries whose id exceeds the row count
   (`build.rs:309`, `if i >= weights.len() { continue; }`).

3. **The safetensors parser indexes shape/data without validation.**
   `build.rs:1392-1394` reads `shape[0]`/`shape[1]` directly — a 1-D tensor
   panics with an opaque `index out of bounds`, not a model-shaped error.

This plan closes gaps that need **no test feature and no reference model** — they
run on every `cargo build`. After it lands, the dangerous failure mode flips from
"compiles, silently wrong" to "fails the build with a message naming the
problem." It also adds a short README section telling users how to get true
fidelity verification (which does need a reference).

## Current state

`build.rs` is the compiler (1761 lines). The relevant pieces:

`parse_safetensors` reads the tensor shape and truncates with no validation,
`build.rs:1390-1396`:

```rust
    let tensor = &meta[&tensor_key];
    let dtype = tensor["dtype"].as_str().unwrap_or("F32");
    let shape = tensor["shape"].as_array().unwrap();
    let params = shape[0].as_u64().unwrap();          // rows — no rank check
    let full_dims = shape[1].as_u64().unwrap();        // cols — panics if 1-D

    let dims = full_dims.min(trunc_dims());            // <-- blind MRL-style truncation
```

The per-element read loop that can over-index a truncated/corrupt blob,
`build.rs:1408-1437` (abridged):

```rust
    let row_src_bytes = full_dims as usize * src_elem_size;
    // ...
    for row in 0..params as usize {
        let row_start = start + row * row_src_bytes;
        for col in 0..dims as usize {
            let elem_start = row_start + col * src_elem_size;
            // data[elem_start..elem_start + k] indexed with no bounds check
        }
    }
    (tensors, dims)        // returns (Vec<Vec<f32>>, dims)
```

`parse_safetensors` returns `(weights, dims)` and is called once in `main`,
`build.rs:184`:

```rust
    let (weights, dims) = parse_safetensors(&model_path);
```

The WordPiece path that silently drops overflow vocab ids, `build.rs:307-312`:

```rust
    for (token, id_val) in vocab {
        let i = id_val.as_u64().unwrap() as usize;
        if i >= weights.len() {
            continue;                    // <-- silent: tokenizer/weights mismatch hidden
        }
        entries.push((token.clone(), i));
```

The built-in default model name (used to recognize "this truncation is expected
because the default IS MRL"), `build.rs:13-14`:

```rust
static DEFAULT_MODEL_NAME: &str =
    "sentence-transformers/static-retrieval-mrl-en-v1@f60985c706f192d45d218078e49e5a8b6f15283a";
```

The Unigram path indexes weights by vocab index too, and panics opaquely on a
mismatch, `build.rs:742-743`:

```rust
        let row = if occupied[slot] {
            &weights[slot_to_weight[slot]]      // panics if a vocab id >= weights.len()
```

The test oracle's unchecked index (latent UB on a mismatched tokenizer under
`--features tests`), `build.rs:1707` (the line uses `get_unchecked` on a
tokenizer-produced id):

```rust
    let row = unsafe { weights.get_unchecked(tok_id as usize) };
```

Repo conventions for build-time errors: existing guards `panic!` with a message
that names the model and what was expected — e.g. `build.rs:1384-1388`
("safetensors file ... contains no non-metadata tensors; ese expects exactly one
embedding weight tensor per file."). **Match that voice**: name the model, state
what was found, state what was expected, and (where relevant) how to fix it.
Warnings use `println!("cargo:warning=...")` (see `build.rs:121`).

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Drift check | `git diff --stat 8a4616c..HEAD -- build.rs` | empty or known |
| Workspace builds | `cargo check --workspace` | exit 0 |
| Default build/test (built-in MRL model) | `cargo test` | all pass, **no** new truncation warning (default is MRL) |
| Parity (WordPiece default) | `cargo test --features tests` | all pass |
| Quant parity | `cargo test --features tests,quant-8` | all pass |
| WordPiece custom acceptance (non-default) | `ESE_MODEL=stephantulkens/NIFE-mxbai-embed-large-v1@970a0151f3faeddbab5c68be38d52bbc5ea719ab ESE_NORMALIZE_OUTPUT=1 cargo build --features dim-1024` | builds; truncation warning expected only if dim < model dim |
| Unigram acceptance | see `.github/workflows/ci.yml` potion-multilingual job | builds + parity |

## Suggested executor toolkit

- The two pinned acceptance models live in `.github/workflows/ci.yml` (NIFE-mxbai
  WordPiece dim-1024; potion-multilingual-128M Unigram dim-256). Use them to
  confirm real models still build after your guards.

## Scope

**In scope** (the only files you should modify):
- `build.rs` (add validation; replace silent drops/opaque panics with named errors)
- `readme.md` (add a short "Verify your model" subsection)

**Out of scope** (do NOT touch):
- Anything under `src/` — the runtime is unaffected.
- The slot layout, quantization math, or PHF construction — you are adding
  guards *around* existing logic, not changing it.
- `distill/` — the Python validator is a separate concern (a different plan).
- Do NOT add a network/reference-based fidelity check to the default build; that
  needs the `tokenizers` dep and is intentionally behind `--features tests`. This
  plan is about checks that need neither.

## Git workflow

- Work on `advisor/improvements`.
- Commit per logical unit; lowercase imperative messages, e.g.
  `add build-time embedding-matrix validation`,
  `warn on non-MRL dim truncation`, `replace silent vocab-id drop with named error`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Validate tensor shape inside `parse_safetensors`

Before the index reads at `build.rs:1393-1394`, validate the tensor rank and
dimensions, and validate the declared data region fits the blob. Replace the bare
`shape[0]`/`shape[1]` indexing with checked access that panics in the repo's
voice. Required checks:

- `shape.as_array()` is `Some` and has length exactly 2 — otherwise panic naming
  the actual rank and the file: "embedding tensor in `<path>` has rank N, expected
  a 2-D `[vocab, dims]` matrix".
- `shape[0]` and `shape[1]` parse as `u64` and are both `> 0`.
- After computing `row_src_bytes` and `start`, assert
  `start + params * row_src_bytes <= data.len()` — otherwise panic: "safetensors
  data region for `<path>` (need X bytes from offset S) exceeds file size Y;
  file is truncated or its header is inconsistent."

**Verify**: `cargo test` → all pass (the default model is a valid 2-D F32 tensor,
so no guard fires).

### Step 2: Scan the parsed matrix for non-finite values

After the row loop fills `tensors` in `parse_safetensors` (around `build.rs:1435`),
scan for NaN/inf and reject. Panic naming the first offending row index:
"embedding row R in `<model>` contains a non-finite value (NaN/inf); refusing to
compile a poisoned table." (Iterate `tensors`; on `!v.is_finite()` panic.)

Keep it O(rows × dims) — it runs once at build time; that's fine.

**Verify**: `cargo test` → all pass (the default model is finite).

### Step 3: Add an explicit vocab-vs-weights row-count check; remove the silent drop

The single most dangerous mismatch is a tokenizer whose vocab size disagrees with
the weight tensor's row count. Add a shared check used by **both** paths.

- In the WordPiece path, replace the silent `if i >= weights.len() { continue; }`
  at `build.rs:309-311` with: collect the max vocab id first (or count entries),
  and if any id `>= weights.len()`, panic naming the mismatch:
  "tokenizer for `<model>` references vocab id I but the weight tensor has only N
  rows — tokenizer and model.safetensors are mismatched." (If you prefer to keep
  the loop shape, hoist the check above the loop: compute the max id and compare
  to `weights.len()` once.)
- In the Unigram path, the analogous opaque panic is at `build.rs:742-743`
  (`&weights[slot_to_weight[slot]]`). Add the same up-front check in
  `build_unigram` after `entries` is built (entries' second field is the vocab
  index): if `entries.iter().map(|(_,id)| id).max()` ≥ `weights.len()`, panic with
  the same named message.

**Important nuance**: the check is "max referenced id < weights.len()", not
strict equality — a model may legitimately have more weight rows than the
tokenizer uses (trailing unused rows). Equality is too strict; over-reference is
the real bug. State this in a comment.

**Verify**:
- `cargo test --features tests` → all pass (default model is consistent).
- The two acceptance models (NIFE-mxbai, potion-multilingual) still build (run at
  least one).

### Step 4: Warn on non-MRL dimension truncation

At the truncation site (`build.rs:1396`), when `full_dims > dims` (columns are
being dropped), emit a loud `cargo:warning` **unless the model is the built-in
default** (which is known-MRL, so truncation is expected and silent for it).

The default model is recognized by name: compare the resolved model name to
`DEFAULT_MODEL_NAME`. The cleanest way is to thread an `is_builtin_default: bool`
from `resolve_model` (set `true` only in the built-in branch at `build.rs:142`)
down to where truncation happens; or, simpler, pass the model name into
`parse_safetensors` and compare. Pick the smaller diff.

Warning text (match the repo voice):
"truncating embedding dimension from {full_dims} to {dims}: ese keeps the first
{dims} columns. This is only correct for MRL/Matryoshka-trained models. If
'{model}' is not MRL-trained, the resulting vectors will be semantically
degraded — build at the model's native dimension, or verify quality with
`cargo test --features tests` / `distill.py --fidelity`."

Do **not** hard-fail (truncation is a legitimate operation for MRL models, and a
hard error would break existing non-MRL users who knowingly accept the tradeoff).
A loud warning is the right severity.

**Verify**:
- `cargo test` (built-in MRL default, dim-512 from a 1024-dim model) → all pass
  and **no** truncation warning in the output (the default is exempt). Confirm
  with: `cargo build 2>&1 | grep -i "truncating embedding dimension"` → no match.
- Build a non-default model at a reduced dim and confirm the warning DOES fire,
  e.g. NIFE-mxbai at `dim-256`: the warning appears once.

### Step 5: Fix the oracle's unchecked index (latent UB under `--features tests`)

At `build.rs:1707`, replace `weights.get_unchecked(tok_id as usize)` with a
checked access that panics in the repo voice on a mismatch:

```rust
    let row = weights.get(tok_id as usize).unwrap_or_else(|| panic!(
        "oracle: tokenizer produced id {tok_id} but the weight tensor has only {} rows \
         — tokenizer/model mismatch", weights.len()));
```

(With step 3 in place this should be unreachable for consistent models, but the
oracle must not invoke UB on the inconsistent case.)

**Verify**: `cargo test --features tests` → all pass; `cargo test --features tests,quant-8` → all pass.

### Step 6: Document how to actually verify fidelity

Add a short subsection to `readme.md` under "Using a different model" titled
**"Verify your model before trusting it"** explaining the layered guarantees:

- A plain `cargo build` now runs **structural** checks (shape, finite values,
  vocab/row consistency) and warns on non-MRL truncation — but it does **not**
  check that the embeddings match the reference model.
- For **fidelity**, run `cargo test --features tests` (verifies the runtime
  reproduces the compiled model using the model's own tokenizer) and/or, for
  distilled artifacts, `distill.py --fidelity` (verifies against the teacher).

Keep it to a tight paragraph plus the two commands. Do not overstate what the
build-time checks cover.

**Verify**: `grep -n "Verify your model" readme.md` returns a match.

## Test plan

The new code is build-time validation; it is exercised by builds, not unit tests.

- **Regression (must stay green)**: `cargo test`, `cargo test --features tests`,
  `cargo test --features tests,quant-8`, `cargo test --features tests,quant-16`
  — the default model is valid, so every guard must pass silently.
- **New positive coverage**: confirm at least one *non-default* model still
  builds with your guards (NIFE-mxbai at native dim-1024 → no truncation warning;
  at dim-256 → exactly one truncation warning).
- **Optional but recommended** — a targeted build-time unit test for the finite
  scan and shape check would need a synthetic safetensors fixture, which is heavy.
  If you can add a small `#[cfg(test)]` helper in `build.rs` that runs the finite
  scan over a hand-built `Vec<Vec<f32>>` containing a NaN and asserts it panics,
  do so; if it complicates the build script, skip it and rely on the build-level
  verification above (note the choice in the PR).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `cargo check --workspace` exits 0
- [ ] `cargo test` exits 0 with **no** truncation warning for the default model
      (`cargo build 2>&1 | grep -ci "truncating embedding dimension"` → `0`)
- [ ] `cargo test --features tests` and `cargo test --features tests,quant-8` exit 0
- [ ] `grep -n "get_unchecked" build.rs` shows the oracle line is gone (or, if other
      `get_unchecked` uses remain elsewhere, the one at the former line 1707 is now checked)
- [ ] A non-default model at a reduced dim emits exactly one truncation warning
      (demonstrated with NIFE-mxbai dim-256 or equivalent)
- [ ] `grep -n "Verify your model" readme.md` returns a match
- [ ] `git status` shows only `build.rs` and `readme.md` modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any guard you add fires on the **built-in default model** (it is a known-good
  valid 2-D finite MRL table — a guard tripping on it means the guard is wrong).
- `build.rs` has drifted from the excerpts above (the truncation site, the vocab
  loop, or the oracle line don't match) — re-confirm before editing.
- The vocab-vs-weights check fires on either pinned acceptance model
  (NIFE-mxbai, potion-multilingual). Those are known-good; a trip means your
  "max referenced id" logic is wrong (e.g. you used strict equality, or counted
  entries instead of max id).
- Adding the finite scan materially slows the default build (more than a couple
  seconds) — report it; the scan should be negligible.

## Maintenance notes

- **Plan 020 (IR refactor) depends on these guards.** When `build.rs` is
  restructured into a frontend/IR, these checks must move into the IR-construction
  step, not be dropped. The vocab-vs-weights check and the finite scan belong in
  the `EmbeddingMatrix` construction; the truncation warning belongs wherever
  `dims` is decided. Leave a comment at each guard saying so.
- A reviewer should check: the truncation warning is correctly exempted for the
  default model (no warning-spam on the common build), and the vocab/row check
  uses max-referenced-id (not equality), so legitimate trailing-row models build.
- Deferred out of this plan (note in PR): promoting optional checksums to
  required for remote `ESE_MODEL` builds (SECURITY finding) — related, but a
  separate behavior change with its own opt-out flag; and a true default-build
  fidelity gate (needs a reference and the `tokenizers` dep — intentionally left
  behind `--features tests`).
