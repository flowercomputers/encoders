# Plan 004: Small correctness batch — special tokens by name, reference-exact scaling, portable defaults, workspace hygiene

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index. The four items are independent — if one hits a STOP
> condition, report it and continue with the others.
>
> **Drift check (run first)**: `git diff --stat 5155c13..HEAD -- build.rs src/lib.rs src/test/mod.rs .cargo/config.toml Cargo.toml api-py/src/lib.rs`
> Changes from plans 001–003 (cache dir, pinned URLs, test thresholds) are
> expected. If the specific excerpts below don't match the live code, STOP.

## Status

- **Priority**: P2
- **Effort**: M (four small items)
- **Risk**: MED (item B changes output magnitudes — see its step)
- **Depends on**: plans/002-ci-and-quant-aware-parity.md
- **Category**: bug
- **Planned at**: commit `5155c13`, 2026-06-12

## Why this matters

Four small, audited defects:

- **(A)** `build.rs` selects the UNK/CLS/SEP embeddings by hardcoded vocab
  IDs 100/101/102 — correct only for BERT-family vocabs. Any other model
  silently gets wrong special-token vectors. This must die before
  multi-model support (plan 006) lands.
- **(B)** `encode_single` divides by a token count that *excludes* CLS/SEP,
  while the build-time reference (and the upstream model) divides by a count
  that *includes* them. Every output vector is therefore scaled by
  `(n+2)/n` relative to the reference. Cosine similarity is unaffected
  (which is why the parity test — cosine-only — never caught it), but
  dot-product/Euclidean consumers and anyone mixing ese vectors with vectors
  produced by the upstream Python model get mismatched magnitudes.
- **(C)** A committed `.cargo/config.toml` forces `-Ctarget-cpu=native` on
  every in-tree build, so any artifact built here (including release wheels)
  only runs on CPUs at least as new as the build machine — SIGILL elsewhere.
- **(D)** Hygiene: the workspace pins `resolver = "1"` (a downgrade from the
  2024-edition default), and the Python binding panics (`.expect`) on wrong
  input types instead of raising a normal Python exception.

## Current state

- `build.rs:46-52` — special tokens by hardcoded ID:

  ```rust
  entries.push((token.clone(), i));
  match i {
      100 => unk = weights[i].clone(),
      101 => cls = weights[i].clone(),
      102 => sep = weights[i].clone(),
      _ => {}
  }
  ```

  The unmerged branch `origin/codex/nife-mxbai-support` (commit `c2f03fb`)
  contains the fix pattern to port: a `special_token_id(vocab, "[UNK]")`
  helper that looks IDs up by token string, with zero-vector defaults.

- `src/lib.rs:87-109` — `encode_single`. `token_count` is incremented only
  inside `wordpiece_accumulate` (per word-piece token); CLS and SEP are
  accumulated without incrementing it:

  ```rust
  let mut vector = [0.0f32; DIMENSIONS];
  wordpiece::accumulate(&mut vector, &lookup::CLS);
  let mut token_count = 0;
  ...
  let scale = if token_count != 0 { 1.0 / token_count as f32 } else { 1.0 };
  let mut sep = [0.0f32; DIMENSIONS];
  wordpiece::accumulate(&mut sep, &lookup::SEP);
  for (v, s) in vector.iter_mut().zip(sep.iter()) {
      *v = (*v + *s) * scale;
  }
  ```

  The reference in `build.rs::generate_testdata` tokenizes with
  `tokenizer.encode(input, true)` (which appends CLS/SEP) and scales by
  `1.0 / token_ids.len()` — i.e. `1/(n+2)`.

- `.cargo/config.toml` (entire file):

  ```toml
  [build]
  rustflags=["-Ctarget-cpu=native"]
  ```

  `api-py/readme.md` already instructs wheel builders to pass
  `RUSTFLAGS="-Ctarget-cpu=native"` explicitly, so deleting the repo-wide
  default does not orphan that workflow.

- `Cargo.toml:14-15`: `[workspace] resolver = "1"`.
- `api-py/src/lib.rs:5-12` — input extraction panics on bad types:

  ```rust
  let texts: Vec<String> = if let Ok(s) = texts.extract::<String>(py) {
      vec![s]
  } else {
      texts.extract::<Vec<String>>(py).expect("expected str or list of str")
  };
  ```

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Unit tests | `cargo test` | all pass |
| Parity (f32) | `cargo test --features tests` | all pass |
| Parity (q8) | `cargo test --features tests,quant-8` | all pass (post plan 002) |
| Workspace check | `cargo check --workspace` | exit 0 |
| Wheel build (item D check) | `cd api-py && uv run maturin build --release` | wheel in `target/wheels/` (skip if `uv`/`maturin` unavailable — note it) |

## Scope

**In scope** (the only files you should modify):
- `build.rs` (item A)
- `src/lib.rs`, `src/test/mod.rs` (item B)
- `.cargo/config.toml` (delete), `readme.md` (one perf note, item C)
- `Cargo.toml` (resolver line only, item D)
- `api-py/src/lib.rs` (item D)
- `plans/README.md` (status row)

**Out of scope** (do NOT touch):
- Multi-model selection / env vars — plan 006.
- `api-py` return-shape semantics (single `str` input currently returns a
  `(1, D)` array; sentence-transformers convention is `(D,)`). Changing the
  shape is a breaking API decision — leave it; it is recorded in
  maintenance notes.
- `src/wordpiece.rs`, `src/pretokenizer.rs` — no tokenization changes here.
- Quantization scheme in `build.rs`.

## Git workflow

- Branch: `advisor/004-correctness-batch`
- One commit per item (A–D), short lowercase imperative messages, e.g.
  `look up special tokens by name`, `include cls/sep in mean-pool count`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1 (A): Select special tokens by name in `build.rs`

Port the pattern from `origin/codex/nife-mxbai-support` (view with
`git show origin/codex/nife-mxbai-support:build.rs`): add

```rust
fn special_token_id(
    vocab: &serde_json::Map<String, serde_json::Value>,
    token: &str,
) -> Option<usize> {
    vocab.get(token).and_then(|id| id.as_u64()).map(|id| id as usize)
}
```

resolve `unk_id`/`cls_id`/`sep_id` from `"[UNK]"`/`"[CLS]"`/`"[SEP]"` before
the entries loop, replace the `match i { 100 => ... }` with comparisons
against those IDs, and (deviating from the branch, which silently
zero-fills) **panic with a clear message if any of the three is missing
from the vocab** — for the single supported model they must exist, and a
silent zero CLS vector would corrupt every embedding. Initialize
`unk`/`cls`/`sep` as zero rows of length `dims` so the types stay the same.

**Verify**: `cargo test --features tests` → all pass (proves name-lookup
found the same rows 100/101/102: identical reference outputs).

### Step 2 (B): Make runtime scaling match the reference exactly

In `src/lib.rs::encode_single`, count CLS and SEP in the denominator:
initialize `let mut token_count: u32 = 2;` (with a comment: CLS + SEP) and
remove the `token_count != 0` special case (it can no longer be zero):

```rust
let scale = 1.0 / token_count as f32;
```

Resulting semantics: `(CLS + SEP + Σ tokens) / (n + 2)` — exactly the
reference computation, including the empty-input case
(`(CLS + SEP) / 2` vs reference `[CLS, SEP]` mean — identical).

Then strengthen the parity test so this can never regress silently: in
`src/test/mod.rs::test_embeddings_match`, add a magnitude check **for
unquantized builds only** (quantized weights legitimately shift magnitudes):

```rust
#[cfg(not(any(feature = "quant-8", feature = "quant-16")))]
{
    let rel = |w: &[f32], g: &[f32]| -> f32 {
        let d: f32 = w.iter().zip(g).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        let n: f32 = w.iter().map(|a| a * a).sum::<f32>().sqrt();
        d / n.max(f32::EPSILON)
    };
    assert!(rel(want, got) < 1e-4, "relative L2 error for {i} too high");
}
```

**Important**: this changes every emitted vector's magnitude. Consumers with
stored ese vectors must re-embed if they use dot product. Flag this in the
commit message body: `note: output magnitudes change; cosine unaffected`.

**Verify**: `cargo test --features tests` → all pass INCLUDING the new
magnitude assertion; `cargo test --features tests,quant-8` → all pass.

### Step 3 (C): Remove the repo-wide `-Ctarget-cpu=native`

Delete `.cargo/config.toml`. Add a short "Performance note" to `readme.md`
(near the build/usage instructions): build with
`RUSTFLAGS="-Ctarget-cpu=native"` for maximum throughput on your own
machine; omit it for binaries you distribute.

**Verify**: `cargo build` → exit 0; `test -f .cargo/config.toml` → exit 1
(file gone); `grep -c "target-cpu=native" readme.md` → ≥ 1.

### Step 4 (D): Workspace resolver + Python error handling

1. In root `Cargo.toml`, change `resolver = "1"` to `resolver = "3"`
   (the edition-2024 default; build/dev-dep features stop unifying with
   runtime features).
2. In `api-py/src/lib.rs`, return errors instead of panicking: change the
   signature to `-> PyResult<Bound<'py, numpy::PyArray2<f32>>>`, replace the
   `.expect(...)` with `.map_err(|_| pyo3::exceptions::PyTypeError::new_err("expected str or list of str"))?`,
   replace the final `.unwrap()` on `reshape` with `?` (its error type
   already converts), and wrap the return value in `Ok(...)`.

**Verify**: `cargo check --workspace` → exit 0. If `maturin` is available:
build the wheel, then in Python confirm `ese.encode(123)` raises `TypeError`
(not a panic/abort) and `ese.encode(["a", "b"]).shape == (2, 512)`. If
maturin is unavailable, note that this was verified by `cargo check` only.

## Test plan

- Item B adds the magnitude assertion to
  `src/test/mod.rs::test_embeddings_match` (unquantized builds) — this is
  the regression test for the scaling fix.
- Items A, C, D are covered by the existing parity suite + workspace check;
  no new test functions (A's panic path is build-time and exercised only by
  future non-BERT models; plan 006 adds those).
- Full gate: `cargo test && cargo test --features tests && cargo test --features tests,quant-8 && cargo check --workspace` → all exit 0.

## Done criteria

- [ ] `grep -n "100 =>" build.rs` → no matches (ID match replaced)
- [ ] `grep -n "token_count: u32 = 2" src/lib.rs` → 1 match
- [ ] New magnitude assertion present: `grep -c "relative L2" src/test/mod.rs` → ≥ 1
- [ ] `.cargo/config.toml` deleted; readme has the RUSTFLAGS note
- [ ] `grep -n 'resolver = "3"' Cargo.toml` → 1 match
- [ ] `grep -c "expect(" api-py/src/lib.rs` → 0
- [ ] `cargo test --features tests` and `cargo test --features tests,quant-8` exit 0
- [ ] `cargo check --workspace` exits 0
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- After Step 1, `cargo test --features tests` fails — name lookup resolved
  different rows than IDs 100/101/102, meaning the vocab isn't what the
  audit believed; report the resolved IDs.
- After Step 2, the magnitude assertion fails at `1e-4` — the runtime and
  reference still disagree by more than float noise; there is a second
  discrepancy the audit missed. Report the measured relative error; do not
  loosen the bound past `1e-3`.
- `resolver = "3"` causes any build failure (feature-resolution change
  surfacing a latent issue) — revert that one line and report.
- The operator (or a consumer in-tree) objects that stored-vector
  compatibility must be preserved — then item B becomes "document the
  deviation in readme + lib.rs doc comment" instead of a behavior change.

## Maintenance notes

- Item B is a **vector-compatibility break** (magnitudes only). Anyone
  upgrading with stored ese embeddings compared via dot product must
  re-embed. Cosine users are unaffected. Call this out in any release notes.
- Item A is a prerequisite for plan 006 (multi-model); 006 generalizes
  further (deriving whether CLS/SEP exist at all from the tokenizer config).
- Deferred deliberately: api-py single-string return shape `(1, D)` vs
  sentence-transformers' `(D,)` — breaking change, decide alongside the
  first real wheel release.
- Reviewer: scrutinize Step 2's empty-input behavior (token_count starts at
  2, so the `!= 0` guard removal is safe) and that no call site depended on
  the old magnitudes.
