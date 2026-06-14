# Plan 019: Vectorize the pooling accumulate loop with runtime-dispatched SIMD (portable, bit-exact)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 8a4616c..HEAD -- src/wordpiece.rs src/unigram.rs src/lib.rs src/lookup.rs`
> If any in-scope file changed since this plan was written, compare the "Current
> state" excerpts against the live code before proceeding; on a mismatch, treat
> it as a STOP condition.
>
> **Step 0 (dispatch only)**: agent worktrees are branched from `master`, NOT
> from `advisor/improvements`. Before anything else run:
> `git reset --hard advisor/improvements`. Safe — the worktree branch is fresh.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (touches the runtime hot path) — mitigated: the SIMD path is
  **bit-exact** with scalar (per-lane independent adds, no reduction), the scalar
  path is retained as fallback + oracle, and the existing parity gates guard it.
- **Depends on**: none. (Soft: land plan 017 first so you have a fair before/after
  benchmark to measure the win.)
- **Category**: perf
- **Planned at**: commit `8a4616c`, 2026-06-13

## Why this matters

`ese`'s reason to exist is speed, yet the single most-executed loop in the
runtime is **fully scalar with zero explicit SIMD**, and default builds ship
baseline-ISA code (`-Ctarget-cpu=native` is opt-in, by deliberate decision for
portability). The pooling `accumulate` runs `DIMENSIONS` (default 512) fused
multiply-adds **per token**, once per matched subword, per Viterbi token, and per
special token. A NEON/AVX2 rewrite of this exact loop shape is the textbook
4–8× win, and it is currently left entirely to the autovectorizer.

The crucial property that makes this low-risk: `accumulate` writes each output
lane independently —

```
vector[i] += min + (p[i] as f32) * scale
```

There is **no cross-lane reduction**, so a SIMD implementation produces
**bit-identical** results to the scalar one for the same inputs (floating-point
summation order across tokens is preserved, because tokens are still applied in
sequence; within one call each lane gets exactly one add). This is not a
"close enough within a cosine threshold" change — it is exact. That is why this
plan targets `accumulate` specifically and leaves the L2-norm reduction (which
*does* depend on summation order) scalar.

After this lands, default portable builds get wide SIMD where the CPU supports it
(via runtime feature detection), without forcing `target-cpu=native` and without
the SIGILL-on-older-CPUs risk that motivated removing the global flag.

## Current state

The dominant loop, `src/wordpiece.rs:7-17`:

```rust
#[inline(always)]
pub fn accumulate(
    vector: &mut [f32; crate::DIMENSIONS],
    param: &crate::lookup::Param,
    min: f32,
    scale: f32,
) {
    for (v, &p) in vector.iter_mut().zip(param.iter()) {
        *v += min + (p as f32) * scale;
    }
}
```

`Param` is `[dtype; DIMENSIONS]` where `dtype` is `f32` (default), `u8`
(`quant-8`), or `u16` (`quant-16`) — emitted by `build.rs` into
`model_constants.rs` (see `src/lookup.rs:1` `include!`). `DIMENSIONS` is always a
multiple of 8 (supported dims: 32, 64, 128, 256, 512, 768, 1024) — so 8-lane
(AVX/NEON f32x8 / f32x4×2) tiling divides evenly; a scalar remainder loop is a
safety net but should not be needed for supported dims.

In **f32 builds**, callers pass `min=0.0, scale=1.0`, so the loop is `*v += p[i]`
(the dequant terms fold away — see the `#[cfg(...)]` arms below). In **quant
builds**, `p` is `u8`/`u16` and `min`/`scale` are the per-row params.

The three call sites:

- `src/lib.rs:136,150` — CLS/SEP special tokens (WordPiece path).
- `src/wordpiece.rs:67,69,88` — per matched subword and UNK:

```rust
                #[cfg(any(feature = "quant-8", feature = "quant-16"))]
                accumulate(&mut scratch, &s.weight, s.row_min, s.row_scale);
                #[cfg(not(any(feature = "quant-8", feature = "quant-16")))]
                accumulate(&mut scratch, &s.weight, 0.0, 1.0);
```

- `src/unigram.rs:447,449,455` — per Viterbi token and fused UNK (same `#[cfg]`
  pattern).

The post-accumulate scale loops (`src/lib.rs:151-159`, `src/unigram.rs:465-467`)
are also per-element (`*v *= scale`) and bit-exact under SIMD, but they run **once
per encode**, not per token — far lower leverage. The L2-norm
(`src/lib.rs:167-173`, `src/unigram.rs:469-474`) is a **reduction** (order
matters) — leave it scalar.

Confirmed absence of any SIMD today: `grep -rE 'std::arch|core::arch|portable_simd|target_feature|is_x86_feature' src/` returns nothing.

Repo conventions:
- Edition 2024, stable toolchain. The project avoids nightly features (no
  `#![feature(...)]` anywhere). Use **stable** `std::arch` intrinsics with
  runtime detection — NOT `core::simd`/`portable_simd` (nightly-only).
- `#[inline(always)]` is used on hot helpers; keep it.
- Existing per-row-quant `#[cfg(any(feature = "quant-8", feature = "quant-16"))]`
  gating is the pattern for build-variant code; match it.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Drift check | `git diff --stat 8a4616c..HEAD -- src/` | empty or known |
| Workspace builds | `cargo check --workspace` | exit 0 |
| Default tests | `cargo test` | all pass |
| Parity (f32) | `cargo test --features tests` | all pass |
| Parity (quant-8) | `cargo test --features tests,quant-8` | all pass |
| Parity (quant-16) | `cargo test --features tests,quant-16` | all pass |
| Bench (before/after) | `cargo bench --bench gooaq --features rayon` | criterion numbers |
| Portable build sanity | `cargo build --release` (no RUSTFLAGS) | exit 0, runs without SIGILL |

## Suggested executor toolkit

- If a Rust SIMD best-practices skill is available, consult it for the
  `is_x86_feature_detected!` + `#[target_feature(enable = "avx2,fma")]` dispatch
  pattern and the safety contract for calling target-feature functions.

## Scope

**In scope** (the only files you should modify or create):
- `src/wordpiece.rs` (replace the body of `accumulate` with a dispatch; keep a
  scalar `accumulate_scalar`)
- `src/simd.rs` (create — the arch-specific implementations) **or** keep it inline
  in `wordpiece.rs` if you prefer one file; creating `src/simd.rs` is cleaner
- `src/lib.rs` (add `mod simd;` if you create the file)

**Out of scope** (do NOT touch):
- The call sites' `#[cfg]` quant gating (`wordpiece.rs`, `unigram.rs`, `lib.rs`) —
  `accumulate`'s signature stays identical, so callers are unchanged.
- `build.rs`, the slot layout, `lookup.rs` — no data-format changes.
- The L2-norm reduction loops — leaving them scalar is intentional (order-sensitive).
- Do NOT reinstate a global `-Ctarget-cpu=native` (`.cargo/config.toml`) — that
  is the portability regression plan 004 deliberately removed. Runtime dispatch
  is the whole point.

## Git workflow

- Work on `advisor/improvements`.
- Commit per logical unit; lowercase imperative, e.g.
  `add scalar accumulate fallback + simd module skeleton`,
  `add avx2 accumulate path with runtime dispatch`,
  `add neon accumulate path`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Extract the scalar implementation as the oracle, keep behavior identical

Rename the current loop body into `accumulate_scalar` (same signature as
`accumulate`), and make `accumulate` call it. No behavior change yet — this is the
refactor that gives you a reference to test SIMD against.

```rust
#[inline(always)]
pub fn accumulate(vector: &mut [f32; crate::DIMENSIONS], param: &crate::lookup::Param, min: f32, scale: f32) {
    accumulate_scalar(vector, param, min, scale);
}

#[inline(always)]
pub fn accumulate_scalar(vector: &mut [f32; crate::DIMENSIONS], param: &crate::lookup::Param, min: f32, scale: f32) {
    for (v, &p) in vector.iter_mut().zip(param.iter()) {
        *v += min + (p as f32) * scale;
    }
}
```

**Verify**: `cargo test --features tests` and `cargo test --features tests,quant-8` → all pass (byte-identical behavior; this is a pure rename).

### Step 2: Add the SIMD module with x86_64 AVX2/FMA and aarch64 NEON paths

Create `src/simd.rs` (and `mod simd;` in `lib.rs`). Implement, per `Param` element
type, a function that computes `vector[i] += min + (p[i] as f32) * scale` 8 lanes
at a time. Use stable `std::arch`:

- **x86_64**: an `#[target_feature(enable = "avx2,fma")]` `unsafe fn` using
  `_mm256_*` (load 8 f32, FMA, store). For `u8`/`u16` params, widen to f32 first
  (`_mm256_cvtepu8_epi32` → `_mm256_cvtepi32_ps`, etc.).
- **aarch64**: a NEON `unsafe fn` using `float32x4_t` (two 4-lane iterations per
  8-element tile) with `vfmaq_f32`. Widen `u8`/`u16` lanes to f32 similarly.

Each arch function processes `DIMENSIONS` in 8-element tiles; since
`DIMENSIONS % 8 == 0` for all supported dims, no remainder is required, but add a
scalar remainder loop guarded by `chunks_exact` semantics for safety/robustness.

Keep the three `Param` element types working. The cleanest structure is a small
generic or three monomorphic helpers selected by the same `#[cfg(feature)]` the
callers use; match the existing quant-gating style.

**Verify**: `cargo check --workspace` → exit 0. `cargo check --features quant-8` and `--features quant-16` → exit 0 (all three element types compile).

### Step 3: Dispatch at runtime in `accumulate`

Make `accumulate` choose the best available implementation via runtime feature
detection, falling back to `accumulate_scalar`:

- x86_64: `if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") { unsafe { accumulate_avx2(...) } } else { accumulate_scalar(...) }`
- aarch64: NEON is baseline on aarch64; call the NEON path directly (or guard with
  `std::arch::is_aarch64_feature_detected!("neon")`).
- other arches: `accumulate_scalar`.

The detection result is cheap but not free; if profiling shows it matters, cache
it in a `OnceLock<fn(...)>` or an atomic, but **start simple** — `is_x86_feature_detected!`
is already cached by std after first call.

**Verify**:
- `cargo test --features tests` → all pass.
- `cargo test --features tests,quant-8` → all pass.
- `cargo test --features tests,quant-16` → all pass.
(These pass only if the SIMD output matches the compiled reference within the
existing thresholds — and because accumulate is bit-exact, they should pass with
the *same* margins as before, not degraded ones.)

### Step 4: Add a direct SIMD-vs-scalar equivalence test

Add a `#[cfg(test)]` test in `src/wordpiece.rs` (or `src/simd.rs`) that fills a
`Param` with varied values, runs both `accumulate` and `accumulate_scalar` into
separate vectors from the same starting state, and asserts they are **bit-equal**
(`assert_eq!` on the `[f32; DIMENSIONS]` arrays, not approximate). Do this for the
active build's element type. Because there is no cross-lane reduction, exact
equality must hold; if it doesn't, the SIMD math is wrong.

For quant builds, use representative `min`/`scale` and `u8`/`u16` values; for f32
builds, use `min=0.0, scale=1.0` and assorted f32s including negatives and zero.

**Verify**: `cargo test simd` (or the test's name) under default, `quant-8`, and
`quant-16` → the equivalence test passes in each.

### Step 5: Measure the win

With plan 017's harness (or at minimum `cargo bench --bench gooaq --features rayon`),
record before/after throughput. Capture single-input latency and batched QPS.
Note the numbers in the PR description (not in a committed report — that is plan
013's job; if 013 has landed, regenerate per its step 6).

**Verify**: the benchmark runs; you have a before (scalar, from git stash or the
pre-change commit) and after (SIMD) number on the same machine.

**If you cannot run benchmarks in this environment**: that is fine for
correctness — the parity + equivalence tests are the gates. Report that the
speedup must be measured on real hardware and provide the before/after procedure.

## Test plan

- **Equivalence (new)**: SIMD vs scalar bit-equality test (step 4), run under
  default / `quant-8` / `quant-16`.
- **Parity (regression)**: `cargo test --features tests`,
  `cargo test --features tests,quant-8`, `cargo test --features tests,quant-16` —
  all must pass with thresholds unchanged.
- **Fuzz (regression)**: `cargo test` runs `test_fuzz` — must still pass (no
  panics/UB on adversarial input through the new dispatch).
- **Portability**: `cargo build --release` with no RUSTFLAGS, then run the binary
  (e.g. `cargo run --example embed_stdin --release` if present) to confirm no
  SIGILL — the runtime dispatch must not assume AVX2 at compile time.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `cargo check --workspace` exits 0
- [ ] `cargo test`, `cargo test --features tests`, `cargo test --features tests,quant-8`, `cargo test --features tests,quant-16` all exit 0
- [ ] A SIMD-vs-scalar bit-equality test exists and passes under default, quant-8, and quant-16
- [ ] `grep -rE 'is_x86_feature_detected|is_aarch64_feature_detected' src/` returns a match (runtime dispatch present)
- [ ] No `.cargo/config.toml` forcing `target-cpu=native` was added (`test ! -f .cargo/config.toml || ! grep -q target-cpu .cargo/config.toml`)
- [ ] `accumulate`'s public signature is unchanged (`grep -n "pub fn accumulate" src/wordpiece.rs` shows the same parameters; call sites in `unigram.rs`/`lib.rs` are unmodified)
- [ ] `git status` shows only `src/` files modified/created
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The SIMD-vs-scalar equivalence test is NOT bit-exact. That means the
  vectorized math diverges from scalar — do not "relax to a tolerance"; the
  whole premise is exactness. Find the bug or report it.
- Any parity test (`--features tests*`) regresses below its existing threshold —
  that indicates the SIMD path is wrong (since accumulate is bit-exact, parity
  margins should be identical, not merely "still passing").
- A portable `cargo build --release` (no RUSTFLAGS) produces a binary that
  SIGILLs on the test machine — the dispatch is compiling AVX2 into the default
  path instead of gating it behind runtime detection.
- You find yourself needing to change a call site or the `Param`/slot layout —
  the signature is meant to stay fixed; if it can't, stop and report why.
- The toolchain rejects `std::arch` intrinsics as nightly — you are reaching for
  `core::simd` by mistake; use stable `std::arch` + `#[target_feature]`.

## Maintenance notes

- This is the first SIMD in the codebase; future SIMD (e.g. a binary/Hamming
  distance path, or vectorizing the L2-norm with a careful order-stable
  reduction) should follow the same runtime-dispatch + scalar-oracle pattern
  established here.
- A reviewer should scrutinize: the `unsafe` target-feature functions are only
  ever called after the corresponding `is_*_feature_detected!` check (the safety
  contract), the widening of `u8`/`u16` lanes is correct (zero-extend, not
  sign-extend), and the bit-equality test actually covers the active element type.
- The scale loops in `lib.rs`/`unigram.rs` (per-encode, not per-token) are
  deliberately left scalar — low leverage. The L2-norm reduction is left scalar
  on purpose (order-sensitive). Note both in the PR so a future contributor
  doesn't "finish the job" and silently change output.
- After this lands, re-run plan 017's benchmark and regenerate the report — a
  fresh, fair speed number that *includes* the SIMD win is the headline payoff.
