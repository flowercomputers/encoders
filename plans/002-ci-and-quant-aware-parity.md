# Plan 002: Add CI and make the parity test pass on quantized builds

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 5155c13..HEAD -- src/test/mod.rs build.rs Cargo.toml .github`
> If `src/test/mod.rs` changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition. (Changes to `build.rs` from plan
> 001's cache-portability merge are expected — read the merged
> `shared_model_dir()` to learn the cache path in Step 4.)

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/001-branch-reconciliation.md
- **Category**: tests
- **Planned at**: commit `5155c13`, 2026-06-12

## Why this matters

Nothing runs this repo's tests automatically, and the one configuration that
is actually shipped to users is broken under test today: the Python wheel
(`api-py/Cargo.toml`) builds the core crate with `quant-8`, and
`cargo test --features tests,quant-8` **fails** — the parity test asserts
cosine similarity > 0.999999 against the reference model, but 8-bit
quantization legitimately lands at ~0.9997728 (measured at commit `5155c13`,
input index 0). The result: quantized builds are untestable, so nobody tests
them, so the shipped configuration has zero coverage. This plan makes the
parity threshold quantization-aware and adds a GitHub Actions workflow that
exercises the feature matrix on every push. Every subsequent plan (pinning,
correctness fixes, multi-model support) relies on this gate to prove it
didn't break encoding.

## Current state

- `src/test/mod.rs` — two tests:
  - `test_fuzz` (always on): feeds adversarial/random strings through
    `encode_single`, asserts no panic.
  - `test::optional::test_embeddings_match` (only with `--features tests`):
    compares `crate::encode(TEST_INPUTS)` against `TEST_OUTPUTS` generated at
    build time by `generate_testdata()` in `build.rs` (which mean-pools raw
    f32 weights using the real `tokenizers` crate as oracle). The assertion,
    at `src/test/mod.rs:73-83`:

    ```rust
    #[test]
    fn test_embeddings_match() {
        let embeddings = crate::encode(TEST_INPUTS);
        for (i, (want, got)) in TEST_OUTPUTS.iter().zip(embeddings.iter()).enumerate() {
            let sim = cosine_similarity(want, got);
            assert!(
                sim > 0.999999,
                "cosine similarity for {i} below threshold: {sim}"
            );
        }
    }
    ```

  The reference outputs are always computed from **unquantized f32** weights,
  while the crate under test may be built with `quant-8`/`quant-16`
  (lossy global min/max quantization, see `build.rs` `quantize_f32_to_bytes`).
  So the comparison is f32-reference vs quantized-runtime, and the fixed
  `0.999999` threshold only fits the unquantized build.

- Feature flags (root `Cargo.toml`): `dim-{1024,768,512,256,128,64,32}`,
  `quant-8`, `quant-16`, `rayon`, `tests`. Default is `dim-512`, f32.
- `api-py/Cargo.toml:11` — the wheel's feature set:
  `features = ["rayon", "dim-512", "quant-8"]`.
- There is no `.github/` directory — no CI of any kind.
- `.cargo/config.toml` sets `rustflags=["-Ctarget-cpu=native"]` — harmless in
  CI (native = the runner).
- The build script downloads ~100 MB from Hugging Face on first build into a
  shared cache dir (on master: `target/ese-cache/`; plan 001 may have moved
  it — read `shared_model_dir()` in `build.rs` for the truth).
- Measured baselines at `5155c13` (your numbers should be close):
  - default features: all tests pass
  - `--features tests`: all pass
  - `--features tests,quant-8`: FAILS, worst cosine ≈ 0.9997728

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Unit tests | `cargo test` | 1 unit test + 3 doctests pass |
| Parity (f32) | `cargo test --features tests` | 2 tests + 3 doctests pass |
| Parity (q8) | `cargo test --features tests,quant-8` | currently FAILS; passes after Step 2 |
| Parity (q16) | `cargo test --features tests,quant-16` | passes after Step 2 |
| Workspace check | `cargo check --workspace` | exit 0 |
| YAML sanity | `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` | exit 0 (use `ruby -ryaml` if PyYAML unavailable) |

## Scope

**In scope** (the only files you should modify/create):
- `src/test/mod.rs`
- `.github/workflows/ci.yml` (create)
- `plans/README.md` (status row)

**Out of scope** (do NOT touch):
- `build.rs` — quantization quality itself (global vs per-row scaling) is a
  separate investigation; this plan only makes the test honest about the
  existing quantization error.
- The `0.999999` unquantized threshold — do not loosen it.
- `api-py/` — wheel publishing/CI for wheels is future work.
- Test input list in `generate_testdata()` — adding cases is welcome
  elsewhere, not here.

## Git workflow

- Branch: `advisor/002-ci-quant-parity`
- Commit messages: short lowercase imperative (repo style, e.g.
  "make parity threshold quantization-aware", "add ci workflow").
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Measure the actual quantized similarity floor

Temporarily change the assertion in `src/test/mod.rs` to print all
similarities (e.g. collect `sim` values and `panic!` with the minimum at the
end), then run:

- `cargo test --features tests,quant-8 -- --nocapture` → record min sim (expect ≈ 0.9997)
- `cargo test --features tests,quant-16 -- --nocapture` → record min sim (expect ≥ 0.999999)

Revert the temporary printing before Step 2. Record both numbers in your
final report.

**Verify**: you have two recorded minima.

### Step 2: Make the threshold quantization-aware

In `src/test/mod.rs`, inside `mod optional`, replace the hardcoded threshold
with a feature-selected constant:

```rust
// Reference outputs are computed from unquantized f32 weights, so quantized
// builds legitimately deviate by the quantization error.
const SIMILARITY_THRESHOLD: f32 = if cfg!(feature = "quant-8") {
    0.999
} else if cfg!(feature = "quant-16") {
    0.99999
} else {
    0.999999
};
```

and use `sim > SIMILARITY_THRESHOLD` in the assertion (keep the failure
message printing both `sim` and the threshold). Sanity-check the constants
against Step 1's measurements: each threshold must sit at least one decade of
error below the measured minimum (e.g. measured 0.9997 → threshold 0.999 is
fine). If a measurement is *below* its proposed threshold, STOP (see below).

**Verify**:
- `cargo test --features tests` → pass
- `cargo test --features tests,quant-8` → pass
- `cargo test --features tests,quant-16` → pass

### Step 3: Add the CI workflow

Create `.github/workflows/ci.yml`:

- Trigger: `push` to `master` and `pull_request`.
- One `test` job, matrix over:
  - os: `ubuntu-latest`, `macos-latest`
  - features: `""` (default), `tests`, `tests,quant-8`, `tests,rayon,dim-256`
- Steps:
  1. `actions/checkout@v4`
  2. `dtolnay/rust-toolchain@stable`
  3. `Swatinem/rust-cache@v2` (caches cargo registry + target)
  4. `actions/cache@v4` for the **model cache directory** — read
     `shared_model_dir()` in the current `build.rs` to determine the path
     (on pre-001 master: `target/ese-cache`). Key it on a constant string
     plus the model URL line, e.g.
     `key: hf-model-${{ hashFiles('build.rs') }}` — coarse but correct:
     the URLs live in build.rs, so the key rotates when they change.
  5. Run tests: `cargo test ${{ matrix.features != '' && format('--features {0}', matrix.features) || '' }}`
- A second job `workspace-check`: ubuntu only, `cargo check --workspace`
  (compiles `api-py` bindings without linking a Python).
- Keep total matrix ≤ 8 jobs. No clippy/fmt gates in this plan — the repo
  has no lint config and retro-formatting is not this plan's call. (If you
  notice `cargo fmt --check` happens to pass cleanly, you may add a
  non-blocking `fmt` job; if it fails, leave it out and note it.)

**Verify**: YAML sanity command from the table → exit 0. Then run the matrix
locally as a smoke test (the four feature combos on your machine):
each `cargo test [--features ...]` invocation → exit 0.

### Step 4: Confirm the model cache is CI-cacheable

Delete the cached model locally (`rm -rf <model-cache-dir>` — the directory
you identified from `shared_model_dir()`, NOT the whole `target/`), then run
`cargo build` once and confirm it re-downloads and builds. This proves a cold
CI runner will succeed.

**Verify**: `cargo build` → exit 0 with a `Downloading ...` warning line on
first run; immediate rerun of `cargo build` → no download line.

## Test plan

This plan IS the test plan for the repo. New coverage delivered:
- parity now enforced on `quant-8` (the shipped wheel config) and `quant-16`
- CI runs fuzz + parity on two OSes × four feature sets on every push

No new test functions beyond the threshold change; the matrix provides the
coverage.

## Done criteria

- [ ] `cargo test --features tests,quant-8` exits 0 (was failing at `5155c13`)
- [ ] `cargo test --features tests,quant-16` exits 0
- [ ] `cargo test --features tests` exits 0 (unquantized threshold unchanged at 0.999999 — `grep -n "0.999999" src/test/mod.rs` still matches)
- [ ] `.github/workflows/ci.yml` exists and parses as YAML
- [ ] `cargo check --workspace` exits 0
- [ ] No files outside the in-scope list modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Step 1's measured quant-8 minimum is below 0.999, or quant-16 below
  0.99999 — the quantization error is larger than the audit measured, which
  may indicate a real quantization bug; report the measured values instead
  of just lowering thresholds further.
- `src/test/mod.rs` no longer matches the excerpt (drifted).
- `cargo check --workspace` fails for reasons unrelated to your changes
  (e.g. pyo3/maturin environment issues) — note it and exclude the
  workspace-check job rather than debugging Python toolchains.
- You cannot determine the model cache path from `build.rs`.

## Maintenance notes

- When plan 003 pins model revisions, update the CI model-cache key to the
  pinned revision string for precise invalidation.
- When plan 006 makes the model selectable via env var, the CI matrix should
  gain one job per supported model profile — the cache step must then key on
  the resolved model, not on `build.rs` alone.
- The quant thresholds encode "expected quantization error for the current
  model + global min/max scheme". If quantization is later improved
  (per-row scales), tighten them.
- Reviewer: check the workflow's matrix actually expands (a typo in matrix
  syntax silently runs one job).
