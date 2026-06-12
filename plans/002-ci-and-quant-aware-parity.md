# Plan 002: Add CI and make the parity test verify quantized builds against a quantized oracle

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 5155c13..HEAD -- src/test/mod.rs Cargo.toml .github`
> `src/test/mod.rs` must still match the excerpt below. Changes to
> `build.rs` from plan 001 (cache-dir resolution) are expected and fine.
>
> **Revision note (2026-06-12, v2)**: v1 proposed loosening the cosine
> threshold per quant feature. A dispatch measured the true quant-8 floor at
> **0.9408** (not ~0.9997): inputs that normalize to zero word tokens embed
> as CLS+SEP only, and the CLS/SEP rows (norm ≈ 22) are 19× smaller than
> typical rows (norm ≈ 424) while the global 8-bit step is ≈ 1.25 — so
> global-affine quantization shreds them. A 0.94-tolerant threshold would be
> too loose to catch real regressions. v2 instead makes the **oracle
> quantization-aware**: reference outputs are computed from
> quantize→dequantize'd weights, so the test verifies "the runtime
> faithfully reproduces the compiled artifact" at a tight threshold for every
> configuration. "Quantized artifact quality vs f32" is a separate, measured
> tradeoff (see the investigate item in `plans/README.md`).

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/001-branch-reconciliation.md (DONE)
- **Category**: tests
- **Planned at**: commit `5155c13`, 2026-06-12 (v2 revised same day at `a037a6b`)

## Why this matters

Nothing runs this repo's tests automatically, and the one configuration that
is actually shipped to users is broken under test today: the Python wheel
(`api-py/Cargo.toml`) builds the core crate with `quant-8`, and
`cargo test --features tests,quant-8` fails — the parity test compares a
quantized runtime against **unquantized f32** reference outputs with a fixed
0.999999 cosine threshold. Quantized builds are therefore untestable, so
nobody tests them, so the shipped configuration has zero coverage. This plan
(a) makes the build-time oracle compute its reference outputs from the same
quantized weights the artifact actually contains — restoring a tight
implementation-fidelity contract for every feature combination — and
(b) adds a GitHub Actions workflow exercising the feature matrix on every
push. Every subsequent plan relies on this gate.

## Current state

(At `a037a6b` on `advisor/improvements`.)

- `src/test/mod.rs` — two tests:
  - `test_fuzz` (always on): adversarial/random strings through
    `encode_single`, asserts no panic.
  - `test::optional::test_embeddings_match` (only with `--features tests`):
    compares `crate::encode(TEST_INPUTS)` against `TEST_OUTPUTS` generated
    at build time. Assertion at `src/test/mod.rs:73-83`:

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

- `build.rs::generate_testdata` (bottom of the file, `#[cfg(feature = "tests")]`)
  computes the reference: for each input, tokenize with the `tokenizers`
  crate, then mean-pool **raw f32 rows** of `weights`:

  ```rust
  for &tok_id in token_ids {
      for (v, &w) in vector
          .iter_mut()
          .zip(unsafe { weights.get_unchecked(tok_id as usize) }.iter())
      {
          *v += w
      }
  }
  ```

- `build.rs` quantization (main path, NOT to be modified): global min/max
  computed over all weights; `quantize_f32_to_bytes(values, min, scale)`
  encodes each value as `((v - min) / scale).round() as u8` (or u16);
  the runtime dequantizes per element as
  `QUANT_MIN + (p as f32) * QUANT_SCALE` (`src/wordpiece.rs::accumulate`).
- Measured facts (2026-06-12, dim-512): quant-8 per-input cosines vs f32
  reference: min 0.94082105 (inputs 2, 3, 10 — empty string, whitespace,
  control chars: all normalize to zero word tokens → CLS+SEP-only
  embeddings); real-text floor 0.9973. quant-16 min: 0.9999985.
- Feature flags (root `Cargo.toml`): `dim-*`, `quant-8`, `quant-16`,
  `rayon`, `tests`. Default `dim-512`, f32.
- No `.github/` directory — no CI.
- Model cache: `shared_model_dir()` in `build.rs` resolves
  (1) `ESE_CACHE_DIR` env var, (2) `target/ese-cache` ancestor walk,
  (3) `$TMPDIR/ese-cache/<pkg>` fallback. First build downloads ~100 MB.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Unit tests | `cargo test` | 1 unit test + 3 doctests pass |
| Parity (f32) | `cargo test --features tests` | 2 tests + 3 doctests pass |
| Parity (q8) | `cargo test --features tests,quant-8` | passes after Step 1 |
| Parity (q16) | `cargo test --features tests,quant-16` | passes after Step 1 |
| Workspace check | `cargo check --workspace` | exit 0 |
| YAML sanity | `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` | exit 0 |

## Scope

**In scope** (the only files you should modify/create):
- `build.rs` — ONLY the `#[cfg(feature = "tests")] generate_testdata`
  function plus one new `#[cfg(feature = "tests")]` helper. The emitted
  artifact (`weights.bin`/`seeds.bin`/`verify.bin`/`model_constants.rs`)
  must be byte-identical to before.
- `src/test/mod.rs` — comment/threshold updates only as described.
- `.github/workflows/ci.yml` (create)

**Out of scope** (do NOT touch):
- The quantization scheme itself (`quantize_f32_to_bytes`, global min/max,
  `QUANT_*` constants) — improving quant quality (per-row scales) is a
  separate investigate item with data in `plans/README.md`.
- The unquantized behavior of the oracle — f32 builds must produce
  identical TEST_OUTPUTS to today.
- `api-py/`; the test input list in `generate_testdata`.

## Git workflow

- Work and commit on your worktree's current branch (integration target is
  `advisor/improvements`).
- Commit messages: short lowercase imperative.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Make the oracle quantization-aware in `generate_testdata`

Add a helper next to `generate_testdata` (both `#[cfg(feature = "tests")]`):

```rust
#[cfg(feature = "tests")]
fn oracle_weight(w: f32, global_min: f32, quant_scale: f32) -> f32 {
    if cfg!(feature = "quant-8") {
        let q = (((w - global_min) / quant_scale).round() as u8).clamp(0, 255);
        global_min + (q as f32) * quant_scale
    } else if cfg!(feature = "quant-16") {
        let q = (((w - global_min) / quant_scale).round() as u16).clamp(0, 65535);
        global_min + (q as f32) * quant_scale
    } else {
        w
    }
}
```

This must mirror `quantize_f32_to_bytes` (encode) composed with the
runtime's `accumulate` dequantization (decode) **exactly**, including the
saturating `as` cast. Then thread `global_min` and `quant_scale` into
`generate_testdata(...)` (they are computed in `main()` before the call)
and apply the helper at the accumulation site:

```rust
*v += oracle_weight(w, global_min, quant_scale)
```

Note: in unquantized builds `main()` sets `(global_min, quant_scale)` to
`(0.0, 1.0)` and `oracle_weight` returns `w` unchanged — f32 reference
outputs must be bit-identical to before (verified below).

In `src/test/mod.rs`, replace the magic-number comment with one explaining
the contract: the reference is computed from the same
(possibly quantized) weight table the artifact contains, so this test
verifies that the runtime faithfully reproduces the compiled model — it
does NOT measure quantization quality vs f32 (that is a quality tradeoff
tracked separately). Keep the threshold at `0.999999`.

**Verify**:
- `cargo test --features tests` → pass (f32 oracle unchanged)
- `cargo test --features tests,quant-8` → pass
- `cargo test --features tests,quant-16` → pass

**Contingency** (not a STOP): if a quant build lands below 0.999999 by
float-accumulation noise only (measure by printing the minimum: expect
≥ 0.9999), set a quant-only threshold of `0.9999` with a comment naming
accumulation-order noise, and record the measured minimum in your report.
Below 0.9999 means the oracle does not match the runtime decode path —
that IS a STOP.

### Step 2: Sanity-check the artifact is untouched

Confirm your build.rs change cannot alter shipped bytes: re-run the f32 and
quant-8 builds and compare the emitted `weights.bin` checksums before/after
your commit (e.g. `shasum target/debug/build/ese-*/out/weights.bin` — pick
the most recent build dir, or set a temp `CARGO_TARGET_DIR` for a clean
before/after). If you cannot conveniently produce a before/after pair,
satisfy this instead by demonstrating your diff touches only
`#[cfg(feature = "tests")]` items: `git diff build.rs` shows changes only
inside `generate_testdata`, the new helper, and the `generate_testdata`
call site.

**Verify**: one of the two demonstrations above, recorded in your report.

### Step 3: Add the CI workflow

Create `.github/workflows/ci.yml`:

- Trigger: `push` to `master` and `pull_request`.
- One `test` job, matrix over:
  - os: `ubuntu-latest`, `macos-latest`
  - features: `""` (default), `tests`, `tests,quant-8`, `tests,rayon,dim-256`
- Steps:
  1. `actions/checkout@v4`
  2. `dtolnay/rust-toolchain@stable`
  3. `Swatinem/rust-cache@v2`
  4. `actions/cache@v4` for the model cache: set `ESE_CACHE_DIR` to a
     stable path (e.g. `~/.cache/ese`) via job-level `env` and cache that
     path with `key: hf-model-${{ hashFiles('build.rs') }}` (rotates when
     the model URLs change).
  5. Run tests with the matrix feature string (pass `--features` only when
     non-empty).
- A second job `workspace-check`: ubuntu only, `cargo check --workspace`.
- Keep total matrix ≤ 8 jobs. No clippy/fmt gates (repo has no lint
  config; retro-formatting is not this plan's call).

**Verify**: YAML sanity command → exit 0; each of the four feature combos
already ran green in Step 1 (`""` = plain `cargo test`; run
`cargo test --features tests,rayon,dim-256` now too → pass).

### Step 4: Confirm the model cache is CI-cacheable

In the WORKTREE ONLY (never the main checkout): move the local model cache
aside (`mv target/ese-cache target/ese-cache.bak`), run `cargo build` once,
confirm it re-downloads and builds; restore or keep the fresh download.
Also exercise the env-var path CI will use:
`ESE_CACHE_DIR=$PWD/target/ci-cache cargo check` → exit 0 and the directory
is created/populated.

**Verify**: download warning on the cold build; no download on rerun;
`ls target/ci-cache` shows `model.safetensors` and `tokenizer.json` (or the
per-model layout — record what you see).

## Test plan

This plan IS the test plan for the repo. New coverage delivered:
- parity enforced on `quant-8` (the shipped wheel config) and `quant-16`
  against a quantization-faithful oracle at a tight threshold
- CI runs fuzz + parity on two OSes × four feature sets on every push

## Done criteria

- [ ] `cargo test --features tests,quant-8` exits 0 (was failing)
- [ ] `cargo test --features tests,quant-16` exits 0
- [ ] `cargo test --features tests` exits 0 with TEST_OUTPUTS unchanged for
      f32 (oracle returns `w` unchanged when unquantized)
- [ ] Threshold in `src/test/mod.rs` is 0.999999 (or documented 0.9999
      quant-only contingency with measured minimum in the report)
- [ ] `git diff` on `build.rs` confined to `#[cfg(feature = "tests")]`
      items + the `generate_testdata` call site
- [ ] `.github/workflows/ci.yml` exists and parses as YAML
- [ ] `cargo check --workspace` exits 0
- [ ] No files outside the in-scope list modified (`git status`)
- [ ] `plans/README.md` status row updated (skip if reviewer maintains it)

## STOP conditions

Stop and report back (do not improvise) if:

- A quant build's minimum similarity lands below 0.9999 after Step 1 — the
  oracle's quantize→dequantize does not match the runtime decode path;
  report per-input similarities and your helper code.
- `src/test/mod.rs` no longer matches the excerpt (drifted).
- Implementing the oracle requires touching anything outside
  `#[cfg(feature = "tests")]` code in `build.rs`.
- `cargo check --workspace` fails for reasons unrelated to your changes —
  note it and drop the workspace-check job rather than debugging Python
  toolchains.

## Maintenance notes

- The parity contract is now "runtime reproduces the compiled artifact".
  Quantization QUALITY (quant-8 vs f32: 0.94 on degenerate inputs, ~0.997
  on short text — global-affine scales are shredded by the small-magnitude
  CLS/SEP rows; per-row scales would fix it) is tracked as an investigate
  item in `plans/README.md` and belongs to the optimization-pass layer
  (plan 005, layer E). When quantization improves, this oracle follows
  automatically since it mirrors encode+decode.
- Plan 003 should update the CI model-cache key to the pinned revision.
- Plan 006 (multi-model) adds per-model CI jobs; the `ESE_CACHE_DIR` env
  approach extends naturally.
- Plan 004 changes pooling magnitudes; this test's cosine gate is
  magnitude-invariant, and 004 adds its own magnitude assertion for
  unquantized builds — no interaction.
- Reviewer: check the workflow matrix actually expands, and that the oracle
  helper's `quant-16` arm uses u16 saturation (a copy-paste u8 clamp would
  silently corrupt the quant-16 oracle).
