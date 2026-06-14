# Plan 018: Shippable wasm bundle — 60 MB → ~5 MB via quant-8 + MRL dim, measured

> **Executor instructions**: Follow step by step; GATES are the contract
> (command + output evidence each). Honor STOP conditions. SKIP updating
> `plans/README.md`. Reply in the gate-evidence format your dispatcher
> specifies.
>
> **Drift check (run first)**: HEAD `1397ee1` or a descendant.

## Status

- **Priority**: P1 (turns the wasm demo from impressive into shippable)
- **Effort**: M
- **Risk**: LOW (feature selection + measurement; no new runtime logic)
- **Depends on**: 015 (wasm crate), 009 (per-row quant), 013 (NanoBEIR harness)
- **Category**: direction (RFC layer D — deployment) + perf
- **Planned at**: `1397ee1`, 2026-06-14

## Why this matters

Plan 015 shipped a working wasm build, but it's **60 MB** — the runtime code
is 38 KB; the rest is the baked `dim-512 f32` weight table. That's a
non-starter for a browser. Two levers already exist in the crate and need
only be applied + measured: **quant-8** (u8 weights, ÷4) and **MRL dimension
truncation** (`dim-256/128/64`, the model is Matryoshka-trained so lower
dims are valid embeddings). `dim-128 + quant-8` is ~**5 MB** — a ~12×
reduction, genuinely shippable. The deliverable is a small default wasm
bundle plus an **honest size↔quality table** so a user can pick their point
on the curve.

## Current state

- `api-wasm/Cargo.toml`: `ese_core` dependency with `features = ["dim-512"]`
  (f32, no quant, no rayon). One model baked at build time.
- `api-wasm/crosscheck.mjs`: node harness comparing wasm output to native
  `embed_stdin` (cosine ≥ 0.9999) + a throughput micro-bench. Built per
  `wasm-pack build api-wasm --target {web,nodejs}`.
- Per-row quant (plan 009): quant builds store `verify(8) + row_min(4) +
  row_scale(4) + weight(dim × 1)` per slot; the runtime dequantizes in
  `wordpiece::accumulate`. quant-8 quality vs f32 is ≥ 0.9999 (plan 009
  `test_quantization_quality`) — i.e. quant-8's quality cost is negligible;
  the size/quality tradeoff is dominated by the **dim** choice.
- Slot-size math (table_size = 32768 for the 30522-token default vocab):
  - dim-512 f32: ~67 MB (matches the measured 60 MB)
  - dim-512 q8: ~17 MB | dim-256 q8: ~9 MB | dim-128 q8: ~5 MB | dim-64 q8: ~2.6 MB
- `benches/py/ese_encoder.py` (plan 013): truncates to a target dim, so it
  produces the NanoBEIR-by-dim curve directly. Default model NanoBEIR at
  dim-512 = 0.4957.
- wasm toolchain present (plan 015 used wasm-pack 0.14, wasm32 target).

## GATES (evidence = command + output)

- **G1 — size table across variants**: build the wasm bundle for
  `dim-512` (f32, the current baseline), `dim-512,quant-8`, `dim-256,quant-8`,
  `dim-128,quant-8`, `dim-64,quant-8` (pass features to `ese_core` via the
  api-wasm crate; the cleanest is to make the api-wasm features configurable
  or build with `--features` overrides — document the method). Record each
  `.wasm` **raw and gzipped** size (`wasm-pack build ...` then `ls -l` +
  `gzip -c file.wasm | wc -c`). Confirm the trend matches the slot math.
- **G2 — fidelity holds under quant**: for at least `dim-256,quant-8` AND
  `dim-128,quant-8`, the wasm output matches the **native binary built with
  the same features** at per-sentence cosine ≥ 0.9999 (`crosscheck.mjs`
  against `embed_stdin` built with matching `--features`). quant-8 + wasm
  must not introduce drift. Record the minima.
- **G3 — quality↔size curve**: NanoBEIR (plan-013 harness / `EseEncoder`)
  for the default model at dim **512, 256, 128, 64**. Combine with G1 to
  produce a table: `dim | wasm size (raw / gzip) | NanoBEIR | retention vs
  dim-512`. Note quant-8's additional quality cost is ≤ ~0.1% (cite plan 009;
  optionally confirm by also measuring the compiled q8 binary at one dim).
- **G4 — pick + document the default**: choose a shippable default for
  `api-wasm` from the curve (recommend the knee — likely `dim-128,quant-8`
  ~5 MB if it retains ≥ ~85% of dim-512 NanoBEIR; otherwise `dim-256,quant-8`).
  Set it in `api-wasm/Cargo.toml`, rebuild, confirm the demo + crosscheck
  still pass. Add a "Bundle size" section to `api-wasm/`'s README (or the
  main readme's WebAssembly section) with the G3 table and how to rebuild
  for a different point.
- **Z**: native crate unchanged (`git diff src/ build.rs` empty — this is
  feature selection, not code); `cargo test` green; `cargo check --workspace`
  green; `git status` clean (built `pkg-*/` gitignored).

## STOP conditions

- A quant-8 wasm variant's crosscheck cosine < 0.9999 — quant + wasm
  interact (e.g. a codegen/rounding difference); report per-sentence values,
  do not loosen.
- The api-wasm feature override can't select quant/dim cleanly (Cargo
  feature plumbing) — report the blocker; a small `api-wasm/Cargo.toml`
  feature-passthrough is in scope, deeper build surgery is not.
- dim-64/128 NanoBEIR collapses (e.g. < 0.5× of dim-512) — MRL truncation
  isn't graceful for this model; report the curve and default to the
  smallest dim that holds ≥ ~85% retention instead.
- Any need to change `src/` or `build.rs` — out of scope; this is
  configuration + measurement.

## Scope

**In scope**: `api-wasm/Cargo.toml` (features + a passthrough if needed),
`api-wasm/` README/demo notes, `benches/py/` (a small size↔quality table or
an addition to `QUALITY_TABLE.md`), `readme.md` (WebAssembly bundle-size
note), `.gitignore` if new build dirs.
**Out of scope**: vocab pruning (the deeper lever — its own plan 019 / Track
B); runtime-loadable model packs; the Rust runtime (`src/`, `build.rs`);
brotli tooling beyond a `gzip` measurement; `api-py`.

## Maintenance notes

- **Vocab pruning (plan 019 / Track B) is the next size lever**: dropping
  rows for tokens absent from a target corpus shrinks the PHF table itself
  (`dim-128 q8` ~5 MB → ~1.5 MB for a ~10k in-domain vocab). It changes
  coverage (OOV → UNK), so it needs a corpus and a build.rs prune pass —
  deliberately deferred here to keep this plan a clean config+measure win.
- The size↔quality table is the artifact users need; keep it updated when
  the default model or quant scheme changes.
- Reviewer: confirm G2 actually compared wasm to native at the SAME features
  (a common error is comparing q8 wasm to f32 native — that would fail for
  the wrong reason), and that the chosen default's retention is honestly
  reported, not rounded up.
