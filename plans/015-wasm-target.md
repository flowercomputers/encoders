# Plan 015: WebAssembly target — fastest static embedding, in a browser tab

> **Executor instructions**: Follow step by step; GATES are the contract
> (command + output evidence each). Honor STOP conditions. SKIP updating
> `plans/README.md`.
>
> **Drift check (run first)**: HEAD `8d9ede2` or a descendant. NOTE: plan 014
> (BPE) may run in parallel in a separate worktree; your work is additive
> (new crate) and should not touch the same files — if you must edit
> `src/lib.rs`, keep it minimal so the reviewer can merge both.

## Status

- **Priority**: P1 (headline proof of the thesis; deployment reach)
- **Effort**: M–L
- **Risk**: MED (new build target; must not regress the native crate)
- **Depends on**: the current core (any compiled model)
- **Category**: direction (RFC layer D — emitters)
- **Planned at**: `8d9ede2`, 2026-06-14

## Why this matters

The blog's stated aim is "deploy anywhere with no special hardware". ese is
a Rust crate + preliminary Python wheel today (the C-ABI crate was removed
early). The most visceral proof of "fastest static embedding in the world"
is a compiled model running **sub-millisecond in a browser tab** — CPU-only,
no model download at runtime, no server. wasm is also the natural edge/lambda
target. This plan adds a wasm build that wraps `ese::encode`/`encode_single`
via `wasm-bindgen`, with a working in-browser (or headless-node) demo, a
faithful cross-check against the native binary, and measured bundle size +
throughput.

## Current state

- `src/lib.rs` exposes `encode_single`, `encode`, `DIMENSIONS`, `MODEL_NAME`,
  `IS_UNIGRAM`. Runtime is `std` but allocation-light: arrays, slices,
  `include_bytes!`-baked weights (no file IO at runtime). The `rayon`
  feature is the only threaded path (off by default; wasm is single-threaded
  — do NOT enable rayon for wasm).
- `build.rs` runs on the HOST at build time (downloads + bakes weights),
  independent of the compile target — so `--target wasm32-unknown-unknown`
  bakes the model on the host and compiles the runtime to wasm.
- There may be an UNTRACKED `api-wasm/` scaffold in the working tree
  (`git ls-files api-wasm` is empty, but check the filesystem) — inspect and
  reuse/replace it; do not assume greenfield.
- `examples/embed_stdin.rs` is the native reference for the cross-check.
- The default model (dim-512, f32) is the demo model; the wasm build picks
  up `ESE_MODEL`/`ESE_MODEL_DIR` + features exactly like the native build.

## Architecture (decided — do not redesign)

- **New `api-wasm/` crate** in the workspace (like `api-py/`): `crate-type =
  ["cdylib"]`, depends on `ese` (path) with the desired model features
  (default `dim-512`; NO rayon), plus `wasm-bindgen`. Expose:
  - `encode_single(text: &str) -> Vec<f32>` (or `Float32Array`)
  - `encode(texts: Vec<String>) -> ...` (a flat `Float32Array` + a
    `dimensions()` getter, or `Array` of `Float32Array` — pick the plainest
    wasm-bindgen shape; document it)
  - `dimensions() -> usize`, `model_name() -> String`
- **Build**: `wasm-pack build api-wasm --target web` (and/or `--target
  nodejs` for the headless test). If `wasm-pack` is unavailable, fall back to
  `cargo build --target wasm32-unknown-unknown` + `wasm-bindgen` CLI; record
  which path was used. Keep the model baked at build time (compile-time
  model — one model per bundle; runtime-loadable packs are a FUTURE plan,
  out of scope).
- **Demo**: a minimal `api-wasm/web/index.html` + `app.js` that loads the
  module, embeds two text inputs, and prints their cosine similarity — the
  "semantic search in a browser, no server" demo. Keep it dependency-free
  (vanilla JS).
- **Cross-check harness**: a small node script
  (`api-wasm/crosscheck.mjs`) that imports the `--target nodejs` build,
  encodes a fixed corpus, and compares against the native `embed_stdin`
  output (same corpus, same model/features) — per-sentence cosine ≥ 0.9999.
  This proves the wasm path is faithful (the whole point).
- **KISS**: thin binding only. No async model loading, no JS framework, no
  bundler config beyond what wasm-pack emits.

## GATES (evidence = command + output)

- **G1 — wasm builds**: `wasm-pack build api-wasm --target web` (or the
  documented fallback) succeeds; record the emitted `.wasm` byte size
  (gzipped too, via `gzip -c | wc -c`). For the default model this is the
  weights-dominated size — report it honestly.
- **G2 — faithful**: `node api-wasm/crosscheck.mjs` (against a `--target
  nodejs` build) shows per-sentence cosine ≥ 0.9999 vs the native
  `embed_stdin` on a ≥10-sentence corpus (incl. one non-ASCII line). This is
  the must-pass gate.
- **G3 — native crate unaffected**: `cargo test`, `cargo test --features
  tests`, `cargo build --example embed_stdin` all green (the workspace gains
  a crate but `ese` itself is untouched, or touched only by additive
  `#[cfg]`). `cargo check --workspace` green.
- **G4 — demo works**: either a screenshot/ःheadless check that the browser
  demo computes a similarity, OR (if no browser automation available) the
  node crosscheck doubles as the functional demo and the HTML/JS is
  present and self-consistent (loads the module, calls encode). State which.
- **G5 — throughput**: a micro-benchmark in `crosscheck.mjs` (encode N=1000
  short strings, report embeddings/sec in node wasm). Tag the machine. This
  is the "how fast in wasm" number for the blog follow-up.
- **Z**: zero warnings in the wasm build and `cargo check --workspace`;
  `git status` clean (the built `pkg/` is gitignored).

## STOP conditions

- The `ese` runtime does not compile to `wasm32-unknown-unknown` due to a
  `std` dependency that wasm lacks — report the exact symbol; the fix may be
  a small `#[cfg(target_arch = "wasm32")]` shim in `src/lib.rs` (allowed,
  minimal) or gating an offending path. If it needs more than a shim, STOP.
- G2 cosine < 0.9999 — the wasm and native builds diverge (likely a
  float/codegen difference or a binding bug); report per-sentence values,
  do not loosen.
- `wasm-pack`/`wasm-bindgen`/`wasm32` target unavailable and not installable
  in this environment — STOP and report what is missing (the user can
  install the toolchain); leave the crate + demo + harness in place so it is
  one command from working.

## Scope

**In scope**: `api-wasm/` (create: `Cargo.toml`, `src/lib.rs`, `web/`,
`crosscheck.mjs`, `.gitignore`), root `Cargo.toml` (add the workspace
member), `src/lib.rs` (ONLY a minimal wasm `#[cfg]` shim if compilation
requires it), `readme.md` (a "WebAssembly" usage section),
`.github/workflows/ci.yml` (optional wasm-build job if the toolchain is
available in CI).
**Out of scope**: runtime-loadable model packs (`.esepack`); the C-ABI
re-add; rayon-in-wasm/threads; the BPE work (plan 014); `api-py`.

## Maintenance notes

- This is the compile-time-baked wasm (one model per bundle). Runtime model
  packs (load any compiled model into one wasm binary) are the follow-up
  (RFC layer D) and need the `src/lookup.rs` alignment work — explicitly
  deferred.
- The cross-check harness is the template for any future emitter (C header,
  packs): same "compare emitter output to native binary on a corpus" shape.
- Reviewer: confirm rayon is NOT enabled for wasm, that G2 actually ran
  against a real wasm build (not a native fallback), and the reported
  `.wasm` size distinguishes code vs baked-weights.
