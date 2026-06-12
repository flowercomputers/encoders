# Plan 009: Layer-E part 1 — per-row quantization and whole-word-UNK fidelity

> **Executor instructions**: Follow this plan step by step. Run every
> verification command (separately — never zsh `${f:+...}` loops) and
> confirm expected results. Honor STOP conditions. SKIP updating
> `plans/README.md`. Items A and B are independent — a STOP in one does
> not block the other. Reply with the gate-evidence report format your
> dispatcher specifies.
>
> **Drift check (run first)**: HEAD must be `4e8be8d` or a descendant that
> has not touched `build.rs`/`src/` since.

## Status

- **Priority**: P1
- **Effort**: M-L
- **Risk**: MED (changes the quantized artifact layout; f32 layout untouched)
- **Depends on**: plans 002 (quantized oracle), 007 (quant helpers), 010
  (SKIP_UNK scratch machinery)
- **Category**: perf/quality (layer E part 1) + correctness
- **Planned at**: commit `4e8be8d`, 2026-06-12

## Why this matters

**Item A**: the shipped Python wheel builds with `quant-8`, whose global
min/max scheme is measurably crude: the weight matrix spans [-151, 167]
(8-bit step ≈ 1.25) while special-token rows have norm ≈ 22 vs ≈ 424
typical — so quantization error concentrates exactly where averaging can't
wash it out. Measured quality vs f32: cosine 0.94 on zero-token inputs,
~0.997 on short text. Per-row affine scales (one min/scale pair per vocab
row, +8 bytes/row ≈ +0.25 MB for the default model) shrink that error by
roughly the ratio of global range to per-row range — for the CLS/SEP rows
that is ~15×. **Item B**: HF WordPiece converts a word to a single [UNK]
when ANY sub-word position fails; ese's default path keeps already-matched
pieces and appends UNK — a latent reference divergence (found during plan
010 review). Both items make the compiled artifact more faithful; both get
machine-checkable gates.

## Current state (at `4e8be8d`)

- `build.rs`: global `(global_min, global_max)` computed over ALL weights
  (`~line 339`); `quant_scale` global; `quantize_u8/quantize_u16` helpers
  (plan 007); `quantize_f32_to_bytes(values, min, scale)` writes each
  slot's weight bytes; `format_quantized_param` renders UNK/CLS/SEP consts;
  `oracle_weight(w, global_min, quant_scale)` mirrors encode+decode for
  `generate_testdata`; emitted consts include `QUANT_MIN`/`QUANT_SCALE`.
- `src/lookup.rs`: `Slot { verify: u64, weight: Param }`,
  `SLOT_SIZE = 8 + DIMENSIONS * DTYPE_SIZE`; slots are `include_bytes!` +
  transmute with `repr(C, align(8))`.
- `src/wordpiece.rs::accumulate(vector, param)` dequantizes per element
  with the GLOBAL params: `QUANT_MIN + (p as f32) * QUANT_SCALE`. Called
  from the WordPiece path, the SKIP_UNK scratch path, lib.rs (CLS/SEP),
  and `unigram.rs::pool_and_mean`.
- `src/wordpiece.rs::wordpiece_accumulate`: when `SKIP_UNK` is true, uses
  a scratch buffer and discards the whole word on mid-word failure (plan
  010); when false (DEFAULT), accumulates pieces directly and appends UNK
  on failure — keeping the partial pieces (the divergence to fix).
- `src/test/mod.rs::test_embeddings_match`: parity = runtime vs
  build-time oracle (quantization-aware since plan 002), thresholds
  0.999999 f32 / 0.9999 quant.
- Measured baselines (2026-06-12): quant-8 vs f32 ideal: min 0.9408
  (inputs 2/3/10: empty/whitespace/control), real-text floor ~0.9973.
- `benches/bench_encode.rs` exists (criterion; compares against
  model2vec-rs) — use for the throughput regression gate.
- Test inputs in `generate_testdata` (11 strings) do NOT currently include
  a word that triggers mid-word WordPiece failure (first piece matches,
  a continuation doesn't) — item B adds one.

## Item A — per-row quantization (quant-8 / quant-16 builds only)

Decided design (do not redesign):

- **Layout**: for quant builds, each slot becomes
  `{ verify: u64, row_min: f32, row_scale: f32, weight: [uN; DIMS] }`
  (`SLOT_SIZE = 8 + 8 + DIMENSIONS * DTYPE_SIZE`). The f32 build's layout
  is UNCHANGED (no per-row params; zero cost). Emit the layout via the
  existing codegen (`model_constants.rs` + `Slot` struct fields gated on
  a new emitted const or on `DTYPE_SIZE != 4` — pick the plainest
  mechanism; a `cfg(any(feature = "quant-8", feature = "quant-16"))` on
  the struct field plus matching SLOT_SIZE arithmetic is acceptable).
  Confirm no struct tail padding: 8 + 8 + DIMS×size is a multiple of 8
  for every supported dim/quant combo (all DIMS are multiples of 8).
- **Encoding**: per row, `row_min = min(row)`, `row_scale =
  (max(row) - min(row)) / quant_max` with the existing epsilon guard;
  empty slots get (0.0, 1.0) and zero bytes. UNK/CLS/SEP consts get their
  own per-row params emitted as consts (`UNK_MIN/UNK_SCALE/...`) — keep it
  plain, three pairs of consts; `format_quantized_param` quantizes each
  with its own row params.
- **Runtime**: `accumulate` takes the row's params. The natural shape:
  `accumulate(vector, param, min, scale)` with the f32 build passing the
  global constants (which are (0.0, 1.0) — compiles to the same code as
  today). Lookup returns the slot, so the WordPiece/Unigram paths read
  `slot.row_min/row_scale` (quant builds) or pass the consts (f32). Keep
  the call sites obvious; comments state which build uses which params.
- **Oracle**: `oracle_weight` becomes per-row aware — quantize+dequantize
  each weight with ITS row's params (compute row min/scale in
  `generate_testdata` exactly as the packer does — or better, factor one
  `row_quant_params(row) -> (f32, f32)` used by both, so they cannot
  drift).
- **Global QUANT_MIN/QUANT_SCALE consts**: keep emitting for the f32 path
  (0.0/1.0) but quant builds no longer use them for slot weights — delete
  or repurpose cleanly; no dead misleading constants.

**The quality gate (new, the point of the exercise)**: in quant builds,
`generate_testdata` additionally emits `TEST_OUTPUTS_F32` (the raw f32
oracle, no quantize-dequantize). A new test
`test_quantization_quality` (in `mod optional`, quant builds only)
asserts `cosine(runtime_output, f32_reference) >= QUALITY_THRESHOLD` for
every input — thresholds: quant-8 ≥ 0.999, quant-16 ≥ 0.999999.

GATES (evidence = command + output):
- **A1 — the gate detects the problem**: implement the quality test FIRST,
  before the per-row change; `cargo test --features tests,quant-8` →
  `test_quantization_quality` FAILS with ~0.94 on the degenerate inputs.
  Record the failing values, commit the test separately.
- **A2 — per-row lands, parity holds**: after the layout change,
  `cargo test --features tests,quant-8` and `tests,quant-16` → ALL pass,
  including parity at the existing thresholds AND the new quality test.
  Record the new worst-case quality cosines (expect ≥ 0.999 even on
  empty/whitespace inputs).
- **A3 — f32 path untouched**: `cargo test` and `cargo test --features
  tests` pass with TEST_OUTPUTS identical semantics (no layout change in
  f32 builds; `git diff` on the f32-relevant hunks shows only the
  threaded-through const params).
- **A4 — model regressions**: NIFE acceptance (with its env vars) and
  potion-multilingual acceptance, plus one quant-8 distilled-model build:
  `ESE_MODEL_DIR=<bge artifacts> ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1
  cargo test --features tests,dim-256,quant-8` → all pass (Unigram +
  distilled + quant compose).
- **A5 — no throughput regression**: `cargo bench --features quant-8 --
  encode_single` (or the closest existing criterion filter) before/after —
  per-call time within noise (record numbers; the extra two f32 reads per
  row are expected to be invisible next to the dims loop).
- **A6 — zero warnings**, `cargo check --workspace` green.

## Item B — whole-word-UNK alignment (default WordPiece path)

Make the default path's UNK semantics match HF WordPiece: on mid-word
failure the WHOLE word contributes exactly one UNK (partial pieces
discarded). Implementation: unify on the plan-010 scratch-buffer walk for
BOTH modes — on failure, SKIP_UNK=true contributes nothing (unchanged);
SKIP_UNK=false contributes one UNK accumulate + count (instead of
partials+UNK). The fast path for fully-matching words is identical work
modulo one array copy; keep it plain.

GATES:
- **B1 — a triggering input exists**: find a probe string for the DEFAULT
  model where `tokenizers` yields `[UNK]` for a word whose first piece
  matches but a continuation doesn't (write a 10-line probe using the
  build-cached tokenizer via python `tokenizers`, or pick from known cases
  — e.g. a word starting with a vocab prefix followed by a character with
  no `##`-continuation entry). Add it to `generate_testdata`'s input list.
  Record the word and its oracle token ids.
- **B2 — the gate detects the divergence**: with the new input and the OLD
  runtime behavior, `cargo test --features tests` parity FAILS (or, if the
  cosine impact is below threshold, the magnitude assertion fails — record
  which). If neither fails, STOP: the divergence claim needs re-checking;
  report the probe data.
- **B3 — fix lands**: after unifying the walk, `cargo test --features
  tests` (now 12 inputs) passes at the unchanged 0.999999 + relative-L2
  gates; quant combos pass; fuzz test passes.
- **B4 — regressions**: same model sweep as A4.

## Scope

**In scope**: `build.rs`, `src/lookup.rs`, `src/wordpiece.rs`,
`src/unigram.rs` (accumulate call sites only), `src/lib.rs` (call sites +
doc line), `src/test/mod.rs`, `benches/` only if a filter name is needed.
**Out of scope**: vocab surgery passes (E1–E3), reweighting (E4), decase
(E5), greedy-ify (E8), readme beyond one quantization paragraph,
`distill/`, `api-py/`, CI matrix changes.

## STOP conditions

- A1's quality test does NOT fail before the change (the baseline claim is
  wrong — report measured values).
- A2's quality cosine lands below 0.999 for quant-8 after per-row scales —
  the scheme delivered less than the data predicted; report per-input
  values and the rows' min/scale stats instead of loosening.
- Slot padding math doesn't hold for some dim/quant combo — report the
  combo.
- B2 shows no detectable divergence — report probe evidence; do not "fix"
  what cannot be demonstrated.
- Any f32-build behavior change (A3) — the f32 path must be inert.

## Maintenance notes

- Per-row params make the quant thresholds in `test_embeddings_match`
  conservative; after A2's measured values, tightening the 0.9999 parity
  threshold is a safe follow-up (note actuals in the report).
- The deferred layer-E passes (E1–E5, E8) remain stubbed in
  `plans/README.md` with rationale; revisit on user pull.
- Reviewer: scrutinize the slot-layout arithmetic (SLOT_SIZE vs struct
  size for every dim×quant combo), the single-source `row_quant_params`
  shared by packer and oracle, and B1's probe evidence.
