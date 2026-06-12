# Plan 007: Reviewability cleanup — make the branch's work read plainly for a human reviewer

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving on.
> If anything in "STOP conditions" occurs, stop and report. Items are
> independent — on a STOP in one, report it and continue with the others.
> SKIP updating `plans/README.md` — the reviewer maintains the index.
>
> **Drift check (run first)**: confirm HEAD is `e0c9808` (or a descendant
> that has not touched build.rs / src/ / readme.md / ci.yml since).

## Status

- **Priority**: P1
- **Effort**: M (many S items)
- **Risk**: LOW-MED (item B tightens build guards — could newly reject models; that is intended)
- **Depends on**: plans 001–006 (all DONE)
- **Category**: tech-debt / dx / docs
- **Planned at**: commit `e0c9808`, 2026-06-12

## Why this matters

The maintainer's bar for this codebase: legible, KISS, "unclever" — an
ordinary developer must be able to read and test it without tripping. A
legibility audit of the branch found: guard code whose comments promise
behavior the runtime doesn't deliver (trust-destroying for a reviewer),
mechanical duplication (two download functions, the same quantize expression
in three places — one missing its clamp), a test hunk with broken
indentation, and doc/CI drift. None of it is architectural; all of it is
the kind of thing a careful human reviewer would flag line by line.

## Current state

All at `e0c9808` on `advisor/improvements`:

- `build.rs:473-525` — `check_normalizer`: accepts `Replace` ("generally
  fine"), `NFC|NFKC|NFKD` and unknown-shape (`None` type) configs that
  `src/pretokenizer.rs` does NOT implement (it does: lowercase, NFD,
  strip combining marks, CJK isolation — nothing else). Absent normalizer
  is called "identity; runtime handles it" — wrong: the runtime always
  lowercases and strips.
- `build.rs:531-591` — `derive_add_special_tokens`: `if uses_cls_sep
  { true } else { false }` (l.564-568); the `specials.contains_key(...)`
  OR-fallback (l.562) can return true for templates that never insert
  CLS/SEP; no cross-check of stored ids vs vocab (known bug class: the
  published potion-base-8M tokenizer stores teacher ids 101/102 in
  `post_processor.special_tokens[*].ids` while its compacted vocab has
  `[CLS]`=2/`[SEP]`=3 — ids 101/102 are the tokens `×` and `ß`; today ese
  silently picks by-name rows and diverges from the reference at cosine
  ≈0.843).
- `build.rs:761-800` — `download_if_missing(path, url, sha)` and
  `download_if_missing_no_checksum(path, url)` are near-duplicates,
  selected by if/else at `build.rs:175-184`.
- `build.rs:719-753, 861-872` — the expression
  `(((v - min) / scale).round() as uN).clamp(...)` appears in
  `quantize_f32_to_bytes` (both arms), `oracle_weight` (both arms), and
  `format_quantized_param` (both arms, MISSING the clamp — a latent
  inconsistency with the shipped bytes).
- `build.rs:253, 369` — `zero_row_special` and `zero_row` are identical
  zero vecs created twice.
- `build.rs:632-635` — `parse_safetensors` falls back to a phantom
  `"embedding.weight"` key when a file has zero tensors, producing a
  confusing downstream panic instead of a clear one.
- `src/test/mod.rs:93-119` — zero-vector guard block: comment + guard sit
  at 8-space indent, the following `assert!`/`rel` block at 12-space,
  inside the same `for` body. `rel` is a one-use closure.
- `src/lib.rs:79-94` — `encode_single` doc: "encoding is case-insensitive"
  stated as unconditional contract; no mention that output is
  L2-normalized when compiled with `ESE_NORMALIZE_OUTPUT=1`.
  `src/lib.rs:102` — `token_count` initialized to 2 before SEP is
  accumulated (correct, but needs the one-line comment explaining the
  pre-count). `src/lib.rs:76` — magic `16` rayon threshold, no comment.
- `readme.md:7` — "two public encoding functions" (crate now also exports
  `DIMENSIONS` and `MODEL_NAME`); `readme.md:~53` — cache-path note doesn't
  mention `cargo clean` wipes it / `ESE_CACHE_DIR` persists it;
  "Running tests" section (~215) lacks a network-required note and says
  `cargo test --workspace` while CI runs `cargo test`.
- `.github/workflows/ci.yml` — matrix lacks `tests,quant-16`; model-cache
  key `hf-model-${{ hashFiles('build.rs') }}` is shared across the two
  OSes (second OS's save is a no-op → macOS re-downloads each run).
- `spikes/greedy-fidelity/` — no README marking it as a frozen one-off;
  `pyproject.toml` pins `skeletoken>=0.1.0` but the script needs the
  v0.3.3 API (`TokenizerModel.make_model_greedy()`).

Models that must keep compiling after this plan: the default
(static-retrieval-mrl-en-v1: BertNormalizer lowercase:true,
BertProcessing post_processor) and NIFE-mxbai (BertNormalizer
lowercase:true, post_processor null). Both verified compatible with the
tightened guards below.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Full local matrix | `for f in "" tests tests,quant-8 tests,quant-16 tests,rayon,dim-256; do cargo test ${f:+--features $f}; done` | all pass |
| NIFE acceptance | `ESE_MODEL=stephantulkens/NIFE-mxbai-embed-large-v1@970a0151f3faeddbab5c68be38d52bbc5ea719ab ESE_MODEL_SHA256=fd3b36cb5e4ffeb92ade9fc370151c45418ae78592476a936d513ccb0ff61d06 ESE_TOKENIZER_SHA256=05f709262ae338d32529b88fe4acb0a1ba119ca1e84a85df772acfb8e8bc1361 ESE_NORMALIZE_OUTPUT=1 cargo test --features tests,dim-1024` | all pass |
| Potion inconsistency rejection (NEW behavior) | `ESE_MODEL=minishlab/potion-base-8M@bf8b056651a2c21b8d2565580b8569da283cab23 cargo check 2>&1 \| grep -i "post_processor"` | clear build error naming the id inconsistency |
| Workspace check | `cargo check --workspace` | exit 0 |
| YAML sanity | `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` | exit 0 |

## Scope

**In scope**: `build.rs`, `src/lib.rs` (docs/comments only), `src/test/mod.rs`,
`readme.md`, `.github/workflows/ci.yml`, `spikes/greedy-fidelity/README.md`
(create), `spikes/greedy-fidelity/pyproject.toml` (one version bound).

**Out of scope**: `src/pretokenizer.rs`, `src/wordpiece.rs`, `src/lookup.rs`,
`api-py/`, any runtime behavior change beyond the guard tightening described,
the PHF construction, `main()` decomposition (deliberately rejected —
linear-script style stays), quantization scheme changes, `plans/`.

## Steps

### Item A: collapse the duplicated download + quantize helpers (build.rs)

1. Merge the two download functions into one:
   `fn download_if_missing(path: &Path, url: &str, expected_sha256: Option<&str>)`.
   When `Some`, verify cached file and downloaded bytes as today; when
   `None`, skip hashing. The call site if/else at ~175-184 collapses to two
   straight calls passing `model_sha256.as_deref()` etc.
2. Add two tiny single-value helpers next to `quantize_f32_to_bytes`:
   ```rust
   fn quantize_u8(v: f32, min: f32, scale: f32) -> u8 {
       ((v - min) / scale).round() as u8 // `as` saturates; no clamp needed
   }
   fn quantize_u16(v: f32, min: f32, scale: f32) -> u16 { ... }
   ```
   Use them in `quantize_f32_to_bytes`, `format_quantized_param`, and
   `oracle_weight` (oracle becomes `min + quantize_u8(w,min,scale) as f32 * scale`).
   This also makes `format_quantized_param` consistent with the shipped
   bytes (it currently lacks the clamp; note: `as` saturates since Rust
   1.45, so the existing `.clamp(0,255)` was already a no-op — delete it
   and say so in the comment, or keep clamp in both; pick ONE convention
   and apply it everywhere).
3. Reuse one `zero_row` vec for both the specials init (~253) and the PHF
   fill (~369).
4. `parse_safetensors`: replace the `"embedding.weight"` phantom fallback
   with a clear panic when a file contains zero non-metadata tensors.

**Verify**: `cargo test --features tests` and `--features tests,quant-8`
pass; `git diff` shows the two download fns merged.

### Item B: make the guards honest and the specials derivation safe (build.rs)

1. `check_normalizer` — make the accepted list mirror
   `src/pretokenizer.rs` exactly, and fix the comments:
   - keep accepting: `BertNormalizer` with `lowercase != false`,
     `Sequence` (recurse), `Lowercase`, `NFD`, `StripAccents`.
   - REJECT (panic, naming the component + roadmap pointer): `Replace`,
     `NFC`, `NFKC`, `NFKD`, anything with an unrecognized or missing
     `type`.
   - Absent normalizer (`null`): keep accepting but emit a
     `cargo:warning` that the runtime applies BERT-uncased normalization
     (lowercase + NFD + strip marks) regardless — fix the "identity"
     comment.
2. `derive_add_special_tokens` —
   - replace `if uses_cls_sep { true } else { false }` with the expression;
   - drop the `specials.contains_key(...)` OR-fallback: the single/pair
     TEMPLATES are the signal (a special_tokens map entry that no template
     references does not get inserted at encode time);
   - NEW cross-check: when the template references `[CLS]`/`[SEP]`, read
     `post_processor.special_tokens["[CLS]"].ids[0]` (and SEP) and compare
     with `vocab["[CLS]"]` / `vocab["[SEP]"]`. On mismatch, panic:
     "model '<name>' post_processor stores id <X> for [CLS] but the vocab
     maps [CLS] to <Y> — the tokenizer.json is internally inconsistent
     (known for model2vec-compacted vocabs); refusing to compile because
     the reference tokenizer and ese would pool different rows." If the
     `ids` field is absent, skip the cross-check (template-name path
     stays).
3. Keep the function-level doc comments in sync with what the code now
   does.

**Verify**: default + NIFE acceptance commands pass (table above);
potion-base-8M command now FAILS with the inconsistency message (capture
it in your report); Unigram rejection unchanged
(`ESE_MODEL=minishlab/potion-multilingual-128M@73908c3438cf03b6a01bcb9611d62b23d0726f08 cargo check` still errors naming Unigram).

### Item C: parity test readability (src/test/mod.rs)

Re-indent the whole zero-vector + assertion block consistently (the `for`
body level). Replace the one-use `rel` closure with a module-level
`fn relative_l2_error(want: &[f32], got: &[f32]) -> f32` next to
`cosine_similarity`, with a one-line doc comment. Keep the existing
threshold comment block. Resulting body shape:

```rust
let want_norm = ...; let got_norm = ...;
if want_norm < f32::EPSILON {
    assert!(got_norm < f32::EPSILON, ...);
    continue;
}
let sim = cosine_similarity(want, got);
assert!(sim > threshold, ...);
#[cfg(not(any(feature = "quant-8", feature = "quant-16")))]
assert!(relative_l2_error(want, got) < 1e-4, "relative L2 error for {i} too high");
```

**Verify**: `cargo test --features tests` passes; visually confirm one
consistent indent level (paste the block in your report).

### Item D: doc accuracy (src/lib.rs comments + readme.md)

1. `encode_single` doc: case-insensitivity described as a property of the
   compiled normalizer ("the default model is uncased; input is
   lowercased"), keep the doctest (it is true for the default build); add
   one sentence: output is L2-normalized when built with
   `ESE_NORMALIZE_OUTPUT=1`.
2. One-line comment at the `token_count` init: pre-counts CLS + SEP; the
   SEP embedding is added after the loop.
3. One-line comment at the rayon `16` threshold: empirically chosen
   batch-size cutoff; below it sequential wins.
4. readme: "two public encoding functions" → mention `encode`,
   `encode_single`, plus the `DIMENSIONS` and `MODEL_NAME` constants;
   cache note gains: lives under `target/` (wiped by `cargo clean`),
   set `ESE_CACHE_DIR` outside `target/` to persist; "Running tests"
   gains a network-required note (crates.io + huggingface.co on first
   run) and `cargo test --workspace` becomes `cargo test` (matching CI;
   the workspace bindings are covered by `cargo check --workspace`).

**Verify**: doctests pass (`cargo test`); `grep -c "ESE_NORMALIZE_OUTPUT" src/lib.rs` ≥ 1;
`grep -c "workspace" readme.md` reflects the change you made.

### Item E: CI matrix + cache keys (.github/workflows/ci.yml)

Add `"tests,quant-16"` to the matrix features. Prefix the test job's model
cache key with the OS: `key: ${{ matrix.os }}-hf-model-${{ hashFiles('build.rs') }}`.
Leave the nife-acceptance job's key as is (single-OS).

**Verify**: YAML sanity command passes; `cargo test --features tests,quant-16`
passes locally.

### Item F: mark the spike as frozen (spikes/greedy-fidelity/)

Create `spikes/greedy-fidelity/README.md` (3–5 lines): one-off measurement
backing `docs/rfc-generalization.md` Appendix A; results frozen in
`results.json`; not maintained, not part of the build/test workflow; how to
re-run if ever needed (`uv run python measure_fidelity.py`). Change
`pyproject.toml`'s `skeletoken>=0.1.0` to `skeletoken>=0.3.3` (the script
uses the v0.3.3 `TokenizerModel.make_model_greedy()` API).

**Verify**: `test -f spikes/greedy-fidelity/README.md`;
`grep "0.3.3" spikes/greedy-fidelity/pyproject.toml` → 1 match.

## Done criteria

- [ ] Full local matrix passes (all five feature combos)
- [ ] NIFE acceptance command passes
- [ ] potion-base-8M build FAILS with the post_processor-id inconsistency
      message (new, intended)
- [ ] Unigram and unpinned rejections unchanged
- [ ] One `download_if_missing` function; `grep -c "no_checksum" build.rs` → 0
- [ ] The quantize round/cast expression exists in exactly one place per
      width (`quantize_u8`/`quantize_u16`)
- [ ] `src/test/mod.rs` block consistently indented; `rel` closure gone
- [ ] readme + lib.rs doc updates present; doctests pass
- [ ] ci.yml has 5 matrix feature entries and OS-prefixed cache keys
- [ ] `spikes/greedy-fidelity/README.md` exists; skeletoken bound is 0.3.3
- [ ] `cargo check --workspace` exits 0
- [ ] No files outside scope modified (`git status`)

## STOP conditions

- Tightened normalizer guard rejects the DEFAULT model or NIFE — the
  accepted-shape list is wrong; report the offending component instead of
  loosening blindly.
- The potion cross-check panics for the DEFAULT model (its post_processor
  ids must match its vocab: 100/101/102) — the cross-check logic is wrong.
- Any test that passed at `e0c9808` fails after a doc-only change — you
  changed more than docs; revert and report.
- `as`-saturation claim in Item A.2 doesn't hold for a case you can
  demonstrate — keep the clamps everywhere and note it.

## Maintenance notes

- The normalizer accepted-list now mirrors `src/pretokenizer.rs` exactly
  and says so — when layer A (plan 008+) makes normalization config-driven,
  delete `check_normalizer` in the same change.
- The specials id cross-check is the permanent fix for the
  model2vec-compacted-vocab inconsistency (recorded in plans/README.md);
  if model2vec fixes their exporter upstream, the check simply never fires.
- Reviewer: scrutinize Item B's rejection messages (they are the product
  surface for "ese refused my model — why?") and confirm Item A.2 picked
  one clamp convention consistently.
