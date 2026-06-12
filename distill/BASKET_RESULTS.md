# Basket Results — Plan 010 Stage 2

Five-model validation of the `distill/` pipeline and `ESE_SKIP_UNK` flag.
Run date: 2026-06-12. model2vec 0.8.2, ese commit range 4500ecc–907d9ca.

## Gate summary

| # | Model | Family | Dims | Vocab | Repairs | B2 build | Min cosine | UNK tokens | Triplets | Verdict |
|---|-------|--------|-----:|------:|---------|----------|------------|------------|----------|---------|
| 1 | BAAI/bge-base-en-v1.5 | WordPiece | 256 | 29525 | none | PASS | 1.000000 | 14/20 sent | 3/3 | GREEN |
| 2 | Snowflake/snowflake-arctic-embed-xs | WordPiece | 256 | 29525 | padding: BatchLongest→Fixed:0 | PASS | 1.000000 | 14/20 sent | 3/3 | GREEN |
| 3 | sentence-transformers/all-MiniLM-L6-v2 | WordPiece | 256 | 29525 | padding: Fixed:128→Fixed:0 | PASS | 1.000000 | 14/20 sent | 3/3 | GREEN |
| 4 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Unigram (XLM-R) | 256 | 249999 | padding: BatchLongest→Fixed:0 | FAIL (clean) | N/A | N/A | N/A | CLEAN FAIL |
| 5 | Alibaba-NLP/gte-modernbert-base | BPE | — | — | — | REJECTED | — | — | — | B5 PASS |

## B6 cargo test sweep (no env flags, default NIFE model)

All three runs pass with zero test failures.

```
$ cargo test
running 1 test
test test::test_fuzz ... ok
Doc-tests: 5 passed

$ cargo test --features tests
running 2 tests
test test::optional::test_embeddings_match ... ok
test test::test_fuzz ... ok
Doc-tests: 5 passed

$ cargo test --features tests,quant-8
running 2 tests
test test::optional::test_embeddings_match ... ok
test test::test_fuzz ... ok
Doc-tests: 5 passed
```

`git status` after sweep: working tree clean.

---

## Model details

### Model 1 — BAAI/bge-base-en-v1.5 (GREEN)

- **B1 distill**: `uv run python distill.py BAAI/bge-base-en-v1.5 --out out/bge-base` — OK
- **B2 build**: `ESE_MODEL_DIR=out/bge-base ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 cargo test --features tests,dim-256` — all pass
- **B3 crosscheck**: min cosine = 1.000000 across 20 sentences (20/20 perfect). 14 UNK tokens skipped.
- **B4 triplets**: 3/3 correct
- **Repairs**: none

### Model 2 — Snowflake/snowflake-arctic-embed-xs (GREEN)

- **B1 distill**: `uv run python distill.py Snowflake/snowflake-arctic-embed-xs --out out/arctic-embed-xs` — OK
- **B2 build**: all pass
- **B3 crosscheck**: min cosine = 1.000000 across 20 sentences (20/20 perfect). 14 UNK tokens skipped.
- **B4 triplets**: 3/3 correct
- **Repairs**: tokenizer had `"strategy": "BatchLongest"` padding. model2vec's `encode_batch_fast` pads shorter sequences to the longest in the batch; PAD (id=0) has non-zero embedding (‖PAD‖ ≈ 3.93) which pollutes mean pooling. `distill.py` repaired to `"strategy": {"Fixed": 0}` automatically. See fix commit `883127e`.

### Model 3 — sentence-transformers/all-MiniLM-L6-v2 (GREEN)

- **B1 distill**: `uv run python distill.py sentence-transformers/all-MiniLM-L6-v2 --out out/all-MiniLM-L6-v2` — OK
- **B2 build**: all pass
- **B3 crosscheck**: min cosine = 1.000000 across 20 sentences (20/20 perfect). 14 UNK tokens skipped.
- **B4 triplets**: 3/3 correct
- **Repairs**: tokenizer had `"strategy": {"Fixed": 128}` padding — all sequences padded to 128 tokens. For a 10-token sentence this added 118 PAD rows to the oracle's mean pool, producing wildly wrong reference vectors. `distill.py` repaired to `"strategy": {"Fixed": 0}` automatically. See fix commit `907d9ca`.

### Model 4 — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (CLEAN FAIL)

- **B1 distill**: `uv run python distill.py sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --out out/paraphrase-multilingual` — OK, vocab=249999, Unigram, BatchLongest→Fixed:0 repair applied.
- **B2 build**: FAIL — clean, expected.

  ```
  thread 'main' panicked at build.rs:857:9:
  model '...' has pre_tokenizer type 'Sequence'; ese's Unigram runtime only
  supports Metaspace pre-tokenizers (see docs/rfc-generalization.md).
  ```

  The distilled XLM-R tokenizer uses a `Sequence` pre-tokenizer (a chain of
  multiple pre-tokenizers). ese's Unigram runtime only accepts a bare `Metaspace`
  pre-tokenizer. This is a documented limitation in `docs/rfc-generalization.md`.
  The failure is clean (clear panic message, no partial compilation), so B2 is
  satisfied as a clean-reject gate.

- **B3–B4**: skipped (B2 never passed).
- **Status**: meets plan minimum (3/4 green). Sequence pre-tokenizer support is
  tracked in rfc-generalization.md.

### Model 5 — Alibaba-NLP/gte-modernbert-base (B5 PASS — BPE reject)

Tokenizer type verified before distillation:

```
$ curl -sL ".../tokenizer.json" | python3 -c "..."
model type: BPE
```

Running `distill.py` on a BPE model produces a distilled artifact (model2vec
itself handles the teacher) but `validate()` rejects it with:

```
VALIDATE ERROR: model 'gte-modernbert' uses a 'BPE' tokenizer; ese compiles
WordPiece and Unigram (SentencePiece) models only — BPE, WordLevel, and other
types are not supported (see docs/rfc-generalization.md).
```

Exit code: 1. No ese compilation attempted. B5 gate: PASS.

---

## Bugs found and fixed during basket run

| Commit | Description |
|--------|-------------|
| `4500ecc` | Add `ESE_SKIP_UNK` build flag; both WordPiece and Unigram paths |
| `bf61ac8` | Fix WordPiece SKIP_UNK: discard partial tokens on UNK fallback (Korean NFD decomposition corner case) |
| `883127e` | Fix two distill parity bugs: (a) `encode_batch_fast` padding pollution — encode one sentence at a time; (b) `BatchLongest` padding repair in `distill.py` |
| `907d9ca` | Fix `distill.py`: also repair `Fixed:N` (N>0) padding strategies, not just `BatchLongest` |

---

## ESE_SKIP_UNK flag (S0 gates)

- **S0a**: default NIFE build (no flag) passes all regressions — verified above in B6 sweep.
- **S0b**: NIFE + potion builds — `cargo test --features tests` and `cargo test --features tests,quant-8` pass without the flag.
- **S0c**: All three WordPiece basket models pass with `ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 --features tests,dim-256`.
- **S0d**: Zero new compiler warnings introduced by `ESE_SKIP_UNK` changes (the one pre-existing `dead_code` warning in `src/test/mod.rs` is from before this plan).
