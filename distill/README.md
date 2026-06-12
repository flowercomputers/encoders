# distill

Turn any Hugging Face sentence-transformers teacher into ese-compatible artifacts.

## Quick start

```sh
cd distill
uv run python distill.py BAAI/bge-base-en-v1.5 --out out/bge-base
```

This downloads the teacher (bge-base-en-v1.5, ~440 MB) to your Hugging Face
cache, distills it with PCA to 256 dimensions, and writes to `out/bge-base/`:

- `model.safetensors` — single F32 tensor named `embeddings`, shape `[vocab, 256]`
- `tokenizer.json` — the distilled (compacted) tokenizer
- `DISTILL_INFO.md` — metadata: teacher, dims, model2vec version, settings

Then compile the artifacts into ese:

```sh
ESE_MODEL_DIR="$PWD/out/bge-base" ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 \
  cargo test --features tests,dim-256
```

`ESE_NORMALIZE_OUTPUT=1` enables L2-normalization of output vectors (required for cosine
similarity, matching model2vec's default). `ESE_SKIP_UNK=1` excludes UNK-id tokens from
mean pooling, matching `model2vec.StaticModel.tokenize()` behavior for distilled artifacts.

## Usage

```
uv run python distill.py <teacher> --out <dir> [--dims N]
uv run python distill.py --check <dir>
uv run python distill.py --fidelity <dir>
```

| Argument | Description |
|---|---|
| `<teacher>` | HF model ID (e.g. `BAAI/bge-base-en-v1.5`) |
| `--out <dir>` | Output directory (created if it doesn't exist) |
| `--dims N` | PCA output dimensions (default: 256) |
| `--check <dir>` | Validate an existing artifact directory, no distillation |
| `--fidelity <dir>` | Run the export fidelity oracle (G3) — 20 sentences, cosine ≥ 0.999999 |

## Validation and repair

After distillation, the tool automatically validates the output. The same
checks can be run standalone with `--check`.

Validation steps (mirror the checks in `build.rs`):

1. **Single tensor** — `model.safetensors` must contain exactly one tensor.
2. **Tensor name** — must be `embeddings` (ese's convention).
3. **Dtype** — must be F32.
4. **Vocab consistency** — tensor shape `[vocab_size, dims]` must match the
   tokenizer's vocabulary count.
5. **Tokenizer family** — must be WordPiece or Unigram (SentencePiece).
   BPE and other families are rejected with a clear message.
6. **Specials-id consistency** — `post_processor.special_tokens[*].ids[0]`
   must match `model.vocab[token]` for `[CLS]` and `[SEP]`.

### The specials-id repair

model2vec compacts the teacher's vocabulary. After compaction, the token ids
shift. However, model2vec's exporter can leave the *teacher's* original ids
(e.g. 101/102 for `[CLS]`/`[SEP]`) inside `post_processor.special_tokens[*].ids`,
even though the compacted vocab maps those tokens to different ids (e.g. 0/1).

ese's build.rs detects this inconsistency and refuses to compile. The distill
tool repairs it automatically: if the named token (`[CLS]` or `[SEP]`) exists
in the compacted vocab, the stored id is rewritten to the vocab's id, and the
repair is printed. For example:

```
  REPAIR: [CLS] post_processor id 101 → 0 (vocab compaction fix)
  REPAIR: [SEP] post_processor id 102 → 1 (vocab compaction fix)
```

If the named token does not exist in the vocab, the inconsistency is not
repairable and the tool exits non-zero with a clear error message.

## Supported tokenizer families

ese compiles **WordPiece** and **Unigram** (SentencePiece) models.

WordPiece requirements (enforced by `guard_tokenizer_family` in `build.rs`):
- Normalizer must be `BertNormalizer` (lowercase=true), or a Sequence of
  `Lowercase`, `NFD`, `StripAccents`.

Unigram requirements (enforced by `guard_unigram_shapes` in `build.rs`):
- Pre-tokenizer must be `Metaspace` with `replacement=▁`, `prepend_scheme=always`,
  `split=false`.
- Normalizer must be a Sequence of: `Precompiled`, zero or more single-char
  `String Replace` (pad ops), `Regex Replace` with `" {2,}"`, `Regex Replace`
  with `\s+`, and optionally `Strip`.
- `byte_fallback` must be false or absent.

BPE models (GPT-2, RoBERTa, many modern transformers) are not supported —
the build will fail with a clear error naming the unsupported tokenizer type.
See `docs/rfc-generalization.md` for the roadmap.

For the authoritative list of accepted shapes, read the `guard_tokenizer_family`,
`check_normalizer`, `derive_add_special_tokens`, and `cross_check_special_token_ids`
functions in `build.rs`.

## How it works

1. `model2vec.distill()` runs the teacher's sentence-transformers model on a
   static vocabulary list and applies PCA to project embeddings to `--dims`.
2. `save_pretrained()` writes a `model.safetensors` + `tokenizer.json` pair.
3. The distill tool validates artifact shapes and repairs specials-id mismatches.
4. The output is a drop-in ese model directory: `ESE_MODEL_DIR=out/bge-base`.

The distilled model is entirely static — no transformer inference at runtime.
ese compiles it into a lookup table at `cargo build` time.

## Notes

- Teacher downloads go to your Hugging Face cache (`~/.cache/huggingface/`).
- Distilled outputs in `out/` are gitignored and should never be committed.
- The `--dims` flag defaults to 256. Higher dims improve quality; lower dims
  reduce binary size and lookup cost. 256 is recommended for bge-base-en-v1.5.
- Always pass `ESE_NORMALIZE_OUTPUT=1` when compiling ese with distilled bge/e5
  models — these are trained with cosine similarity and expect normalized vectors.
