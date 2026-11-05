![Our mouse Scott](.github/willie.svg)

# encoders

Crate `encoders` is the fastest way to embed 

```rust
let v = encoders::encode_single("hello world");
let batch = encoders::encode(["potato", "root vegetable"]);
```

`encode` encodes a batch, in parallel for batches of 128 or more when the `rayon` feature is on. `encode_single` encodes one string; `encode_into` writes one embedding into caller-provided storage.

### Choosing a model

Every model in the catalog is selectable with a feature of the same name. With no model feature the default is `flower-computer/bekko-static-v1-slim-75k`. Widths are `mrl-*` features and storage is `float32`, `int16` (the default feature) or `int8`; an explicit `float32` or `int8` overrides the default:

```toml
encoders = { features = ["bekko-static-v1-slim-32k", "mrl-256", "int8", "rayon"] }
```

| feature                                 | model                                                                             | trained widths                                        |
| --------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `bekko-static-v1-slim-100k`             | flower-computer/bekko-static-v1-slim-100k (25 languages, 100k vocab, f16)         | 1024, 512, 256, 128, 64, 32                           |
| `bekko-static-v1-slim-75k` (default)    | flower-computer/bekko-static-v1-slim-75k (25 languages, 75k vocab, f16)           | 1024, 512, 256, 128, 64, 32                           |
| `bekko-static-v1-slim-50k`              | flower-computer/bekko-static-v1-slim-50k (25 languages, 50k vocab, f16)           | 1024, 512, 256, 128, 64, 32                           |
| `bekko-static-v1-slim-32k`              | flower-computer/bekko-static-v1-slim-32k (14 languages, 32k vocab, f16)           | 1024, 512, 256, 128, 64, 32                           |
| `bekko-static-embedding-v1`             | hotchpotch/bekko-static-embedding-v1 (40 languages, 200k vocab, f16)              | 1024, 512, 256, 128, 64, 32                           |
| `static-retrieval-mrl-en-v1`            | sentence-transformers/static-retrieval-mrl-en-v1                                  | 1024, 512, 256, 128, 64, 32                           |
| `static-similarity-mrl-multilingual-v1` | sentence-transformers/static-similarity-mrl-multilingual-v1                       | 1024, 512, 256, 128, 64, 32                           |
| `potion-base-2m`                        | minishlab/potion-base-2M                                                          | 64                                                    |
| `potion-base-4m`                        | minishlab/potion-base-4M                                                          | 128                                                   |
| `potion-base-8m`                        | minishlab/potion-base-8M                                                          | 256                                                   |
| `potion-base-32m`                       | minishlab/potion-base-32M                                                         | 512                                                   |
| `potion-retrieval-32m`                  | minishlab/potion-retrieval-32M                                                    | 512                                                   |
| `potion-science-8m`                     | minishlab/potion-science-8M                                                       | 256                                                   |
| `potion-science-32m`                    | minishlab/potion-science-32M                                                      | 256                                                   |
| `potion-code-16m`                       | minishlab/potion-code-16M                                                         | 256 (ships per-token weights, use `int16` or `float32`) |
| `nife-mxbai-embed-large-v1`             | stephantulkens/NIFE-mxbai-embed-large-v1                                          | 1024                                                  |
| `nife-gte-modernbert-base`              | stephantulkens/NIFE-gte-modernbert-base                                           | 768                                                   |
| `m2v-base-output`                       | minishlab/M2V_base_output                                                         | 256                                                   |

The `bekko-static-v1-slim` family is vocabulary-pruned from `bekko-static-embedding-v1`: same embedding space, 32k to 100k of its 200k WordPiece tokens, chosen by expected retrieval damage and lightly self-distilled; see the [collection](https://huggingface.co/collections/flower-computer/bekko-static-v1-slim). Each entry is pinned to a commit sha in `build.rs` together with the widths it was trained at; those are also exposed as `encoders::model::TRUNCATION_POINTS`, `NATIVE_DIMENSIONS` and `MATRYOSHKA`. The build fails with a clear message if more than one model is enabled, or if an `mrl-*` feature asks for a width the model was not trained at. Only the unquantized upstream weights are used; quantization is done here. `potion-multilingual-128M` is not in the catalog because its Unigram tokenizer is not supported by the runtime.

#### Storage and truncation

`int16` and `int8` store the table as 16- or 8-bit integers with a separate affine range per dimension and a per-row power-of-two exponent, so small-norm rows (an untrained `[UNK]`, down-weighted frequent tokens) keep their resolution. Rows are summed in integer lanes and dequantized once per embedding, so integer tables are also faster and read half (or a quarter) of the bytes. `float32` keeps the weights as they are. `mrl-*` truncates Matryoshka models to one of their trained widths; see the table above.

### Compiling in a different model

For a model outside the catalog, copy this crate's `build.rs` into your own crate; the whole integration is one `Builder` call in `build.rs`:

```rust
let summary = ese_build::Builder::hub("sentence-transformers/static-retrieval-mrl-en-v1")
    .revision("main")
    .dims(Some(256))
    .quant(ese_build::Quant::U8)
    .generate(&out_dir)?;
println!("cargo:warning={summary}");
```

and `lib.rs` is just `ese::include_model!();`. The summary line names the model, what was detected, and how many bytes of tables the crate embeds, which is what the model adds to your final binary.

`ese-build` reads everything else from the model's own files:

| detected                                                                             | from                                                                             |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| where the weights and tokenizer live                                                 | `modules.json`, path of the `StaticEmbedding` module                             |
| L2-normalize the output                                                              | a `Normalize` module in `modules.json`, else `config.json`'s `normalize`         |
| drop `[UNK]` before pooling                                                          | `config.json` says `model_type: model2vec` (sentence-transformers keeps the row) |
| embedding tensor                                                                     | `embedding.weight`, `embeddings`, `0`, or the only 2-D tensor                    |
| vocabulary quantization                                                              | the optional `mapping` and `weights` tensors                                     |
| lowercase, accent stripping, CJK padding, subword prefix, max word length, unk token | `tokenizer.json`                                                                 |

Models with a tokenizer other than `BertNormalizer` + `BertPreTokenizer` + `WordPiece` (for example `potion-multilingual-128M`, which is Unigram) fail the build with a message naming what was found. Any module in `modules.json` other than `StaticEmbedding` and `Normalize` is rejected as not static.

Downloads are cached under `target/ese-cache/<repo>/<revision>` and shared by every consumer in the workspace.

### Fidelity against the reference implementation

Both sentence-transformers' `StaticEmbedding` and model2vec tokenize with `add_special_tokens=false` and mean-pool, so no `[CLS]`/`[SEP]` rows are ever added. `cargo test --features tests` has `ese-build` compute reference embeddings with the `tokenizers` crate under the detected policy and compares `encode_single` against them. Special tokens typed literally into the input (`"[SEP]"`) are matched the same way `tokenizers` does: on the raw text, before normalization, as single tokens.

### Benchmarks

`cargo bench` compares against `model2vec-rs` on the same model. The NanoBEIR quality benchmarks in [`./benches/py`](./benches/py) run through the Python wheel.

### Model licenses

The sentence-transformers static models are Apache-2.0; the MinishLab potion and M2V models, the NIFE models and bekko are MIT. This repository is Apache-2.0.
