![Our mouse Scott](.github/willie.svg)

# encoders

Crate `encoders` is the fastest way to embed 

```rust
let v = encoders::encode_single("hello world");
let batch = encoders::encode(["potato", "root vegetable"]);
```

`encode` encodes a batch, in parallel for batches of 64 or more when the `rayon` feature is on. `encode_single` encodes one string; `encode_into` writes one embedding into caller-provided storage.

### Choosing a model

The default is `sentence-transformers/static-retrieval-mrl-en-v1` truncated to 512 dimensions and stored 8-bit. Every model in the catalog is selectable with a `model-*` feature. Cargo features are additive, so to pick another model turn the defaults off and enable exactly one model, optionally a `dim-*` feature, and optionally a `quant-*` feature:

```toml
encoders = { default-features = false, features = ["model-potion-base-8m", "quant-8", "rayon"] }
```

| feature                                       | model                                                                             | trained widths                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `model-static-retrieval-mrl-en-v1`            | sentence-transformers/static-retrieval-mrl-en-v1                                  | 1024, 512, 256, 128, 64, 32                          |
| `model-static-similarity-mrl-multilingual-v1` | sentence-transformers/static-similarity-mrl-multilingual-v1                       | 1024, 512, 256, 128, 64, 32                          |
| `model-bekko-static-embedding-v1`             | hotchpotch/bekko-static-embedding-v1 (multilingual, 200k vocab, published as f16) | 1024, 512, 256, 128, 64, 32                          |
| `model-potion-base-2m`                        | minishlab/potion-base-2M                                                          | 64                                                   |
| `model-potion-base-4m`                        | minishlab/potion-base-4M                                                          | 128                                                  |
| `model-potion-base-8m`                        | minishlab/potion-base-8M                                                          | 256                                                  |
| `model-potion-base-32m`                       | minishlab/potion-base-32M                                                         | 512                                                  |
| `model-potion-retrieval-32m`                  | minishlab/potion-retrieval-32M                                                    | 512                                                  |
| `model-potion-science-8m`                     | minishlab/potion-science-8M                                                       | 256                                                  |
| `model-potion-science-32m`                    | minishlab/potion-science-32M                                                      | 256                                                  |
| `model-potion-code-16m`                       | minishlab/potion-code-16M                                                         | 256 (ships per-token weights, use `quant-16` or f32) |
| `model-nife-mxbai-embed-large-v1`             | stephantulkens/NIFE-mxbai-embed-large-v1                                          | 1024                                                 |
| `model-nife-gte-modernbert-base`              | stephantulkens/NIFE-gte-modernbert-base                                           | 768                                                  |
| `model-m2v-base-output`                       | minishlab/M2V_base_output                                                         | 256                                                  |

Each entry is pinned to a commit sha in `build.rs` together with the widths it was trained at; those are also exposed as `encoders::model::TRUNCATION_POINTS`, `NATIVE_DIMENSIONS` and `MATRYOSHKA`. The build fails with a clear message if no model or more than one is enabled, or if a `dim-*` feature asks for a width the model was not trained at. Only the unquantized upstream weights are used; quantization is done here. `potion-multilingual-128M` is not in the catalog because its Unigram tokenizer is not supported by the runtime.

#### Quantization and truncation

`quant-8` and `quant-16` store the table as 8- or 16-bit integers with a separate affine range per dimension and a per-row power-of-two exponent, so small-norm rows (an untrained `[UNK]`, down-weighted frequent tokens) keep their resolution. Rows are summed in integer lanes, one bin per exponent, and dequantized once per embedding, so quantized builds are also faster and read a quarter (or half) of the bytes. With neither feature the table is f32. `dim-*` truncates Matryoshka models to one of their trained widths; see the table above.

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
