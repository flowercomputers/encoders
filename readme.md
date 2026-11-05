![Our mouse Scott](.github/willie.svg)

# encoders

Text embeddings compiled into your binary. No model files to ship, nothing to load at startup, no Python. A sentence goes in, a fixed-width `[f32; DIMENSIONS]` comes out in well under a microsecond.

```rust
let query = encoders::encode_single("What is the capital of France?");
let docs = encoders::encode(["Paris is the capital of France.", "Tokyo is the capital of Japan."]);

for (doc, text) in docs.iter().zip(["Paris...", "Tokyo..."]) {
    println!("{:.3}  {text}", encoders::similarity(&query, doc));
}
```

```console
$ cargo add encoders --features bekko-static-v1-slim-32k,mrl-256,int8
```

## What it costs

**Quality.** These are static embeddings: each token has one fixed vector and a text is the mean of its tokens' vectors. There is no attention, so word order and context are invisible, and quality is a step below transformer encoders. On retrieval benchmarks the gap is real but not enormous; the bekko family scores about 49 to 54 NanoBEIR against the low 60s for a small transformer. In exchange a text costs a few hundred nanoseconds instead of a few milliseconds, with no GPU, no runtime dependencies and no startup. For a lot of search, dedup, clustering and routing work that is the right trade, and we hope the speed makes up for the rest.

**Binary size and compile time.** The model's table is embedded in your executable: from a few MB for a small model at a narrow width to a few hundred MB for a large vocabulary at full width in f32. The build summary prints the exact figure as `embedded=`. Any embedding model needs its weights at runtime somehow; this crate just puts them in the binary rather than next to it. The first build of a model also downloads it from the Hugging Face hub (into `target/ese-cache`) and the final link of a large table takes a while.

## Choosing a model

Every model is a cargo feature named after it, plus optional features for width and storage:

```toml
[dependencies]
encoders = { version = "1", features = ["bekko-static-v1-slim-32k", "mrl-256", "int8", "rayon"] }
```

- **Model.** Exactly one model feature. With none enabled you get the current default, `bekko-static-v1-slim-75k`. **The default may change between major versions.** If you store embeddings, name a model feature explicitly: that pins the model and its commit, so engine updates cannot silently change your vectors.
- **Width.** `mrl-1024` down to `mrl-32` truncates a Matryoshka-trained model to one of the widths it was trained at. Omit it for the model's native width. Asking a model for a width it was not trained at fails the build with the list of widths it was.
- **Storage.** `int16` is the default and the fallback: 16-bit integers, indistinguishable from f32 in practice, half the size. `int8` is a quarter the size and the fastest, with cosine agreement against f32 above 0.999 on multi-token text. `float32` keeps the weights exactly as published. An explicit `float32` or `int8` overrides the default `int16`; `int8` and `float32` together is an error.
- **`rayon`** makes `encode` parallel for batches of `PARALLEL_MIN` texts or more.

### bekko-static-v1 (recommended)

Our models, derived from [hotchpotch/bekko-static-embedding-v1](https://huggingface.co/hotchpotch/bekko-static-embedding-v1): the same embedding space with the vocabulary pruned to the tokens that matter most for the covered languages, then lightly self-distilled. Smaller vocabularies mean smaller binaries and a little less quality. All are Matryoshka-trained at 1024, 512, 256, 128, 64 and 32 dimensions, uncased, MIT licensed. Numbers are nDCG@10 at 1024 dimensions from the [collection](https://huggingface.co/collections/flower-computer/bekko-static-v1-slim).

| feature                              | vocab | languages | NanoBEIR-en | MNanoBEIR |
| ------------------------------------ | ----: | --------: | ----------: | --------: |
| `bekko-static-embedding-v1`          |  200k |        40 |        53.7 |      43.7 |
| `bekko-static-v1-slim-100k`          |  100k |        25 |        52.9 |      43.0 |
| `bekko-static-v1-slim-75k` (default) |   75k |        25 |        52.1 |      42.2 |
| `bekko-static-v1-slim-50k`           |   50k |        25 |        50.2 |      40.4 |
| `bekko-static-v1-slim-32k`           |   32k |        14 |        49.3 |      39.9 |

The slim members cover English, Arabic, German, Spanish, French, Italian, Japanese, Korean, Norwegian, Portuguese, Serbian, Swedish, Thai and Vietnamese, and the 50k and larger ones eleven more. Text in other languages tokenizes to characters or `[UNK]`.

<details>
<summary><b>sentence-transformers static models</b></summary>

Trained by the sentence-transformers team with Matryoshka loss at 1024, 512, 256, 128, 64 and 32. Apache-2.0.

| feature                                 | model                                                                                                                                             | notes                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `static-retrieval-mrl-en-v1`            | [sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1)                       | English, tuned for retrieval                             |
| `static-similarity-mrl-multilingual-v1` | [sentence-transformers/static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | multilingual, tuned for similarity rather than retrieval |

</details>

<details>
<summary><b>potion (MinishLab)</b></summary>

Distilled with model2vec: small, fast, English only, not Matryoshka-trained, so each is used at its native width. Outputs are unit length. MIT.

| feature                | model                                                                                   | native width | notes                                                             |
| ---------------------- | --------------------------------------------------------------------------------------- | -----------: | ----------------------------------------------------------------- |
| `potion-base-2m`       | [minishlab/potion-base-2M](https://huggingface.co/minishlab/potion-base-2M)             |           64 | general purpose                                                   |
| `potion-base-4m`       | [minishlab/potion-base-4M](https://huggingface.co/minishlab/potion-base-4M)             |          128 | general purpose                                                   |
| `potion-base-8m`       | [minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)             |          256 | general purpose                                                   |
| `potion-base-32m`      | [minishlab/potion-base-32M](https://huggingface.co/minishlab/potion-base-32M)           |          512 | general purpose, larger vocabulary                                |
| `potion-retrieval-32m` | [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) |          512 | tuned for retrieval                                               |
| `potion-science-8m`    | [minishlab/potion-science-8M](https://huggingface.co/minishlab/potion-science-8M)       |          256 | scientific text                                                   |
| `potion-science-32m`   | [minishlab/potion-science-32M](https://huggingface.co/minishlab/potion-science-32M)     |          256 | scientific text, larger vocabulary                                |
| `potion-code-16m`      | [minishlab/potion-code-16M](https://huggingface.co/minishlab/potion-code-16M)           |          256 | source code; ships per-token weights, so use `int16` or `float32` |
| `m2v-base-output`      | [minishlab/M2V_base_output](https://huggingface.co/minishlab/M2V_base_output)           |          256 | the original model2vec release, superseded by potion              |

</details>

<details>
<summary><b>NIFE (Nearly Inference Free Embeddings)</b></summary>

Static models aligned with a transformer teacher, so an index built with the teacher can be queried with the static model. English, native width only, unit-length outputs. MIT.

| feature                     | model                                                                                                       | native width | teacher                            |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | -----------: | ---------------------------------- |
| `nife-mxbai-embed-large-v1` | [stephantulkens/NIFE-mxbai-embed-large-v1](https://huggingface.co/stephantulkens/NIFE-mxbai-embed-large-v1) |         1024 | mixedbread-ai/mxbai-embed-large-v1 |
| `nife-gte-modernbert-base`  | [stephantulkens/NIFE-gte-modernbert-base](https://huggingface.co/stephantulkens/NIFE-gte-modernbert-base)   |          768 | Alibaba-NLP/gte-modernbert-base    |

</details>

Each model is pinned to a commit in `build.rs`. The crate's tests compare its output against the reference implementation (the `tokenizers` crate plus sentence-transformers' pooling) for every model.

## Licensing

The code in this repo is licensed Apache-2.0. However, each model has its own license, which you must respect; it is available at build time in the summary and at run time as `encoders::model::LICENSE`.

## Under the hood

The workspace has four crates. `encoders` is the one above. `ese` is the runtime: a byte-for-byte reimplementation of the BERT tokenizer pipeline, mean pooling in integer lanes, and a perfect-hash table over the vocabulary. `ese-build` runs in `build.rs`: it fetches a model, works out how it is laid out and how it expects to be pooled from its own files, quantizes it, and writes the tables. `ese-unicode` is the 64 KB of Unicode tables the tokenizer needs, generated from pinned crates and checked in.

To compile in a model outside the catalog, copy this crate's `build.rs` and point `ese_build::Builder::hub` at any sentence-transformers or model2vec static model with a BERT WordPiece tokenizer; `lib.rs` is just `ese::include_model!();`.

Benchmarks: `cargo bench` compares against `model2vec-rs` on the same model; the retrieval quality harness in [`benches/py`](./benches/py) runs through the Python wheel in [`api-py`](./api-py).
