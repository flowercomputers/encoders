![Our mouse Scott](.github/willie.svg)

# encoders

The encoders crate is the **fastest way to embed text** on CPU.[^1] We take [a collection of state-of-the-art static embedding models](#configuration-and-model-selection), and compile them directly into Rust code. That means no downloading models at runtime, no GPU, no inference runtime, no Python... just nanosecond scale text embedding!

```rust
let query = encoders::encode_single("What is the capital of France?");
let docs = encoders::encode(["Paris is the capital of France.", "Tokyo is the capital of Japan."]);

for (doc, text) in docs.iter().zip(["Paris...", "Tokyo..."]) {
    println!("{:.3}  {text}", encoders::similarity(&query, doc));
}
```

You can also configure encoders with a number of crate features for model selection, quantization, Matryoshka truncation (depending on a model's support for it), and parallelization (via [rayon](https://github.com/rayon-rs/rayon)).

```console
# defaults to bekko-static-v1-slim-50k, int16 quantization
$ cargo add encoders
$ cargo add encoders --features='potion-retrieval-32m, int8, mrl-256, rayon'
```

## Configuration and model selection

```toml
[dependencies]
encoders = { version = "1", features = ["bekko-static-v1-slim-32k", "mrl-256", "int8", "rayon"] }
```

> [!WARNING]
> If you store embeddings from this crate, select a model feature explicitly: that pins the model so that crate updates cannot silently change your vectors.
> _**We will change the default model as the state of the art changes!**_

- **Model.** With none enabled you get the current default, `bekko-static-v1-slim-50k`.
- **Width.** `mrl-1024` down to `mrl-32` truncates a Matryoshka-trained model to one of the widths it was trained at. Omit it for the model's native width; asking for a width a model was not trained at fails the build!
- **Storage.** `int16` is the default, as they're indistinguishable from f32 in practice at half the size. `int8` is a quarter the size and the fastest, with cosine agreement against f32 above 0.999 on multi-token text.
- **`rayon`** makes `encode` parallel for large batches.

### bekko models (recommended)

The model [hotchpotch/bekko-static-embedding-v1](https://huggingface.co/hotchpotch/bekko-static-embedding-v1) represents the current state-of-the-art as far as static embedding. We've trained a family of models derived from this model to shrink the vocabulary (and thus the model size), specifically for use in encoders.

These are multilingual models, and trained with MRL so their vectors can be truncated (reducing vector and model size at the cost of embedding quality). Licensed MIT.

| feature                                                                                                     | vocab | languages | NanoBEIR-en | MNanoBEIR | size int8 / int16 / f32 |
| ----------------------------------------------------------------------------------------------------------- | ----: | --------: | ----------: | --------: | ----------------------: |
| [`bekko-static-embedding-v1`](https://huggingface.co/hotchpotch/bekko-static-embedding-v1)                  |  200k |        40 |        53.7 |      43.7 |    258 / 514 / 1026 MiB |
| [`bekko-static-v1-slim-100k`](https://huggingface.co/flower-computer/bekko-static-v1-slim-100k)             |  100k |        25 |        52.9 |      43.0 |     129 / 257 / 513 MiB |
| [`bekko-static-v1-slim-75k`](https://huggingface.co/flower-computer/bekko-static-v1-slim-75k)               |   75k |        25 |        52.1 |      42.2 |     129 / 257 / 513 MiB |
| [**`bekko-static-v1-slim-50k`**](https://huggingface.co/flower-computer/bekko-static-v1-slim-50k) (default) |   50k |        25 |        50.2 |      40.4 |  65 / **129** / 257 MiB |
| [`bekko-static-v1-slim-32k`](https://huggingface.co/flower-computer/bekko-static-v1-slim-32k)               |   32k |        14 |        49.3 |      39.9 |       32 / 64 / 128 MiB |

Our smallest model, `bekko-static-v1-slim-32k`, covers English, Arabic, German, Spanish, French, Italian, Japanese, Korean, Norwegian, Portuguese, Serbian, Swedish, Thai, Vietnamese. The larger models also support Bengali, Persian, Finnish, Hindi, Indonesian, Dutch, Russian, Swahili, Telugu, Yoruba, and Chinese.

<details>
<summary><b>Potion models</b></summary>

Distilled by [MinishLab](https://github.com/MinishLab) with [model2vec](https://github.com/MinishLab/model2vec): English only, licensed MIT.

| feature                | model                                                                                   | native width | notes                                                             | size int8 / int16 / f32 |
| ---------------------- | --------------------------------------------------------------------------------------- | -----------: | ----------------------------------------------------------------- | ----------------------: |
| `potion-base-2m`       | [minishlab/potion-base-2M](https://huggingface.co/minishlab/potion-base-2M)             |           64 | general purpose                                                   |     2.3 / 4.3 / 8.3 MiB |
| `potion-base-4m`       | [minishlab/potion-base-4M](https://huggingface.co/minishlab/potion-base-4M)             |          128 | general purpose                                                   |      4.3 / 8.3 / 16 MiB |
| `potion-base-8m`       | [minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)             |          256 | general purpose                                                   |       8.3 / 16 / 32 MiB |
| `potion-base-32m`      | [minishlab/potion-base-32M](https://huggingface.co/minishlab/potion-base-32M)           |          512 | general purpose, larger vocabulary                                |       33 / 65 / 129 MiB |
| `potion-retrieval-32m` | [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) |          512 | tuned for retrieval                                               |       33 / 65 / 129 MiB |
| `potion-science-8m`    | [minishlab/potion-science-8M](https://huggingface.co/minishlab/potion-science-8M)       |          256 | scientific text                                                   |       8.3 / 16 / 32 MiB |
| `potion-science-32m`   | [minishlab/potion-science-32M](https://huggingface.co/minishlab/potion-science-32M)     |          256 | scientific text, larger vocabulary                                |       33 / 65 / 129 MiB |
| `potion-code-16m`      | [minishlab/potion-code-16M](https://huggingface.co/minishlab/potion-code-16M)           |          256 | source code; ships per-token weights, so use `int16` or `float32` |        17 / 33 / 65 MiB |
| `m2v-base-output`      | [minishlab/M2V_base_output](https://huggingface.co/minishlab/M2V_base_output)           |          256 | the original model2vec release, superseded by potion              |       8.3 / 16 / 32 MiB |

</details>

<details>
<summary><b>sentence-transformers static models</b></summary>

Trained by the [sentence-transformers](https://sbert.net) team with Matryoshka loss (see their [write-up](https://huggingface.co/blog/static-embeddings)), licensed Apache-2.0.

| feature                                 | model                                                                                                                                             | notes                                                    | size int8 / int16 / f32 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------: |
| `static-retrieval-mrl-en-v1`            | [sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1)                       | English, tuned for retrieval                             |       32 / 64 / 128 MiB |
| `static-similarity-mrl-multilingual-v1` | [sentence-transformers/static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | multilingual, tuned for similarity rather than retrieval |     129 / 257 / 513 MiB |

</details>

<details>
<summary><b>NIFE (Nearly Inference Free Embeddings)</b></summary>

Static models aligned with a transformer teacher, so an index built with the teacher can be queried with the static model. English language only, licensed MIT.

| feature                     | model                                                                                                       | native width | teacher                                                                                         | size int8 / int16 / f32 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | -----------: | ----------------------------------------------------------------------------------------------- | ----------------------: |
| `nife-mxbai-embed-large-v1` | [stephantulkens/NIFE-mxbai-embed-large-v1](https://huggingface.co/stephantulkens/NIFE-mxbai-embed-large-v1) |         1024 | [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) |     129 / 257 / 513 MiB |
| `nife-gte-modernbert-base`  | [stephantulkens/NIFE-gte-modernbert-base](https://huggingface.co/stephantulkens/NIFE-gte-modernbert-base)   |          768 | [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)       |      97 / 193 / 385 MiB |

</details>

<details>
<summary><b>Other models</b></summary>

We try to cover the spread as far as static embedding models, but if you have a static embedding model not yet included in this crate, please let us know! We are happy to add support for it.

Otherwise, check out the [`ese-build`](./ese-build) and [`ese`](./ese) crates, which can be used to deploy private and local models.

</details>

## Methodology

The idea behind encoders (and the crate that underpins it, [`ese`](./ese)) is simple: static embedding models are essentially lookup tables from token IDs to weights. Why not just resolve the tokens to their weights with a compile-time map? So, that's exactly what it does—with a lot of other performance-oriented tricks to boot!

The workspace has four crates. `encoders` is the one above. [`ese`](./ese) is the runtime: a byte-for-byte reimplementation of the BERT tokenizer pipeline, mean pooling in integer lanes, and a perfect-hash table over the vocabulary. [`ese-build`](./ese-build) runs in [`build.rs`](./build.rs): it fetches a model, works out how it is laid out and how it expects to be pooled from its own files, quantizes it, and writes the tables. [`ese-unicode`](./ese-unicode) is the 64 KB of Unicode tables the tokenizer needs, generated from pinned crates and checked in.

### Caveats emptor

**Embedding quality.** Encoders only works with static embedding models: each token has one fixed vector and a text is the mean of its tokens' vectors. There is no attention, so word order and context are invisible, and quality is a step below even small transformer encoders. On retrieval benchmarks the gap is real but not enormous; the bekko family scores about 49 to 54 NanoBEIR against the low 60s for a small transformer. In exchange, embedding some text costs nanoseconds instead of a few milliseconds, with no GPU, no runtime dependencies, and no startup cost. For a lot of (if not most) search, dedup, clustering, and routing work, that is the right trade (and we hope the speed makes up for the rest).

**Binary size and compile time.** The model's lookup table is embedded in your executable: from a few MB for a small model at a narrow width to a few hundred MB for a large vocabulary at full width in f32. Any embedding model needs its weights at runtime somehow; this crate just puts them in the binary rather than downloading/loading them at runtime. The first build of a model downloads it from the Hugging Face hub (into `target/ese-cache`, so it's cleaned up by `cargo clean`) and the final link of a large table takes a while. You also _really, really need to compile this with optimizations on_... give LLVM a chance to autovectorize!

## Licensing

The code in this repo is licensed Apache-2.0. However, each model has its own license, which you must respect; it is available at build time in the summary and at run time as `encoders::model::LICENSE`.

[^1]: Any person wishing to dispute this claim is invited to challenge us to hand-to-hand combat at the our office in Brooklyn, New York, _OR_ create an issue on this repository referencing relevant benchmarks.
