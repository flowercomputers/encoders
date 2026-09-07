## Much faster text embedding preview

This repo contains a rust crate containing the static embedding model deployment pipeline described [here](https://www.flowercomputer.com/news/fast-static-embedding/). 

The crate exposes two functions, `encode` and `encode_single`, take a look in `lib.rs` for more info.

### `encode`
This accepts an array of strings, encoding them in parallel

### `encode_single`
This accepts a single string

## To use

Clone the repo, and reference locally as a crate in your rust project. If you want to use a preliminary python wheel version [see the documentation at `./api-py/readme.md`](./api-py/readme.md). You might need to `cargo install maturin`.

### Crate features

#### Quantization and Truncation
By default this crate provides an unquantized model truncated to 512 dimensions. To change this, change the feature setting in your cargo.toml for your `ese` dependency ([see `Cargo.toml` for all crate features](./Cargo.toml)).

#### Model selection
By default the crate compiles [`sentence-transformers/static-retrieval-mrl-en-v1`](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1). Enable the `model-potion-code-16m-v2` feature to compile [`minishlab/potion-code-16M-v2`](https://huggingface.co/minishlab/potion-code-16M-v2), a code-retrieval model2vec model, instead (it is 256-dimensional natively, so `dim-*` features above 256 have no effect).

For model2vec models, ese reproduces the reference inference semantics exactly (UNK tokens dropped before pooling, 512-token truncation, L2-normalized output; cosine = 1.0 against `model2vec-rs`, see `tests/m2v_parity.rs`). One deliberate difference: ese encodes each input independently, so results never depend on what else is in the batch — the reference implementations pool `[PAD]` tokens into shorter sequences when batching, ese matches their single-input output.

You can also point the build at any local model directory containing a `model.safetensors` and `tokenizer.json` (plus an optional model2vec `config.json`) with the `ESE_MODEL_DIR` environment variable. Only BERT-style WordPiece tokenizers are supported; the build fails with an explanatory error for anything else (e.g. Unigram-based models like `potion-multilingual-128M`).

Note: the `dim-*` truncation is only principled for Matryoshka-trained models like the default one; for other models it just clamps to the model's native dimensionality.
