## Much faster text embedding

This repo contains a rust crate that implements direct lookup of bytearray to token embedding, then mean pooling to determine the whole vector. 

At build time, we download the model and tokenizer from [sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1). This is then pulled apart and built into the rust binary. The crate exposes two functions, `encode` and `encode_single`, (take a look in `lib.rs` for more info)[./src/lib.rs].

### `encode`
This accepts an array of strings, and will run them all in parallel if your environment supports `rayon`. 

### `encode_single`
This accepts a single string

## To use

Clone the repo, and reference locally as a crate in your rust project. If you want to use a preliminary python wheel version [see here `./api-py/readme.md`](./api-py/readme.md).

### Crate features

#### Quantization and Truncation
By default we build an unquantized version that's truncated to 512 dimensions. To change this, change the feature setting in your cargo.toml for your `ese` dependency ([see `Cargo.toml` for all the options](./Cargo.toml)).

#### Rayon
We enable Rayon by default to enable parallel embedding. This dependency can be turned off.
