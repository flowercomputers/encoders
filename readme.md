## Much faster text embedding preview

This repo contains a rust crate containing the static embedding model deployment pipeline described [here](https://www.flowercomputer.com/news/fast-static-embedding/). 

`ese` is meant for embedding workloads that are too latency-sensitive or too
numerous to send over a network API. The interesting target is not only
index-time embedding for a vector database, but also application-runtime
embedding: small semantic comparisons, reranking passes, and agent tools that
need to run inside a hot loop.

The crate exposes two functions, `encode` and `encode_single`; take a look in
`lib.rs` for more info.

### `encode`
This accepts an array of strings, encoding them in parallel

### `encode_single`
This accepts a single string

## Embedding as a local primitive

Because the model is static and compiled into the crate output, embedding can
be cheap enough to call like ordinary application logic. That lets a host
application construct semantic queries at runtime instead of only querying
pre-built vector indexes.

For example, a caller can rank a small in-memory working set directly:

```rust
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut a_norm = 0.0;
    let mut b_norm = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        a_norm += x * x;
        b_norm += y * y;
    }

    dot / (a_norm.sqrt() * b_norm.sqrt())
}

let query = ese::encode_single("urgent technical bug or production issue");
let mut emails = vec![
    ("deploy failed", "production deploy is returning 500s"),
    ("lunch", "where should we eat today?"),
    ("invoice", "can you review the vendor invoice?"),
];

emails.sort_by(|(_, a), (_, b)| {
    let a_score = cosine(&ese::encode_single(a), &query);
    let b_score = cosine(&ese::encode_single(b), &query);
    b_score.total_cmp(&a_score)
});
```

For larger corpora, precompute and index embeddings as usual. The point is that
`ese` also makes the small, dynamic cases practical: rerank a page of search
results, score candidate memories, filter recent messages, or expose embedding
as a primitive inside an agent runtime without adding network latency, API
cost, or online availability assumptions.

## To use

Clone the repo, and reference locally as a crate in your rust project. If you want to use a preliminary python wheel version [see the documentation at `./api-py/readme.md`](./api-py/readme.md). You might need to `cargo install maturin`.

### Crate features

#### Quantization and Truncation
By default this crate provides an unquantized model truncated to 512 dimensions. To change this, change the feature setting in your cargo.toml for your `ese` dependency ([see `Cargo.toml` for all crate features](./Cargo.toml)).
