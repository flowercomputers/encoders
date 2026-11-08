//! Throughput (bytes/s of UTF-8 input) when each query is embedded on its
//! own, as in a search request path, over gooaq questions (about 43 bytes
//! each). The readme chart also shows an approximate queries/s, from the
//! text count and size recorded by `common::record_texts`. Mirrored for python by `py/qps.py`;
//! `corpus` is the batched, document-side counterpart.
//!
//! - `encoders`: `encode_single` at its default model, width and storage
//!   (not model2vec-rs's model; build with `--features potion-base-8m,float32`
//!   for a same-model comparison).
//! - `model2vec-rs`: `encode_single` with model2vec's default model
//!   (`minishlab/potion-base-8M`), downloaded through hf-hub on first run.
//!
//! Each iteration embeds all `POOL` questions one call at a time, so the
//! timing covers a spread of query lengths and the table rows aren't all
//! cache-hot. Single-text calls run on the calling thread in both crates, so
//! the `rayon` build has nothing to add and records nothing here.

use criterion::{Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use model2vec_rs::model::StaticModel;
use std::hint::black_box;
use std::time::Duration;

mod common;

use common::load_sentences;

const MODEL2VEC_DEFAULT: &str = "minishlab/potion-base-8M";

fn bench_queries(c: &mut Criterion) {
    if cfg!(feature = "rayon") {
        return;
    }
    // common::require_model(MODEL2VEC_DEFAULT);
    let pool = &load_sentences()[..65536];
    let bytes = common::record_texts("queries", pool);

    let mut group = c.benchmark_group("queries");
    group.throughput(Throughput::Bytes(bytes));
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function("encoders", |b| {
        b.iter(|| {
            for q in pool {
                black_box(encoders::encode_single(black_box(q.as_str())));
            }
        });
    });

    let model2vec = StaticModel::from_pretrained(MODEL2VEC_DEFAULT, None, None, None)
        .unwrap_or_else(|e| panic!("model2vec-rs failed to load {MODEL2VEC_DEFAULT}: {e}"));
    group.bench_function("model2vec-rs", |b| {
        b.iter(|| {
            for q in pool {
                black_box(model2vec.encode_single(black_box(q.as_str())));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_queries);
criterion_main!(benches);
