//! Throughput (bytes/s of UTF-8 input) when indexing a corpus: one batch of
//! gooaq answers (paragraph-length, about 253 bytes each, unlike the
//! question-sized `queries` bench) through each library's batch API. The
//! readme chart also shows an approximate documents/s, from the text count
//! and size recorded by `common::record_texts`. Mirrored for python by `py/qps.py --corpus`.
//! The throughput chart reuses the encoders bar from here.
//!
//! - `encoders-rayon`: `encode` over the batch on every core, recorded only
//!   with `--features rayon` (which skips model2vec-rs). The single-threaded
//!   batch encode is the `throughput` bench's encoders bar.
//! - `model2vec-rs`: model2vec's default model (`minishlab/potion-base-8M`,
//!   256 dimensions), downloaded through hf-hub on first run. Its `encode`
//!   tokenizes and pools each document in a rayon parallel iterator, so it
//!   runs on every core.

use criterion::{Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use model2vec_rs::model::StaticModel;
use std::hint::black_box;
use std::time::Duration;

mod common;

use common::load_answers;

const MODEL2VEC_DEFAULT: &str = "minishlab/potion-base-8M";

fn bench_corpus(c: &mut Criterion) {
    // common::require_model(MODEL2VEC_DEFAULT);
    let batch = &load_answers()[..65536];
    let bytes = common::record_texts("corpus", batch);

    let mut group = c.benchmark_group("corpus");
    group.throughput(Throughput::Bytes(bytes));
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    if cfg!(feature = "rayon") {
        // The rayon build adds only the parallel encoders bar.
        group.bench_function("encoders-rayon", |b| {
            b.iter(|| black_box(encoders::encode(black_box(batch))));
        });
        group.finish();
        return;
    }

    let model2vec = StaticModel::from_pretrained(MODEL2VEC_DEFAULT, None, None, None)
        .unwrap_or_else(|e| panic!("model2vec-rs failed to load {MODEL2VEC_DEFAULT}: {e}"));
    group.bench_function("model2vec-rs", |b| {
        b.iter(|| black_box(model2vec.encode_with_args(black_box(batch), None, batch.len())));
    });

    group.finish();
}

criterion_group!(benches, bench_corpus);
criterion_main!(benches);
