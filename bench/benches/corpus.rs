//! Documents per second when indexing a corpus: one batch of gooaq answers
//! (paragraph-length, unlike the question-sized `queries` bench) through each
//! library's batch API. Mirrored for python by `py/qps.py --corpus`.
//!
//! - `encoders`: `encode` over the batch on one thread, built with
//!   `--features potion-base-8m` (the same model as model2vec-rs, stored as
//!   float32). With `--features rayon` this is recorded as
//!   `encoders-rayon` on every core and model2vec-rs is skipped.
//! - `model2vec-rs`: model2vec's default model (`minishlab/potion-base-8M`,
//!   256 dimensions), downloaded through hf-hub on first run. Its `encode`
//!   tokenizes and pools each document in a rayon parallel iterator, so it
//!   runs on every core.

use criterion::{Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use model2vec_rs::model::StaticModel;
use std::hint::black_box;
use std::time::Duration;

mod common;

/// Criterion id of the encoders bar; `--features rayon` records a second one.
const ENCODERS_ID: &str = if cfg!(feature = "rayon") {
    "encoders-rayon"
} else {
    "encoders"
};
use common::load_answers;

const MODEL2VEC_DEFAULT: &str = "minishlab/potion-base-8M";

fn bench_corpus(c: &mut Criterion) {
    // common::require_model(MODEL2VEC_DEFAULT);
    let batch = &load_answers()[..65536];
    let n = batch.len();

    let mut group = c.benchmark_group("corpus");
    group.throughput(Throughput::Elements(n as u64));
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function(ENCODERS_ID, |b| {
        b.iter(|| black_box(encoders::encode(black_box(batch))));
    });
    if cfg!(feature = "rayon") {
        // Only the encoders bar differs between builds.
        group.finish();
        return;
    }

    let model2vec = StaticModel::from_pretrained(MODEL2VEC_DEFAULT, None, None, None)
        .unwrap_or_else(|e| panic!("model2vec-rs failed to load {MODEL2VEC_DEFAULT}: {e}"));
    group.bench_function("model2vec-rs", |b| {
        b.iter(|| black_box(model2vec.encode_with_args(black_box(&batch), None, batch.len())));
    });

    group.finish();
}

criterion_group!(benches, bench_corpus);
criterion_main!(benches);
