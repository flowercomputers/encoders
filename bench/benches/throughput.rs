//! Throughput (bytes per second of UTF-8 input) of `encoders` next to a few
//! familiar per-byte workloads over the same 65,536 gooaq answers `corpus`
//! embeds, each driven the way its API expects. This bench records nothing in
//! the `rayon` build; the chart's parallel encoders bar is `corpus`'s
//! `encoders-rayon`, measured over that same batch.
//!
//! - `encoders`: `encode` over the whole slice on one thread.
//! - `tokenize`: the HuggingFace `tokenizers` crate running this model's own
//!   `tokenizer.json`, ids only, no embedding lookup.
//! - `sha256`: `sha2` with whatever hardware backend the CPU offers (SHA-NI,
//!   ARMv8 crypto extensions).
//! - `sha256_soft`: `sha2` 0.10 with `force-soft`, portable code only.
//! - `gzip`: `flate2` at the default level, sentences streamed newline-separated
//!   through one encoder into a sink (one gzip member per sentence would mostly
//!   measure deflate setup).
//! - `tantivy_index`: building an in-RAM index over all the sentences, one
//!   `TEXT` field, including the commit. tantivy's writer indexes on
//!   several threads.

use criterion::{Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use flate2::{Compression, write::GzEncoder};
use sha2::{Digest, Sha256};
use sha2_soft::Digest as _;
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;
use tantivy::schema::{Schema, TEXT};
use tantivy::{Index, doc};

mod common;

use common::{load_answers, target_dir};

/// The `tokenizer.json` ese-build downloaded for the compiled model. It sits
/// at the cache root or, for sentence-transformers layouts, one folder down.
fn tokenizer_path() -> PathBuf {
    let root = target_dir()
        .join("ese-cache")
        .join(encoders::model::MODEL_ID.replace('/', "--"))
        .join(encoders::model::MODEL_REVISION.replace('/', "--"));
    let direct = root.join("tokenizer.json");
    if direct.is_file() {
        return direct;
    }
    std::fs::read_dir(&root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path().join("tokenizer.json"))
        .find(|p| p.is_file())
        .unwrap_or_else(|| panic!("no tokenizer.json under {}", root.display()))
}

fn bench_throughput(c: &mut Criterion) {
    if cfg!(feature = "rayon") {
        // Nothing here depends on encoders' rayon feature.
        return;
    }
    let sentences = &load_answers()[..65536];
    let bytes: u64 = sentences.iter().map(|s| s.len() as u64).sum();
    eprintln!(
        "throughput: {} sentences, {bytes} bytes, mean {:.1} bytes/sentence",
        sentences.len(),
        bytes as f64 / sentences.len() as f64
    );

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Bytes(bytes));
    // Each bar gets a fixed window: a short warm-up, then ten samples spread
    // over 15 s. Flat sampling keeps criterion from scaling iteration counts
    // up when a single pass is already long.
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function("encoders", |b| {
        b.iter(|| black_box(encoders::encode(black_box(sentences))));
    });

    {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path())
            .unwrap_or_else(|e| panic!("tokenizers failed to load tokenizer.json: {e}"));
        group.bench_function("tokenize", |b| {
            b.iter(|| {
                for s in sentences {
                    let enc = tokenizer.encode_fast(black_box(s.as_str()), false).unwrap();
                    black_box(enc.get_ids());
                }
            });
        });
    }

    group.bench_function("sha256", |b| {
        b.iter(|| {
            for s in sentences {
                black_box(Sha256::digest(black_box(s).as_bytes()));
            }
        });
    });

    group.bench_function("sha256_soft", |b| {
        b.iter(|| {
            for s in sentences {
                black_box(sha2_soft::Sha256::digest(black_box(s).as_bytes()));
            }
        });
    });

    group.bench_function("gzip", |b| {
        b.iter(|| {
            let mut gz = GzEncoder::new(std::io::sink(), Compression::default());
            for s in sentences {
                gz.write_all(black_box(s).as_bytes()).unwrap();
                gz.write_all(b"\n").unwrap();
            }
            black_box(gz.finish().unwrap());
        });
    });

    {
        let mut schema = Schema::builder();
        let body = schema.add_text_field("body", TEXT);
        let schema = schema.build();
        let build_index = || {
            let index = Index::create_in_ram(schema.clone());
            let mut writer = index.writer(64 << 20).unwrap();
            for s in sentences {
                writer.add_document(doc!(body => s.as_str())).unwrap();
            }
            writer.commit().unwrap();
            index
        };

        group.bench_function("tantivy_index", |b| {
            b.iter(|| black_box(build_index()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
