//! One throughput number (bytes per second of UTF-8 input) for `encoders` at
//! its default settings next to a few familiar per-byte workloads, all over the
//! same gooaq question set. Every sentence is processed individually, so
//! per-call overhead is included for everything.
//!
//! - `encoders`: `encode_into` with a reused output buffer.
//! - `tokenize`: the HuggingFace `tokenizers` crate running this model's own
//!   `tokenizer.json`, ids only, no embedding lookup.
//! - `sha256_soft`: `sha2` with `force-soft`, no SHA-NI / ARMv8 crypto.
//! - `gzip`: `flate2` at the default level, sentences streamed newline-separated
//!   through one encoder into a sink (one gzip member per sentence would mostly
//!   measure deflate setup).
//! - `tantivy_search`: each sentence parsed as a query and run for the top 10
//!   hits against an in-RAM index of the same sentences. Index construction is
//!   outside the timed region; only query parsing and search are measured.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use flate2::{Compression, write::GzEncoder};
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, TEXT};
use tantivy::{Index, doc};

mod common;
use common::load_sentences;

/// The `tokenizer.json` ese-build downloaded for the compiled model. It sits
/// at the cache root or, for sentence-transformers layouts, one folder down.
fn tokenizer_path() -> PathBuf {
    let root = PathBuf::from("target/ese-cache")
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
    let sentences = &load_sentences()[..65536];
    let bytes: u64 = sentences.iter().map(|s| s.len() as u64).sum();
    eprintln!(
        "throughput: {} sentences, {bytes} bytes, mean {:.1} bytes/sentence",
        sentences.len(),
        bytes as f64 / sentences.len() as f64
    );

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Bytes(bytes));
    group.sample_size(100);

    group.bench_function("encoders", |b| {
        let mut out = [0f32; encoders::DIMENSIONS];
        b.iter(|| {
            for s in sentences {
                encoders::encode_into(black_box(s), &mut out);
                black_box(&out);
            }
        });
    });

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

    group.bench_function("sha256_soft", |b| {
        b.iter(|| {
            for s in sentences {
                black_box(Sha256::digest(black_box(s).as_bytes()));
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

    let mut schema = Schema::builder();
    let body = schema.add_text_field("body", TEXT);
    let index = Index::create_in_ram(schema.build());
    {
        let mut writer = index.writer(64 << 20).unwrap();
        for s in sentences {
            writer.add_document(doc!(body => s.as_str())).unwrap();
        }
        writer.commit().unwrap();
    }
    let searcher = index.reader().unwrap().searcher();
    let parser = QueryParser::for_index(&index, vec![body]);
    group.bench_function("tantivy_search", |b| {
        b.iter(|| {
            let mut hits = 0usize;
            for s in sentences {
                let (query, _) = parser.parse_query_lenient(black_box(s));
                hits += searcher.search(&query, &TopDocs::with_limit(10).order_by_score()).unwrap().len();
            }
            black_box(hits)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
