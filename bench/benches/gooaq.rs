use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

mod common;
use common::load_sentences;

fn bench_encode_single(c: &mut Criterion) {
    let sentences = load_sentences();
    assert!(!sentences.is_empty(), "dataset is empty");

    let mut group = c.benchmark_group("encode_single");
    group.throughput(Throughput::Elements(1));

    group.bench_function("short", |b| {
        let s = &sentences[0];
        b.iter(|| encoders::encode_single(std::hint::black_box(s)));
    });

    let longest = sentences.iter().max_by_key(|s| s.len()).unwrap();
    group.bench_with_input(
        BenchmarkId::new("longest", format!("{}chars", longest.len())),
        longest,
        |b, s| {
            b.iter(|| encoders::encode_single(std::hint::black_box(s)));
        },
    );

    group.finish();
}

fn bench_encode_batch(c: &mut Criterion) {
    let sentences = load_sentences();
    let n = sentences.len();

    let mut group = c.benchmark_group("encode_batch");

    for size in [1, 16, 64, 256, 1000, 10_000, 65_536, 100_000, 1_000_000] {
        if size > n {
            continue;
        }
        let batch: Vec<_> = sentences[..size].to_vec();
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &batch, |b, batch| {
            b.iter(|| encoders::encode(std::hint::black_box(batch.iter())));
        });
    }

    group.finish();
}

fn bench_encode_by_length(c: &mut Criterion) {
    let sentences = load_sentences();
    let mut group = c.benchmark_group("encode_by_length");

    // Build inputs of increasing character length
    for &char_len in &[10, 20, 36, 50, 100, 250] {
        // Find a sentence at least this long, or skip
        let Some(s) = sentences.iter().find(|s| s.len() >= char_len) else {
            continue;
        };
        let input: String = s.chars().take(char_len).collect();
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_with_input(BenchmarkId::new("chars", char_len), &input, |b, input| {
            b.iter(|| encoders::encode_single(std::hint::black_box(input)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_single,
    bench_encode_batch,
    bench_encode_by_length
);

criterion_main!(benches);
