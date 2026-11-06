//! Sentences per second for three ways of getting a text embedding in Rust,
//! each encoding the same batch of gooaq questions through its own API:
//!
//! - `encoders`: this crate at its default settings, `encode` over the batch.
//!   With `--features rayon` this is recorded as `encoders-rayon` on every
//!   core and the other two are skipped.
//! - `model2vec-rs`: model2vec's default model (`minishlab/potion-base-8M`),
//!   downloaded through hf-hub on first run. Its only parallelism is the
//!   `tokenizers` batch encode, which runs here on every core in its own rayon
//!   pool, matching sentence-transformers, whose tokenizer is not limited by
//!   torch's thread setting either.
//! - `candle/all-MiniLM-L6-v2`: the usual small transformer baseline, run on
//!   CPU with candle in f32, mean pooled and normalized like
//!   sentence-transformers does. Weights are downloaded to `target/minilm`
//!   on first run. Outside the `rayon` build, encoders and candle are pinned
//!   to one thread.

use criterion::{Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use model2vec_rs::model::StaticModel;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

mod common;

/// Criterion id of the encoders bar; `--features rayon` records a second one.
const ENCODERS_ID: &str = if cfg!(feature = "rayon") { "encoders-rayon" } else { "encoders" };
use common::{load_sentences, target_dir};

const BATCH: usize = 4096;
const MODEL2VEC_DEFAULT: &str = "minishlab/potion-base-8M";

const MINILM_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// Sentences per forward pass; each chunk is padded to its longest member.
const MINILM_CHUNK: usize = 64;

fn minilm_file(name: &str) -> PathBuf {
    let dir = target_dir().join("minilm");
    fs::create_dir_all(&dir).ok();
    let path = dir.join(name);
    if !path.exists() {
        let url = format!("https://huggingface.co/{MINILM_REPO}/resolve/main/{name}");
        eprintln!("Downloading {url} ...");
        let resp = minreq::get(&url)
            .with_timeout(600)
            .send()
            .unwrap_or_else(|e| panic!("failed to download {url}: {e}"));
        assert_eq!(resp.status_code, 200, "{url}: HTTP {}", resp.status_code);
        fs::write(&path, resp.as_bytes()).unwrap();
    }
    path
}

struct MiniLm {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl MiniLm {
    fn load() -> Self {
        let device = Device::Cpu;
        let config: Config =
            serde_json::from_str(&fs::read_to_string(minilm_file("config.json")).unwrap()).unwrap();
        let mut tokenizer = Tokenizer::from_file(minilm_file("tokenizer.json")).unwrap();
        tokenizer
            .with_padding(Some(PaddingParams::default()))
            .with_truncation(Some(TruncationParams {
                max_length: 256,
                ..Default::default()
            }))
            .unwrap();
        let weights = minilm_file("model.safetensors");
        // SAFETY: the file is not modified while mapped.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[Path::new(&weights)], DTYPE, &device).unwrap()
        };
        let model = BertModel::load(vb, &config).unwrap();
        Self {
            model,
            tokenizer,
            device,
        }
    }

    /// Mean-pooled, L2-normalized embeddings, one row per sentence.
    fn embed(&self, texts: &[String]) -> Vec<Tensor> {
        texts
            .chunks(MINILM_CHUNK)
            .map(|chunk| {
                let enc = self.tokenizer.encode_batch(chunk.to_vec(), true).unwrap();
                let ids: Vec<&[u32]> = enc.iter().map(|e| e.get_ids()).collect();
                let mask: Vec<&[u32]> = enc.iter().map(|e| e.get_attention_mask()).collect();
                let ids = Tensor::new(ids, &self.device).unwrap();
                let mask = Tensor::new(mask, &self.device).unwrap();
                let types = ids.zeros_like().unwrap();
                let hidden = self.model.forward(&ids, &types, Some(&mask)).unwrap();
                let maskf = mask.to_dtype(DTYPE).unwrap().unsqueeze(2).unwrap();
                let summed = hidden.broadcast_mul(&maskf).unwrap().sum(1).unwrap();
                let counts = maskf.sum(1).unwrap();
                let mean = summed.broadcast_div(&counts).unwrap();
                let norm = mean.sqr().unwrap().sum_keepdim(1).unwrap().sqrt().unwrap();
                mean.broadcast_div(&norm).unwrap()
            })
            .collect()
    }
}

fn bench_embedders(c: &mut Criterion) {
    // candle's matmuls and the tokenizer both fan out over rayon; this must be
    // set before the global pool is first touched. The `rayon` build is the
    // one bar meant to use every core, so it is left alone there.
    if !cfg!(feature = "rayon") {
        // SAFETY: single-threaded at this point, no other thread reads the env.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", "1") };
    }
    let sentences = load_sentences();
    let batch: Vec<String> = sentences[..BATCH.min(sentences.len())].to_vec();
    let n = batch.len();

    let mut group = c.benchmark_group("embedders");
    group.throughput(Throughput::Elements(n as u64));
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    group.bench_function(ENCODERS_ID, |b| {
        b.iter(|| black_box(encoders::encode(black_box(&batch))));
    });
    if cfg!(feature = "rayon") {
        // Only the encoders bar differs between builds.
        group.finish();
        return;
    }

    let model2vec = StaticModel::from_pretrained(MODEL2VEC_DEFAULT, None, None, None)
        .unwrap_or_else(|e| panic!("model2vec-rs failed to load {MODEL2VEC_DEFAULT}: {e}"));
    // The global pool is pinned to one thread for candle; model2vec-rs gets
    // a pool of its own on every core. Rayon's parallel iterators (here the
    // `tokenizers` batch encode) run in whichever pool they are called from.
    let all_cores = rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get()))
        .build()
        .expect("failed to build the model2vec-rs thread pool");
    group.bench_function("model2vec-rs", |b| {
        b.iter(|| all_cores.install(|| black_box(model2vec.encode(black_box(&batch)))));
    });

    let minilm = MiniLm::load();
    group.bench_function("candle/all-MiniLM-L6-v2", |b| {
        b.iter(|| black_box(minilm.embed(black_box(&batch))));
    });

    group.finish();
}

criterion_group!(benches, bench_embedders);
criterion_main!(benches);
