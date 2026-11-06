//! Shared gooaq dataset loader for benches.

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use std::fs;
use std::path::PathBuf;

const PARQUET_URLS: &[&str] = &[
    "https://huggingface.co/api/datasets/sentence-transformers/gooaq/parquet/pair/train/0.parquet",
    "https://huggingface.co/api/datasets/sentence-transformers/gooaq/parquet/pair/train/1.parquet",
];

const CACHE_DIR: &str = "target/ese-cache/dataset-gooaq";

fn download_parquet(url: &str, path: &PathBuf) {
    let resp = minreq::get(url)
        .send()
        .unwrap_or_else(|e| panic!("failed to download {url}: {e}"));
    let bytes = resp.as_bytes();
    fs::write(path, bytes).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

fn extract_questions(path: &PathBuf) -> Vec<String> {
    let file = fs::File::open(path).expect("failed to open parquet file");
    let reader = SerializedFileReader::new(file).expect("failed to create parquet reader");
    let mut sentences = Vec::new();
    let iter = reader
        .get_row_iter(None)
        .expect("failed to get row iterator");
    for row in iter {
        let row = row.expect("failed to read row");
        for (name, field) in row.get_column_iter() {
            if (name == "question" || name == "sentence1")
                && let Field::Str(s) = field
            {
                sentences.push(s.clone());
            }
        }
    }
    sentences
}

pub fn load_sentences() -> Vec<String> {
    let cache_dir = PathBuf::from(CACHE_DIR);
    fs::create_dir_all(&cache_dir).ok();

    let mut all_sentences = Vec::new();
    for (i, url) in PARQUET_URLS.iter().enumerate() {
        let cached = cache_dir.join(format!("gooaq_{i}.parquet"));
        if !cached.exists() {
            eprintln!("Downloading {url} ...");
            download_parquet(url, &cached);
        }
        all_sentences.extend(extract_questions(&cached));
    }
    eprintln!(
        "Loaded {} sentences from gooaq dataset",
        all_sentences.len()
    );
    all_sentences
}
