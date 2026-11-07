//! Shared gooaq dataset loader for benches: the questions (short, query-like)
//! or the answers (paragraph-length, document-like).
#![allow(dead_code)]

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use std::fs;
use std::path::PathBuf;

const PARQUET_URLS: &[&str] = &[
    "https://huggingface.co/api/datasets/sentence-transformers/gooaq/parquet/pair/train/0.parquet",
    "https://huggingface.co/api/datasets/sentence-transformers/gooaq/parquet/pair/train/1.parquet",
];

/// The workspace `target/` directory. Benches run with the package directory
/// as cwd, and the shared caches (gooaq, ese-cache) live one level up.
pub fn target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target")
}

/// Panics unless encoders was compiled with `model` (the embedder benches
/// compare against model2vec-rs on the same weights).
pub fn require_model(model: &str) {
    assert_eq!(
        encoders::model::MODEL_ID,
        model,
        "this bench compares against {model}; run it with `--features potion-base-8m`",
    );
}

fn download_parquet(url: &str, path: &PathBuf) {
    let resp = minreq::get(url)
        .send()
        .unwrap_or_else(|e| panic!("failed to download {url}: {e}"));
    let bytes = resp.as_bytes();
    fs::write(path, bytes).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

/// The string fields of `path` whose column name is in `columns`.
fn extract_column(path: &PathBuf, columns: &[&str]) -> Vec<String> {
    let file = fs::File::open(path).expect("failed to open parquet file");
    let reader = SerializedFileReader::new(file).expect("failed to create parquet reader");
    let mut sentences = Vec::new();
    let iter = reader
        .get_row_iter(None)
        .expect("failed to get row iterator");
    for row in iter {
        let row = row.expect("failed to read row");
        for (name, field) in row.get_column_iter() {
            if columns.contains(&name.as_str())
                && let Field::Str(s) = field
            {
                sentences.push(s.clone());
            }
        }
    }
    sentences
}

/// Every gooaq question.
pub fn load_sentences() -> Vec<String> {
    load_column(&["question", "sentence1"], "questions")
}

/// Every gooaq answer.
pub fn load_answers() -> Vec<String> {
    load_column(&["answer", "sentence2"], "answers")
}

fn load_column(columns: &[&str], what: &str) -> Vec<String> {
    let cache_dir = target_dir().join("gooaq");
    fs::create_dir_all(&cache_dir).ok();

    let mut all_sentences = Vec::new();
    for (i, url) in PARQUET_URLS.iter().enumerate() {
        let cached = cache_dir.join(format!("gooaq_{i}.parquet"));
        if !cached.exists() {
            eprintln!("Downloading {url} ...");
            download_parquet(url, &cached);
        }
        all_sentences.extend(extract_column(&cached, columns));
    }
    eprintln!("Loaded {} {what} from gooaq dataset", all_sentences.len());
    all_sentences
}
