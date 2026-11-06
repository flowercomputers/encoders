//! Offline detection tests on synthetic model directories.

use std::{
    fs,
    path::{Path, PathBuf},
};

use ese_build::{Builder, Error, Quant};

const VOCAB: &[&str] = &[
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##ld", "wor", "!",
];
const DIMS: usize = 4;

fn wordpiece_tokenizer() -> String {
    let vocab: Vec<String> = VOCAB
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{t:?}: {i}"))
        .collect();
    format!(
        r###"{{
  "added_tokens": [
    {{"id": 0, "content": "[PAD]", "special": true}},
    {{"id": 1, "content": "[UNK]", "special": true}}
  ],
  "normalizer": {{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}},
  "pre_tokenizer": {{"type": "BertPreTokenizer"}},
  "model": {{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {{{}}}}}
}}"###,
        vocab.join(", ")
    )
}

fn unigram_tokenizer() -> &'static str {
    r#"{"normalizer": {"type": "Precompiled", "precompiled_charsmap": ""}, "pre_tokenizer": {"type": "Metaspace"}, "model": {"type": "Unigram", "vocab": [["<unk>", 0.0]]}}"#
}

/// A safetensors file with `name` as an F32 [rows, DIMS] tensor whose row r
/// is filled with r+1, plus optional 1-D extras.
fn safetensors(name: &str, rows: usize, extra: &[(&str, &str, Vec<u8>, usize)]) -> Vec<u8> {
    let mut body: Vec<u8> = Vec::new();
    for r in 0..rows {
        for _ in 0..DIMS {
            body.extend_from_slice(&((r + 1) as f32).to_le_bytes());
        }
    }
    let mut header = format!(
        r#"{{"{name}":{{"dtype":"F32","shape":[{rows},{DIMS}],"data_offsets":[0,{}]}}"#,
        body.len()
    );
    for (ename, dtype, bytes, len) in extra {
        let start = body.len();
        body.extend_from_slice(bytes);
        header.push_str(&format!(
            r#","{ename}":{{"dtype":"{dtype}","shape":[{len}],"data_offsets":[{start},{}]}}"#,
            body.len()
        ));
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&body);
    out
}

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ese-build-{tag}-{}", std::process::id()));
        fs::remove_dir_all(&p).ok();
        fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn write(&self, rel: &str, data: impl AsRef<[u8]>) -> &Self {
        let p = self.0.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, data).unwrap();
        self
    }
    fn out(&self) -> PathBuf {
        let o = self.0.join("out");
        fs::create_dir_all(&o).unwrap();
        o
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

fn constants(out: &Path) -> String {
    fs::read_to_string(out.join("model_constants.rs")).unwrap()
}

#[test]
fn sentence_transformers_layout() {
    let d = Dir::new("st");
    d.write(
        "modules.json",
        r#"[{"idx":0,"name":"0","path":"0_StaticEmbedding","type":"sentence_transformers.models.StaticEmbedding"}]"#,
    )
    .write("0_StaticEmbedding/tokenizer.json", wordpiece_tokenizer())
    .write(
        "0_StaticEmbedding/model.safetensors",
        safetensors("embedding.weight", VOCAB.len(), &[]),
    );
    let s = Builder::local(&d.0).generate(&d.out()).unwrap();
    assert_eq!(s.framework, "sentence-transformers");
    assert!(!s.drop_unk && !s.normalize);
    assert_eq!(
        (s.dims, s.native_dims, s.vocab_size, s.table_size),
        (4, 4, 9, 16)
    );
    let c = constants(&d.out());
    assert!(c.contains("pub const DROP_UNK: bool = false;"));
    assert!(c.contains("pub const NORMALIZE: bool = false;"));
    assert!(c.contains("pub const LOWERCASE: bool = true;"));
    assert!(c.contains("pub const STRIP_ACCENTS: bool = true;"));
    assert!(c.contains(r#"pub const SPECIAL_TOKENS: &[&str] = &["[PAD]", "[UNK]"];"#));
    // unk is row 1, filled with 2.0
    assert!(c.contains("pub const UNK: Param = [2.0, 2.0, 2.0, 2.0];"));
    // 4 f32 = 16 bytes per row, padded to one cache line
    assert!(c.contains("pub const ROW_STRIDE: usize = 64;"));
    assert_eq!(
        fs::metadata(d.out().join("weights.bin")).unwrap().len(),
        16 * 64
    );
    assert_eq!(
        fs::metadata(d.out().join("verify.bin")).unwrap().len(),
        16 * 8
    );
}

#[test]
fn model2vec_layout_with_normalize_and_vocab_quantization() {
    let d = Dir::new("m2v");
    let mapping: Vec<u8> = [0i32, 1, 2, 3, 0, 1, 2, 3, 0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let weights: Vec<u8> = (0..9)
        .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
        .collect();
    d.write(
        "modules.json",
        r#"[{"idx":0,"name":"0","path":".","type":"sentence_transformers.models.StaticEmbedding"},{"idx":1,"name":"1","path":"1_Normalize","type":"sentence_transformers.models.Normalize"}]"#,
    )
    .write("config.json", r#"{"model_type":"model2vec","normalize":true}"#)
    .write("tokenizer.json", wordpiece_tokenizer())
    .write(
        "model.safetensors",
        safetensors(
            "embeddings",
            4,
            &[("mapping", "I32", mapping, 9), ("weights", "F32", weights, 9)],
        ),
    );
    let s = Builder::local(&d.0)
        .quant(Quant::U8)
        .generate(&d.out())
        .unwrap();
    assert_eq!(s.framework, "model2vec");
    assert!(s.drop_unk && s.normalize);
    assert_eq!(s.quant, Quant::U8);
    let c = constants(&d.out());
    assert!(c.contains("pub type Elem = u8;"));
    assert!(c.contains("pub const DROP_UNK: bool = true;"));
    assert!(c.contains("pub const NORMALIZE: bool = true;"));
    // u8 rows of 4 bytes still occupy one cache line each
    assert!(c.contains("pub const ROW_STRIDE: usize = 64;"));
    assert_eq!(
        fs::metadata(d.out().join("weights.bin")).unwrap().len(),
        16 * 64
    );
}

#[test]
fn model2vec_detected_without_model_type() {
    // Newer model2vec exports (potion-code-16M) omit `model_type`.
    let d = Dir::new("m2v-new");
    d.write(
        "modules.json",
        r#"[{"idx":0,"name":"0","path":".","type":"sentence_transformers.models.StaticEmbedding"}]"#,
    )
    .write(
        "config.json",
        r#"{"normalize": true, "embedding_dtype": "float32", "vocabulary_quantization": 9}"#,
    )
    .write("tokenizer.json", wordpiece_tokenizer())
    .write("model.safetensors", safetensors("embeddings", VOCAB.len(), &[]));
    let s = Builder::local(&d.0).generate(&d.out()).unwrap();
    assert_eq!(s.framework, "model2vec");
    assert!(s.drop_unk);
    assert!(
        !s.normalize,
        "Normalize module absent and modules.json present"
    );
}

#[test]
fn truncation_and_clamping() {
    let d = Dir::new("dims");
    d.write("tokenizer.json", wordpiece_tokenizer()).write(
        "model.safetensors",
        safetensors("embeddings", VOCAB.len(), &[]),
    );
    let s = Builder::local(&d.0)
        .dims(Some(2))
        .generate(&d.out())
        .unwrap();
    assert_eq!((s.dims, s.native_dims), (2, 4));
    assert!(constants(&d.out()).contains("pub const UNK: Param = [2.0, 2.0];"));
    assert!(constants(&d.out()).contains("pub const TRUNCATION_POINTS: &[usize] = &[];"));
    assert!(constants(&d.out()).contains("pub const LICENSE: Option<&str> = None;"));
    assert!(constants(&d.out()).contains("pub const PARALLEL_MIN: usize = 128;"));
    // wider than the model is an error
    assert!(matches!(
        Builder::local(&d.0).dims(Some(64)).generate(&d.out()),
        Err(Error::Invalid(_))
    ));
    // explicit training widths are enforced and emitted
    let s = Builder::local(&d.0)
        .dims(Some(2))
        .truncation_points(Some(&[4, 2]))
        .license("MIT")
        .multilingual(true)
        .parallel_min(7)
        .generate(&d.out())
        .unwrap();
    assert_eq!(s.truncation_points, [4, 2]);
    let c = constants(&d.out());
    assert!(c.contains("pub const LICENSE: Option<&str> = Some(\"MIT\");"));
    assert!(c.contains("pub const MULTILINGUAL: bool = true;"));
    assert!(c.contains("pub const PARALLEL_MIN: usize = 7;"));
    assert!(c.contains("pub const NATIVE_DIMENSIONS: usize = 4;"));
    assert!(c.contains("pub const TRUNCATION_POINTS: &[usize] = &[4, 2];"));
    assert!(c.contains("pub const MATRYOSHKA: bool = true;"));
    match Builder::local(&d.0)
        .dims(Some(3))
        .truncation_points(Some(&[4, 2]))
        .generate(&d.out())
    {
        Err(Error::Invalid(msg)) => assert!(msg.contains("[2]"), "{msg}"),
        other => panic!("expected Invalid, got {other:?}"),
    }
    match Builder::local(&d.0)
        .dims(Some(2))
        .truncation_points(Some(&[4]))
        .generate(&d.out())
    {
        Err(Error::Invalid(msg)) => assert!(msg.contains("not Matryoshka"), "{msg}"),
        other => panic!("expected Invalid, got {other:?}"),
    }
}

#[test]
fn small_rows_get_exponents() {
    // Row 1 ([UNK]) is 1/1000 of a typical row: it must be scaled up (by up
    // to 2^8) rather than collapse to a constant under 8-bit quantization.
    let d = Dir::new("exp");
    let mut body: Vec<u8> = Vec::new();
    for r in 0..VOCAB.len() {
        let mag = if r == 1 { 0.001 } else { 1.0 + r as f32 * 0.1 };
        for j in 0..DIMS {
            let sign = if j % 2 == 0 { 1.0 } else { -0.5 };
            body.extend_from_slice(&(mag * sign).to_le_bytes());
        }
    }
    let header = format!(
        r#"{{"embeddings":{{"dtype":"F32","shape":[{},{DIMS}],"data_offsets":[0,{}]}}}}"#,
        VOCAB.len(),
        body.len()
    );
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header.as_bytes());
    st.extend_from_slice(&body);
    d.write("tokenizer.json", wordpiece_tokenizer())
        .write("model.safetensors", st);
    Builder::local(&d.0)
        .quant(Quant::U8)
        .generate(&d.out())
        .unwrap();
    // The row exponent is the low byte of each slot's verifier.
    let verify = fs::read(d.out().join("verify.bin")).unwrap();
    let exps: Vec<u8> = verify.as_chunks::<8>().0.iter().map(|v| v[0]).collect();
    assert_eq!(exps.len(), 16);
    let c = constants(&d.out());
    assert!(c.contains("pub const UNK_EXP: u8 = 8;"), "{c}");
    assert!(c.contains("pub const ROW_EXP_MAX: u8 = 8;"));
    assert_eq!(exps.iter().filter(|&&e| e == 8).count(), 1, "{exps:?}");
}

#[test]
fn rejects_non_static_modules() {
    let d = Dir::new("dense");
    d.write(
        "modules.json",
        r#"[{"idx":0,"name":"0","path":".","type":"sentence_transformers.models.StaticEmbedding"},{"idx":1,"name":"1","path":"1_Dense","type":"sentence_transformers.models.Dense"}]"#,
    );
    match Builder::local(&d.0).generate(&d.out()) {
        Err(Error::NotStatic(msg)) => assert!(msg.contains("Dense"), "{msg}"),
        other => panic!("expected NotStatic, got {other:?}"),
    }
}

#[test]
fn rejects_unigram_tokenizer() {
    let d = Dir::new("unigram");
    d.write("tokenizer.json", unigram_tokenizer())
        .write("model.safetensors", safetensors("embeddings", 1, &[]));
    match Builder::local(&d.0).generate(&d.out()) {
        Err(Error::UnsupportedTokenizer(msg)) => assert!(msg.contains("Precompiled"), "{msg}"),
        other => panic!("expected UnsupportedTokenizer, got {other:?}"),
    }
}

#[test]
fn rejects_row_count_mismatch_without_mapping() {
    let d = Dir::new("rows");
    d.write("tokenizer.json", wordpiece_tokenizer())
        .write("model.safetensors", safetensors("embeddings", 5, &[]));
    match Builder::local(&d.0).generate(&d.out()) {
        Err(Error::UnsupportedTensor(msg)) => assert!(msg.contains("5 rows"), "{msg}"),
        other => panic!("expected UnsupportedTensor, got {other:?}"),
    }
}

#[test]
fn missing_files_are_reported() {
    let d = Dir::new("missing");
    d.write("tokenizer.json", wordpiece_tokenizer());
    match Builder::local(&d.0).generate(&d.out()) {
        Err(Error::MissingFile(msg)) => assert!(msg.contains("model.safetensors"), "{msg}"),
        other => panic!("expected MissingFile, got {other:?}"),
    }
}
