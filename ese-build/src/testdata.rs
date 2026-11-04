//! Reference embeddings computed with the `tokenizers` crate using the same
//! pooling policy that was detected for the model. The consumer's tests
//! compare `encode_single` against these, which makes them the fidelity
//! oracle for the runtime's hand-written tokenizer.

use std::{fs, path::Path};

use crate::{Error, Quant, emit::QuantInfo};

/// Inputs chosen to exercise each policy: normalization, accents, CJK
/// padding, punctuation splitting, over-long words, all-unknown text, empty
/// text and literal special tokens (matched as single added tokens, which
/// the runtime mirrors; `[sep]` in lowercase must not match).
const INPUTS: &[&str] = &[
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "",
    "   ",
    "café résumé naïve",
    "Zürich ÅNGSTRÖM straße",
    "你好世界",
    "日本語のテキストと English mixed",
    "Hello, world! How are you?",
    "UPPER CASE text",
    "multiple   spaces   between",
    "a]b[c{d}e",
    "\x00\x01\x02\x03",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "pneumonoultramicroscopicsilicovolcanoconiosispneumonoultramicroscopicsilicovolcanoconiosispneumonoultramicroscopicsilicovolcanoconiosis",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝔣𝔯𝔞𝔨𝔱𝔲𝔯",
    "hello 😀 world 🦀",
    "tabs\tand\nnewlines\r\nhere",
    "e-mail john.doe@example.com http://example.com/path?q=1",
    "1234567890 3.14159 -42",
    "hello [SEP] world",
    "[CLS] what is this? [SEP]",
    "x[UNK]y and a [MASK] token [PAD]",
    "[sep] [ CLS ] [UNK",
    "co\u{AD}operate zero\u{200B}width fam\u{200D}ily \u{FEFF}bom \u{E000}private",
];

#[allow(clippy::too_many_arguments)]
pub(crate) fn write(
    out: &Path,
    tokenizer_path: &Path,
    rows: &[Vec<f32>],
    dims: usize,
    unk_id: usize,
    drop_unk: bool,
    normalize: bool,
    quant: Quant,
    qi: &QuantInfo,
) -> Result<(), Error> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| Error::Invalid(format!("tokenizers failed to load the tokenizer: {e}")))?;

    // A text past 255 tokens exercises the quantized accumulator's spill path.
    let long = INPUTS[1].repeat(40);
    let inputs: Vec<&str> = INPUTS.iter().copied().chain([long.as_str()]).collect();

    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    // Norm of each mean before the optional L2 normalization, and the RMS
    // quantization error expected on that mean: per pooled row, uniform noise
    // of (step * 2^-exp) / sqrt(12) per column, averaged over the rows. An
    // input whose mean is within a few of these is noise even in f32, so the
    // consumer's test skips its direction check.
    let mut ref_norms: Vec<f32> = Vec::with_capacity(inputs.len());
    let mut ref_noise: Vec<f32> = Vec::with_capacity(inputs.len());
    let col_var: f32 = qi.col_step.iter().map(|s| s * s / 12.0).sum();
    for &input in &inputs {
        let enc = tokenizer
            .encode(input, false)
            .map_err(|e| Error::Invalid(format!("tokenizers failed on {input:?}: {e}")))?;
        let ids: Vec<usize> = enc
            .get_ids()
            .iter()
            .map(|&id| id as usize)
            .filter(|&id| !(drop_unk && id == unk_id))
            .collect();
        let mut v = vec![0.0f32; dims];
        for &id in &ids {
            for (x, w) in v.iter_mut().zip(&rows[id]) {
                *x += w;
            }
        }
        if !ids.is_empty() {
            let inv = 1.0 / ids.len() as f32;
            v.iter_mut().for_each(|x| *x *= inv);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        ref_norms.push(norm);
        let noise_sq: f32 = ids
            .iter()
            .map(|&id| {
                let f = 1.0 / (1u32 << qi.row_exp[id]) as f32;
                col_var * f * f
            })
            .sum();
        ref_noise.push(if ids.is_empty() {
            0.0
        } else {
            noise_sq.sqrt() / ids.len() as f32
        });
        if normalize {
            let norm = norm.max(1e-12);
            v.iter_mut().for_each(|x| *x /= norm);
        }
        outputs.push(v);
    }

    // This is a pipeline check, not a quality bar. Multi-token inputs sit at
    // cosine >= 0.9999 under per-column 8-bit, but a single small-norm token
    // (an [UNK] on the sentence-transformers models) can drop to ~0.99;
    // 16-bit is indistinguishable from f32.
    let cosine_min = match quant {
        Quant::F32 => 0.999999,
        Quant::U16 => 0.99999,
        Quant::U8 => 0.99,
    };

    let mut code = String::from("pub const TEST_INPUTS: &[&str] = &[\n");
    for input in &inputs {
        code.push_str(&format!("    {input:?},\n"));
    }
    code.push_str("];\n\n");
    code.push_str(&format!(
        "pub const TEST_OUTPUTS: &[[f32; {dims}]; {}] = &[\n",
        outputs.len()
    ));
    for v in &outputs {
        code.push_str(&format!("    {:?},\n", v.as_slice()));
    }
    code.push_str("];\n\n");
    code.push_str(&format!(
        "pub const TEST_COSINE_MIN: f32 = {cosine_min:?};\n"
    ));
    code.push_str(&format!(
        "pub const TEST_REF_NORMS: &[f32] = &{ref_norms:?};\n"
    ));
    code.push_str(&format!(
        "pub const TEST_REF_NOISE: &[f32] = &{ref_noise:?};\n"
    ));
    fs::write(out.join("optional_testdata.rs"), code)?;
    Ok(())
}
