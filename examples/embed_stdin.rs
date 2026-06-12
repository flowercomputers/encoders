// embed_stdin: read UTF-8 lines from stdin, print one embedding per line.
//
// Each output line is the embedding for the corresponding input line,
// printed as space-separated f32 values. Empty lines produce zero vectors.
//
// Usage:
//   echo "hello world" | cargo run --example embed_stdin
//   cat sentences.txt | ESE_MODEL_DIR=distill/out/bge-base \
//       ESE_NORMALIZE_OUTPUT=1 cargo run --example embed_stdin --features dim-256
//
// This example doubles as the cross-language comparison harness: pipe the
// same sentences through here and through a Python model2vec encoder, then
// check cosine similarity between the outputs.

use std::io::{self, BufRead};

fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let text = line.expect("failed to read line from stdin");
        let embedding = ese::encode_single(&text);
        let parts: Vec<String> = embedding.iter().map(|v| v.to_string()).collect();
        println!("{}", parts.join(" "));
    }
}
