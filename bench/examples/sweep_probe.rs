//! Single-threaded embedding rate of whatever model/width/dtype encoders was
//! compiled with, for `py/sweep.py`. Prints one JSON line.
//!
//! Times `encode_single` over 4096 gooaq questions (query-like) and answers
//! (document-like), read one per line from the two files given as arguments
//! (`sweep.py` writes them once): repeated passes for about 1 s each after a
//! short warm-up, reporting the median pass. Build without `rayon`. Reading
//! plain files keeps parquet out of the binary, so each rebuild links fast.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Median texts/s and bytes/s over repeated passes, for `SWEEP_SECONDS`
/// (default 1) per text set; raise it when profiling.
fn rate(texts: &[String]) -> (f64, f64) {
    let secs: f64 = std::env::var("SWEEP_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let bytes: usize = texts.iter().map(String::len).sum();
    let pass = || {
        let t = Instant::now();
        for s in texts {
            black_box(encoders::encode_single(black_box(s.as_str())));
        }
        t.elapsed().as_secs_f64()
    };
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(200) {
        pass();
    }
    let mut times = Vec::new();
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs_f64(secs) || times.len() < 5 {
        times.push(pass());
    }
    times.sort_by(f64::total_cmp);
    let t = times[times.len() / 2];
    (texts.len() as f64 / t, bytes as f64 / t)
}

fn main() {
    let read = |path: String| -> Vec<String> {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .lines()
            .map(str::to_owned)
            .collect()
    };
    let mut args = std::env::args().skip(1);
    let (Some(q), Some(a)) = (args.next(), args.next()) else {
        panic!("usage: sweep_probe <questions.txt> <answers.txt>");
    };
    let (qps, q_bps) = rate(&read(q));
    let (dps, d_bps) = rate(&read(a));
    println!(
        "{{\"model\":{:?},\"dims\":{},\"vocab\":{},\"queries_per_s\":{qps:.1},\
         \"query_bytes_per_s\":{q_bps:.1},\"docs_per_s\":{dps:.1},\"doc_bytes_per_s\":{d_bps:.1}}}",
        encoders::model::MODEL_ID,
        encoders::model::DIMENSIONS,
        encoders::model::VOCAB_SIZE,
    );
}
