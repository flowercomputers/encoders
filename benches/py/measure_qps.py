"""
measure_qps.py — QPS benchmarks for ese models via embed_stdin.

Measures single-sentence and batched QPS for ≥2 ese models.
Tags the machine hardware.

Usage:
    python measure_qps.py [model_key ...]
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ese_encoder import get_model_configs, REPO

BENCH_SECONDS = 10
BATCH_SIZE = 32
WARMUP_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models encode semantic meaning.",
]


def get_machine_info() -> str:
    info = platform.uname()
    cpu = info.processor or info.machine
    # Try to get a better CPU name on macOS
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            cpu = result.stdout.strip()
    except Exception:
        pass
    return f"{info.system} {info.release} | CPU: {cpu}"


def _make_env(cfg) -> dict:
    env = os.environ.copy()
    env["ESE_MODEL_DIR"] = str(cfg.artifact_dir.resolve())
    if cfg.normalize:
        env["ESE_NORMALIZE_OUTPUT"] = "1"
    else:
        env.pop("ESE_NORMALIZE_OUTPUT", None)
    if cfg.skip_unk:
        env["ESE_SKIP_UNK"] = "1"
    else:
        env.pop("ESE_SKIP_UNK", None)
    env.pop("ESE_ADD_SPECIAL_TOKENS", None)
    return env


def _encode_via_binary(sentences: list[str], cfg, env: dict) -> list[str]:
    """Run embed_stdin binary, return output lines."""
    inp = "\n".join(sentences) + "\n"
    cmd = [
        "cargo", "run", "--quiet", "--example", "embed_stdin",
        "--features", cfg.cargo_feature,
    ]
    result = subprocess.run(
        cmd, cwd=str(REPO), input=inp, capture_output=True, text=True, env=env,
        timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"embed_stdin failed: {result.stderr[:500]}")
    return [l for l in result.stdout.strip().split("\n") if l.strip()]


def measure_qps(cfg, bench_seconds: int = BENCH_SECONDS) -> dict:
    """Measure single-query and batch QPS for a model via the compiled binary."""
    env = _make_env(cfg)

    # Prepare a pool of 1000 sentences for benchmarking
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can encode semantic meaning in vector space.",
        "Static embeddings are efficient for production inference.",
        "Information retrieval systems use dense vectors for semantic search.",
        "The embedding dimension determines the capacity of the vector space.",
        "Natural language processing is a subfield of artificial intelligence.",
        "Cosine similarity measures the angle between two high-dimensional vectors.",
        "Pre-tokenization splits text into words before subword segmentation occurs.",
        "Open vocabulary models handle out-of-vocabulary words via subword units.",
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
        "Quantum computers exploit superposition and entanglement for computation.",
        "Electric vehicles are rapidly becoming more affordable and widespread.",
        "Climate change poses significant risks to global ecosystems and economies.",
        "Astronomers detected water ice near the lunar south pole last year.",
        "The chef prepared a classic French dish with fresh herbs and butter.",
        "Photosynthesis is the process by which plants convert sunlight to energy.",
        "Mount Everest is the highest peak above sea level on Earth.",
        "Cryptographic hash functions produce fixed-length outputs from arbitrary data.",
        "The library houses thousands of rare manuscripts from the medieval period.",
        "Neural networks are inspired by the biological structure of the brain.",
    ]
    # Repeat to get 1000
    sentences_pool = (base_sentences * 50)[:1000]

    print(f"\n  Warming up embed_stdin for {cfg.key}...")
    _encode_via_binary(WARMUP_SENTENCES, cfg, env)
    print(f"  Warmup done.")

    # Single-query QPS: one sentence at a time
    print(f"  Measuring single-query QPS ({bench_seconds}s)...")
    single_times = []
    single_start = time.perf_counter()
    idx = 0
    while True:
        sent = sentences_pool[idx % len(sentences_pool)]
        t0 = time.perf_counter()
        _encode_via_binary([sent], cfg, env)
        single_times.append(time.perf_counter() - t0)
        idx += 1
        if time.perf_counter() - single_start >= bench_seconds:
            break
        print(f"    single live QPS: {idx / (time.perf_counter() - single_start):.1f}", end="\r")

    # Discard warmup (first 10%)
    discard = max(1, len(single_times) // 10)
    single_times = single_times[discard:]
    single_count = len(single_times)
    single_qps = single_count / sum(single_times) if single_times else 0
    print(f"  Single-query QPS: {single_qps:.1f} ({single_count} queries)       ")

    # Batch QPS: BATCH_SIZE sentences at a time
    print(f"  Measuring batched QPS (batch={BATCH_SIZE}, {bench_seconds}s)...")
    batch_times = []
    batch_start = time.perf_counter()
    idx = 0
    while True:
        batch = sentences_pool[idx % len(sentences_pool): (idx % len(sentences_pool)) + BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            batch = batch + sentences_pool[:BATCH_SIZE - len(batch)]
        t0 = time.perf_counter()
        _encode_via_binary(batch, cfg, env)
        batch_times.append(time.perf_counter() - t0)
        idx += BATCH_SIZE
        if time.perf_counter() - batch_start >= bench_seconds:
            break
        elapsed = time.perf_counter() - batch_start
        total_queries = len(batch_times) * BATCH_SIZE
        print(f"    batch live QPS: {total_queries / elapsed:.1f}", end="\r")

    # Discard warmup
    discard = max(1, len(batch_times) // 10)
    batch_times = batch_times[discard:]
    batch_count = len(batch_times) * BATCH_SIZE
    batch_qps = batch_count / sum(batch_times) if batch_times else 0
    print(f"  Batched QPS: {batch_qps:.1f} ({batch_count} queries)              ")

    return {
        "model_key": cfg.key,
        "display_name": cfg.display_name,
        "dims": cfg.dims,
        "vocab": cfg.vocab,
        "cargo_feature": cfg.cargo_feature,
        "single_qps": round(single_qps, 1),
        "batch_qps": round(batch_qps, 1),
        "batch_size": BATCH_SIZE,
    }


if __name__ == "__main__":
    import json

    machine = get_machine_info()
    print(f"Machine: {machine}")

    cfgs = get_model_configs()

    # Determine which models to benchmark
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    else:
        # Default: benchmark default and potion-multilingual (different vocab sizes)
        keys = ["default", "potion-multilingual"]

    results = {"machine": machine, "models": {}}
    for key in keys:
        if key not in cfgs:
            print(f"WARNING: {key} not in configs", file=sys.stderr)
            continue
        cfg = cfgs[key]
        # Build the binary first
        print(f"\nBuilding embed_stdin for {cfg.key} (feature={cfg.cargo_feature})...")
        env = _make_env(cfg)
        build_result = subprocess.run(
            ["cargo", "build", "--quiet", "--example", "embed_stdin",
             "--features", cfg.cargo_feature],
            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120
        )
        if build_result.returncode != 0:
            print(f"  BUILD FAILED: {build_result.stderr[:300]}", file=sys.stderr)
            continue
        print(f"  Build OK.")
        r = measure_qps(cfg)
        results["models"][key] = r
        print(f"  RESULT: single={r['single_qps']:.1f} QPS, batch={r['batch_qps']:.1f} QPS")

    # Save
    out_path = Path(__file__).parent / "benchmarks" / "qps_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote QPS results to {out_path}")
    print(f"\n=== QPS Summary ===")
    print(f"Machine: {machine}")
    for key, r in results["models"].items():
        print(f"  {r['display_name']:40s}: single={r['single_qps']:.1f} QPS, batch={r['batch_qps']:.1f} QPS")
