"""
measure_qps_direct.py — QPS benchmarks via pre-built release ese binaries.

Builds release binaries once, then calls them directly (avoiding cargo
startup overhead). Measures single-sentence and batched QPS.

Usage:
    uv run --with tokenizers python measure_qps_direct.py [model_key ...]
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ese_encoder import get_model_configs, REPO

BENCH_SECONDS = 15
BATCH_SIZE = 32
WARMUP_COUNT = 50


def get_machine_info() -> str:
    info = platform.uname()
    cpu = info.processor or info.machine
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            cpu = result.stdout.strip()
    except Exception:
        pass
    mem_gb = None
    try:
        r2 = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if r2.returncode == 0:
            mem_gb = int(r2.stdout.strip()) // (1024**3)
    except Exception:
        pass
    mem_str = f", {mem_gb}GB RAM" if mem_gb else ""
    return f"{info.system} {info.release} | CPU: {cpu}{mem_str}"


def build_release_binary(cfg, env: dict) -> Path:
    """Build release embed_stdin binary. Returns path to binary."""
    print(f"  Building release binary for {cfg.key} (feature={cfg.cargo_feature})...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--quiet", "--example", "embed_stdin",
         "--features", cfg.cargo_feature],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed: {result.stderr[:500]}")
    # Find the binary
    bin_path = REPO / "target" / "release" / "examples" / "embed_stdin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary not found at {bin_path}")
    print(f"  Binary built: {bin_path}")
    return bin_path


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


def _run_binary(binary: Path, sentences: list[str], env: dict) -> None:
    inp = "\n".join(sentences) + "\n"
    result = subprocess.run(
        [str(binary)], input=inp, capture_output=True, text=True,
        env=env, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"Binary failed: {result.stderr[:200]}")


# Sentence pool for benchmarking
SENTENCES_POOL = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can encode semantic meaning in vector space.",
    "Static embeddings are efficient for production inference at scale.",
    "Information retrieval systems use dense vectors for semantic search.",
    "The embedding dimension determines the capacity of the vector space.",
    "Natural language processing is a subfield of artificial intelligence.",
    "Cosine similarity measures the angle between two high-dimensional vectors.",
    "Pre-tokenization splits text into words before subword segmentation.",
    "Open vocabulary models handle out-of-vocabulary words via subword units.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Quantum computers exploit superposition and entanglement for computation.",
    "Electric vehicles are rapidly becoming more affordable and widespread.",
    "Climate change poses significant risks to global ecosystems and economies.",
    "Astronomers detected water ice near the lunar south pole.",
    "The chef prepared a classic dish with fresh herbs and butter.",
    "Photosynthesis is the process by which plants convert sunlight to energy.",
    "Mount Everest is the highest peak above sea level on Earth.",
    "Cryptographic hash functions produce fixed-length outputs from arbitrary data.",
    "The library houses thousands of rare manuscripts from medieval times.",
    "Neural networks are inspired by the biological structure of the brain.",
    "Renewable energy sources include solar panels and wind turbines.",
    "The algorithm runs in O(n log n) time complexity for sorting.",
    "Dense retrieval methods outperform sparse methods on many benchmarks.",
    "The conference featured talks on deep learning and natural language.",
    "Satellite imagery reveals significant glacier retreat over the past decade.",
    "A healthy diet includes fruits, vegetables, and whole grains.",
    "The programmers refactored the codebase to improve maintainability.",
    "Stock market volatility increased following the central bank announcement.",
    "The museum exhibit showcases ancient artifacts from the Roman Empire.",
    "Biologists discovered a new species of deep-sea fish last year.",
    "The spacecraft successfully completed its mission to the outer planets.",
    "Researchers published findings on the effects of sleep on memory.",
]
SENTENCES_POOL = (SENTENCES_POOL * 32)[:1000]  # 1000 sentences


def measure_qps_direct(cfg, bench_seconds: int = BENCH_SECONDS) -> dict:
    env = _make_env(cfg)
    binary = build_release_binary(cfg, env)

    # Warmup
    print(f"  Warming up ({WARMUP_COUNT} calls)...")
    for i in range(WARMUP_COUNT):
        _run_binary(binary, [SENTENCES_POOL[i % len(SENTENCES_POOL)]], env)
    print(f"  Warmup done.")

    # Single-sentence QPS
    print(f"  Measuring single-sentence QPS ({bench_seconds}s)...")
    single_times = []
    single_start = time.perf_counter()
    idx = 0
    while time.perf_counter() - single_start < bench_seconds:
        sent = SENTENCES_POOL[idx % len(SENTENCES_POOL)]
        t0 = time.perf_counter()
        _run_binary(binary, [sent], env)
        single_times.append(time.perf_counter() - t0)
        idx += 1
    single_count = len(single_times)
    single_qps = single_count / sum(single_times)
    # Latency stats
    import statistics
    latencies = [t * 1000 for t in single_times]  # ms
    lat_p50 = statistics.median(latencies)
    lat_p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    print(f"  Single: {single_qps:.1f} QPS, p50={lat_p50:.1f}ms, p95={lat_p95:.1f}ms")

    # Batched QPS (batch of BATCH_SIZE)
    print(f"  Measuring batched QPS (batch={BATCH_SIZE}, {bench_seconds}s)...")
    batch_times = []
    batch_start = time.perf_counter()
    idx = 0
    while time.perf_counter() - batch_start < bench_seconds:
        start = idx % len(SENTENCES_POOL)
        batch = SENTENCES_POOL[start:start + BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            batch = batch + SENTENCES_POOL[:BATCH_SIZE - len(batch)]
        t0 = time.perf_counter()
        _run_binary(binary, batch, env)
        batch_times.append(time.perf_counter() - t0)
        idx += BATCH_SIZE
    batch_count = len(batch_times) * BATCH_SIZE
    batch_qps = batch_count / sum(batch_times)
    print(f"  Batch: {batch_qps:.1f} QPS ({batch_count} sentences total)")

    return {
        "model_key": cfg.key,
        "display_name": cfg.display_name,
        "dims": cfg.dims,
        "vocab": cfg.vocab,
        "cargo_feature": cfg.cargo_feature,
        "single_qps": round(single_qps, 1),
        "single_lat_p50_ms": round(lat_p50, 2),
        "single_lat_p95_ms": round(lat_p95, 2),
        "batch_qps": round(batch_qps, 1),
        "batch_size": BATCH_SIZE,
    }


if __name__ == "__main__":
    machine = get_machine_info()
    print(f"Machine: {machine}")

    cfgs = get_model_configs()
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    else:
        keys = ["default", "potion-multilingual"]

    results = {"machine": machine, "models": {}}
    for key in keys:
        if key not in cfgs:
            print(f"WARNING: {key} not in configs", file=sys.stderr)
            continue
        cfg = cfgs[key]
        print(f"\n{'='*60}")
        print(f"Benchmarking: {cfg.display_name} ({key})")
        try:
            r = measure_qps_direct(cfg)
            results["models"][key] = r
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    out_path = Path(__file__).parent / "benchmarks" / "qps_direct_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote QPS results to {out_path}")
    print(f"\n=== QPS Summary ===")
    print(f"Machine: {machine}")
    for key, r in results["models"].items():
        print(f"  {r['display_name']:40s}: single={r['single_qps']:.1f} QPS  "
              f"(p50={r.get('single_lat_p50_ms', '?'):.1f}ms), "
              f"batch={r['batch_qps']:.1f} QPS")
