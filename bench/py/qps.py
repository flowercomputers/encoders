"""Sentences per second through sentence-transformers, mirroring benches/embedders.rs.

Encodes the same first 4096 gooaq questions as the Rust bench, repeatedly for
about 15 s per model, and reports the mean QPS with the first pass discarded
as warm-up. Defaults to model2vec's default model and all-MiniLM-L6-v2, run
on a single CPU thread to match the Rust bench; pass model names to override.

    uv run qps.py
    uv run qps.py minishlab/potion-base-8M sentence-transformers/all-MiniLM-L6-v2
"""

import sys
import time

import torch
from datasets import load_dataset

from model_loader import load_model

BATCH = 4096
BENCH_TIME = 15.0
DEFAULT_MODELS = [
    "minishlab/potion-base-8M",
    "sentence-transformers/all-MiniLM-L6-v2",
]



def measure(models: list[str]) -> dict[str, float]:
    """Mean sentences per second for each model over the 4096-question batch."""
    torch.set_num_threads(1)
    queries = load_dataset("sentence-transformers/gooaq", split="train")["question"][:BATCH]
    n = len(queries)
    out = {}
    for name in models:
        model = load_model(name, require_cpu=True)
        model.encode(queries[:8])  # initialization

        times = []
        start = time.perf_counter()
        while time.perf_counter() - start < BENCH_TIME or len(times) < 2:
            s = time.perf_counter()
            model.encode(queries)
            times.append(time.perf_counter() - s)
            print(f"{name}: {n / times[-1]:9.1f} sentences/s ({len(times)} passes)", end="\r")

        times = times[1:]  # first pass is warm-up
        mean = sum(times) / len(times)
        print(
            f"{name}: {n / mean:9.1f} sentences/s "
            f"(mean of {len(times)} passes over {n} sentences, {mean * 1e3:.1f} ms/pass)"
            "          "
        )
        out[name] = n / mean
    return out


if __name__ == "__main__":
    measure(sys.argv[1:] or DEFAULT_MODELS)
