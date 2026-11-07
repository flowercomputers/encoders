"""Embedding rate through sentence-transformers, mirroring the Rust benches.

Default (benches/queries.rs): queries per second with each gooaq question
encoded by its own `encode` call, cycling through the first 4096 for about
15 s per model.

`--corpus` (benches/corpus.rs): documents per second encoding the first 4096
gooaq answers as one batch, repeatedly for about 15 s per model, first pass
discarded as warm-up. The Hugging Face tokenizer spreads the batch across
every core.

Torch is limited to one thread in both. Defaults to model2vec's default model
and all-MiniLM-L6-v2; pass model names to override.

    uv run qps.py
    uv run qps.py --corpus
    uv run qps.py minishlab/potion-base-8M sentence-transformers/all-MiniLM-L6-v2
"""

import sys
import time

import torch
from datasets import load_dataset

from model_loader import load_model

N = 4096
BENCH_TIME = 15.0
DEFAULT_MODELS = [
    "minishlab/potion-base-8M",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def _gooaq(column: str) -> list[str]:
    return load_dataset("sentence-transformers/gooaq", split="train")[column][:N]


def measure_queries(models: list[str]) -> dict[str, float]:
    """Queries per second for each model, one `encode` call per question."""
    torch.set_num_threads(1)
    queries = _gooaq("question")
    out = {}
    for name in models:
        model = load_model(name, require_cpu=True)
        for q in queries[:64]:  # initialization and warm-up
            model.encode(q)

        done = 0
        start = time.perf_counter()
        while (elapsed := time.perf_counter() - start) < BENCH_TIME:
            model.encode(queries[done % len(queries)])
            done += 1
            if done % 256 == 0:
                print(f"{name}: {done / elapsed:9.1f} queries/s ({done} queries)", end="\r")
        elapsed = time.perf_counter() - start
        print(
            f"{name}: {done / elapsed:9.1f} queries/s "
            f"({done} queries, {elapsed / done * 1e6:.1f} us/query)"
            "          "
        )
        out[name] = done / elapsed
    return out


def measure_corpus(models: list[str]) -> dict[str, float]:
    """Documents per second for each model, the 4096 answers as one batch."""
    torch.set_num_threads(1)
    docs = _gooaq("answer")
    n = len(docs)
    out = {}
    for name in models:
        model = load_model(name, require_cpu=True)
        model.encode(docs[:8])  # initialization

        times = []
        start = time.perf_counter()
        while time.perf_counter() - start < BENCH_TIME or len(times) < 2:
            s = time.perf_counter()
            model.encode(docs)
            times.append(time.perf_counter() - s)
            print(f"{name}: {n / times[-1]:9.1f} docs/s ({len(times)} passes)", end="\r")

        times = times[1:]  # first pass is warm-up
        mean = sum(times) / len(times)
        print(
            f"{name}: {n / mean:9.1f} docs/s "
            f"(mean of {len(times)} passes over {n} docs, {mean * 1e3:.1f} ms/pass)"
            "          "
        )
        out[name] = n / mean
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--corpus"]
    run = measure_corpus if "--corpus" in sys.argv else measure_queries
    run(args or DEFAULT_MODELS)
