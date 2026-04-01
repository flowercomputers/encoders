import json
import time

import bm25s
import numpy as np
from datasets import load_dataset

bench_time = 10
discard_percent = 10
model_name = "bm25"
print(f"Benchmarking GooAQ QPS for {model_name}")

ds = load_dataset("sentence-transformers/gooaq", split="train")
queries = ds["question"]
corpus = ds["answer"]
n = len(queries)

step = 32

# Build BM25 index
print("Building BM25 index...")
corpus_tokens = bm25s.tokenize(
    corpus,
)
model = bm25s.BM25()
model.index(corpus_tokens)
print("Index built.")

# Single-query QPS
offset = 0
single_times = []
single_start = time.perf_counter()
while True:
    if offset + step <= n:
        batch = queries[offset : offset + step]
    else:
        batch = queries[offset:] + queries[: (offset + step) - n]

    for q in batch:
        s = time.perf_counter()
        scores = model.get_scores(
            bm25s.tokenize(
                [q],
            ).ids[0]
        )
        single_times.append(time.perf_counter() - s)
    offset += step
    offset %= n

    single_total = sum(single_times)
    single_count = len(single_times)
    single_qps = single_count / single_total
    wall_elapsed = time.perf_counter() - single_start

    if wall_elapsed >= bench_time:
        break

    print(
        f"Single live QPS: {single_qps:4.2f} ({single_count}, {single_total:.2f}s, {bench_time - wall_elapsed:.2f}s)",
        end="\r",
    )

# discard early samples as warmups
single_discard = len(single_times) // discard_percent
single_times = single_times[single_discard:]
single_total = sum(single_times)
single_count = len(single_times)
single_qps = single_count / single_total

print(
    f"QPS: {single_qps:.2f} ({single_count} in {single_total:.6f}s)"
    "                     "
)

results = {
    "model": model_name,
    "bench_time_target": bench_time,
    "step": step,
    "single_query": {
        "count": single_count,
        "total_time": single_total,
        "qps": single_qps,
        "discard": single_discard,
        "latency_mean": np.mean(single_times),
        "latency_median": np.median(single_times),
        "latency_std": np.std(single_times),
        "latency_p50": np.percentile(single_times, 50),
        "latency_p95": np.percentile(single_times, 95),
        "latency_p99": np.percentile(single_times, 99),
        "latency_min": np.min(single_times),
        "latency_max": np.max(single_times),
    },
}

out_path = f"benchmarks/bench_{model_name.replace('/', '_')}.json"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


with open(out_path, "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"Results saved to {out_path}")
