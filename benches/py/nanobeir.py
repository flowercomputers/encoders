import json
import os
import sys

from sentence_transformers.evaluation import NanoBEIREvaluator

from model_loader import load_model

model_name = sys.argv[1]
benchmarks_dir = os.environ.get("BENCHMARKS_DIR", "benchmarks")

print(f"Evaluating on NanoBEIR with {model_name}")

model = load_model(model_name, require_cpu=False)
evaluator = NanoBEIREvaluator(show_progress_bar=True, write_csv=False)
results = evaluator(model)

out_path = os.path.join(
    benchmarks_dir, f"{model_name.replace('/', '_')}_nanobeir.json"
)

os.makedirs(benchmarks_dir, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Wrote results for {model_name}")
