"""
run_nanobeir_ese.py — NanoBEIR evaluation for all ese models.

Runs sentence_transformers.evaluation.NanoBEIREvaluator on each EseEncoder
and saves results to benchmarks/nanobeir_ese_results.json.

Usage:
    uv run --with tokenizers python run_nanobeir_ese.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ese_encoder import get_model_configs, EseEncoder

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "benchmarks"
OUT_DIR.mkdir(exist_ok=True)


def run_nanobeir(model_key: str) -> dict | None:
    """Run NanoBEIREvaluator for one ese model. Returns results dict or None on error."""
    from sentence_transformers.evaluation import NanoBEIREvaluator

    cfgs = get_model_configs()
    if model_key not in cfgs:
        print(f"  ERROR: model {model_key!r} not in configs", file=sys.stderr)
        return None

    cfg = cfgs[model_key]
    print(f"\n{'='*60}")
    print(f"NanoBEIR: {cfg.display_name} (key={model_key})")
    print(f"  artifact: {cfg.artifact_dir}")
    print(f"  dims={cfg.dims}, normalize={cfg.normalize}, skip_unk={cfg.skip_unk}")

    encoder = EseEncoder(cfg)
    evaluator = NanoBEIREvaluator(show_progress_bar=True, write_csv=False)
    results = evaluator(encoder)

    # Extract headline metric
    # NanoBEIREvaluator returns a dict with keys like "NanoBEIR_mean_ndcg@10" etc.
    headline = None
    for k, v in results.items():
        if "mean" in k.lower() and "ndcg" in k.lower():
            headline = v
            break
    if headline is None:
        # fallback: try average of all ndcg@10
        ndcg_vals = [v for k, v in results.items() if "ndcg@10" in k.lower()]
        if ndcg_vals:
            headline = sum(ndcg_vals) / len(ndcg_vals)

    print(f"  headline nDCG@10 = {headline:.4f}" if headline is not None else "  headline = N/A")
    return {"model_key": model_key, "headline_ndcg10": headline, "full": results}


if __name__ == "__main__":
    all_results = {}
    cfgs = get_model_configs()
    model_keys = list(cfgs.keys())

    # Allow running a subset via command line args
    if len(sys.argv) > 1:
        model_keys = sys.argv[1:]

    for key in model_keys:
        try:
            r = run_nanobeir(key)
            if r is not None:
                all_results[key] = r
        except Exception as e:
            print(f"ERROR running NanoBEIR for {key}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    out_path = OUT_DIR / "nanobeir_ese_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote results to {out_path}")

    # Print summary
    print("\n=== NanoBEIR Results Summary ===")
    for key, r in all_results.items():
        h = r.get("headline_ndcg10")
        if h is not None:
            print(f"  {key:30s}: nDCG@10 = {h:.4f}")
        else:
            print(f"  {key:30s}: nDCG@10 = N/A")
