"""
crosscheck.py — cross-language parity check between model2vec and ese.

Usage:
    uv run python crosscheck.py <artifact-dir> [--dims N]

For each of ~20 fixed sentences (English + CJK + code), encodes via:
  (a) model2vec.StaticModel.from_pretrained(artifact-dir)
  (b) cargo run --example embed_stdin --features dim-<dims>
      with ESE_MODEL_DIR=<artifact-dir> ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1

Prints per-sentence cosine, minimum cosine, and evaluates 3 semantic triplets
on the ese outputs. Reports UNK incidence per model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fixed sentence set: English + CJK + code (~20 sentences)
# ---------------------------------------------------------------------------

SENTENCES = [
    # English
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models encode semantic meaning in vector space.",
    "Static embeddings are efficient for production inference.",
    "The capital of France is Paris.",
    "Natural language processing is a subfield of AI.",
    "Cosine similarity measures the angle between two vectors.",
    "Information retrieval systems use embeddings for semantic search.",
    "The embedding dimension determines the vector space size.",
    "Pre-tokenization splits text into words before subword segmentation.",
    "Open vocabulary models handle unseen words via subword units.",
    # CJK
    "你好世界",          # Hello world (Chinese)
    "机器学习是人工智能的子领域。",  # Machine learning is a subfield of AI (Chinese)
    "東京は日本の首都です。",        # Tokyo is the capital of Japan (Japanese)
    "자연어 처리는 AI의 하위 분야입니다.",  # NLP is a subfield of AI (Korean)
    # Code / mixed
    "def encode(text: str) -> list[float]:",
    "cargo run --example embed_stdin --features dim-256",
    "import numpy as np; arr = np.zeros((vocab_size, dims))",
    "ESE_MODEL_DIR=/path/to/model ESE_NORMALIZE_OUTPUT=1 cargo build",
    "hello world",
    "café résumé naïve",
]

# Semantic triplets: (anchor, similar, dissimilar)
# Test: cosine(anchor, similar) > cosine(anchor, dissimilar) on ese outputs.
TRIPLETS = [
    (
        "A dog is playing in the park.",
        "The puppy is running outside.",
        "The stock market fell sharply today.",
    ),
    (
        "Machine learning models can encode semantic meaning.",
        "Neural networks learn representations from data.",
        "The weather in Tokyo is rainy this week.",
    ),
    (
        "The capital of France is Paris.",
        "Paris is the largest city in France.",
        "def main(): print('hello world')",
    ),
]


# ---------------------------------------------------------------------------
# Encode with model2vec
# ---------------------------------------------------------------------------

def encode_model2vec(artifact_dir: Path) -> np.ndarray:
    """Encode SENTENCES via model2vec.StaticModel. Returns [N, dims] float32.

    NOTE: encode_batch_fast (used when _can_encode_fast=True) pads shorter
    sequences to match the longest in the batch. Padding id=0 has a non-zero
    embedding and pollutes mean pooling. We encode one sentence at a time to
    avoid this padding artifact and get the same result as model2vec's own
    per-sentence tokenize()-then-mean-pool path.
    """
    from model2vec import StaticModel
    model = StaticModel.from_pretrained(str(artifact_dir))
    # Encode one at a time: avoids batch-padding pollution in encode_batch_fast.
    rows = [model.encode(s, show_progressbar=False) for s in SENTENCES]
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Encode with ese via embed_stdin
# ---------------------------------------------------------------------------

def encode_ese(artifact_dir: Path, dims: int) -> np.ndarray:
    """Encode SENTENCES via cargo run --example embed_stdin. Returns [N, dims] float32."""
    repo_root = Path(__file__).parent.parent
    env_override = {
        "ESE_MODEL_DIR": str(artifact_dir.resolve()),
        "ESE_NORMALIZE_OUTPUT": "1",
        "ESE_SKIP_UNK": "1",
    }
    import os
    env = os.environ.copy()
    env.update(env_override)

    # Build the example first (ensures it is compiled for this artifact)
    build_cmd = [
        "cargo", "build", "--example", "embed_stdin",
        "--features", f"dim-{dims}",
        "--quiet",
    ]
    result = subprocess.run(build_cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STOP: cargo build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Run the example with sentences on stdin
    run_cmd = [
        "cargo", "run", "--example", "embed_stdin",
        "--features", f"dim-{dims}",
        "--quiet",
    ]
    stdin_text = "\n".join(SENTENCES) + "\n"
    result = subprocess.run(
        run_cmd, cwd=repo_root, env=env,
        input=stdin_text, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"STOP: embed_stdin failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    if len(lines) != len(SENTENCES):
        print(
            f"STOP: expected {len(SENTENCES)} output lines, got {len(lines)}",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = []
    for line in lines:
        vals = [float(x) for x in line.split()]
        rows.append(vals)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# UNK incidence
# ---------------------------------------------------------------------------

def count_unk(artifact_dir: Path) -> int:
    """Count UNK token occurrences across SENTENCES using the tokenizers library."""
    from tokenizers import Tokenizer
    tk_path = artifact_dir / "tokenizer.json"
    tk_json = json.loads(tk_path.read_text())
    model_type = tk_json.get("model", {}).get("type", "unknown")

    tk = Tokenizer.from_file(str(tk_path))

    # Determine unk_id
    if model_type == "WordPiece":
        unk_token = tk_json.get("model", {}).get("unk_token", "[UNK]")
        vocab = tk_json.get("model", {}).get("vocab", {})
        unk_id = vocab.get(unk_token)
    elif model_type == "Unigram":
        unk_id = tk_json.get("model", {}).get("unk_id", 0)
    else:
        unk_id = None

    if unk_id is None:
        return 0

    total_unk = 0
    for sentence in SENTENCES:
        encoding = tk.encode(sentence, add_special_tokens=False)
        total_unk += sum(1 for tid in encoding.ids if tid == unk_id)
    return total_unk


# ---------------------------------------------------------------------------
# Cosine helpers
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x / norms


def upper_tri_sim(x: np.ndarray) -> np.ndarray:
    """Upper-triangular entries of the pairwise cosine similarity matrix.

    Mirror of spikes/specials-parity/check.py's upper_tri_sim — this is the
    maintainer's retrieval-relevant metric (MinishLab/model2vec#335): differences
    in the S-matrix capture reordering of neighbor similarities across the whole
    corpus, not just individual pair cosines.
    """
    xn = l2_normalize(x)
    S = xn @ xn.T
    iu = np.triu_indices(S.shape[0], k=1)
    return S[iu]


def s_matrix_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (mean|S_a - S_b|, corr(S_a, S_b)) for two embedding matrices."""
    sa = upper_tri_sim(a)
    sb = upper_tri_sim(b)
    delta = float(np.abs(sa - sb).mean())
    corr = float(np.corrcoef(sa, sb)[0, 1])
    return delta, corr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-language parity: model2vec vs ese embed_stdin."
    )
    parser.add_argument("artifact_dir", type=Path, help="Artifact directory (model.safetensors + tokenizer.json)")
    parser.add_argument("--dims", type=int, default=256, help="Embedding dims (must match artifact, default 256)")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    dims = args.dims

    print(f"Crosscheck: {artifact_dir}  dims={dims}")
    print(f"Sentences: {len(SENTENCES)}")

    # --- model2vec encodings ---
    print("\n[model2vec] encoding ...")
    mv_embs = encode_model2vec(artifact_dir)
    mv_embs = l2_normalize(mv_embs)

    # --- ese encodings ---
    print("[ese] encoding via embed_stdin ...")
    ese_embs = encode_ese(artifact_dir, dims)
    # ese with ESE_NORMALIZE_OUTPUT=1 already outputs normalized vectors,
    # but renormalize defensively for the comparison.
    ese_embs = l2_normalize(ese_embs)

    # --- UNK incidence ---
    unk_count = count_unk(artifact_dir)
    print(f"\nUNK incidence across {len(SENTENCES)} sentences: {unk_count} token(s)\n")

    # --- Per-sentence cosine ---
    print(f"{'#':>3}  {'cosine':>10}  sentence")
    print("-" * 70)
    min_cosine = float("inf")
    all_ok = True
    for i, sentence in enumerate(SENTENCES):
        c = cosine(mv_embs[i], ese_embs[i])
        min_cosine = min(min_cosine, c)
        status = "OK" if c >= 0.999 else ("AMBER" if c >= 0.99 else "FAIL")
        if c < 0.999:
            all_ok = False
        print(f"{i+1:>3}  {c:>10.6f}  [{status}]  {sentence[:55]!r}")

    print("-" * 70)
    threshold_ok = min_cosine >= 0.999
    print(f"  min cosine = {min_cosine:.6f}  (threshold: 0.999)  {'PASS' if threshold_ok else 'FAIL'}")

    # --- S-matrix delta (maintainer's retrieval-relevant metric) ---
    s_delta, s_corr = s_matrix_delta(mv_embs, ese_embs)
    print(f"\n[S-matrix] upper-triangular similarity matrix delta (model2vec vs ese):")
    print(f"  mean|S_m2v - S_ese| = {s_delta:.5f}")
    print(f"  corr(S_m2v, S_ese)  = {s_corr:.5f}")
    print(f"  (0.00000 / 1.00000 = perfect geometry match)")

    # --- Semantic triplets on ese outputs ---
    print("\n[Semantic triplets on ese outputs]")
    # We need to encode the 3 triplet sentences too
    # They are not in SENTENCES, so re-encode them.
    triplet_sents = []
    for anchor, similar, dissimilar in TRIPLETS:
        for s in (anchor, similar, dissimilar):
            if s not in triplet_sents:
                triplet_sents.append(s)

    # Encode via ese
    repo_root = Path(__file__).parent.parent
    env_override = {
        "ESE_MODEL_DIR": str(artifact_dir.resolve()),
        "ESE_NORMALIZE_OUTPUT": "1",
        "ESE_SKIP_UNK": "1",
    }
    import os
    env = os.environ.copy()
    env.update(env_override)
    run_cmd = ["cargo", "run", "--example", "embed_stdin", "--features", f"dim-{dims}", "--quiet"]
    stdin_text = "\n".join(triplet_sents) + "\n"
    result = subprocess.run(run_cmd, cwd=repo_root, env=env, input=stdin_text, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STOP: embed_stdin for triplets failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    t_lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    t_embs = {}
    for i, sent in enumerate(triplet_sents):
        vals = [float(x) for x in t_lines[i].split()]
        t_embs[sent] = np.array(vals, dtype=np.float32)

    triplet_pass = 0
    for i, (anchor, similar, dissimilar) in enumerate(TRIPLETS):
        a = t_embs[anchor]
        s = t_embs[similar]
        d = t_embs[dissimilar]
        cos_sim = cosine(a, s)
        cos_dis = cosine(a, d)
        ok = cos_sim > cos_dis
        if ok:
            triplet_pass += 1
        status = "PASS" if ok else "FAIL"
        print(f"  Triplet {i+1}: [{status}]  sim={cos_sim:.4f}  dis={cos_dis:.4f}")
        print(f"    anchor:     {anchor[:60]!r}")
        print(f"    similar:    {similar[:60]!r}")
        print(f"    dissimilar: {dissimilar[:60]!r}")

    print(f"\n  Triplets: {triplet_pass}/{len(TRIPLETS)} correct")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    print(f"  min cosine (model2vec vs ese): {min_cosine:.6f}  {'PASS' if threshold_ok else 'FAIL'}")
    print(f"  S-matrix delta:                {s_delta:.5f}  corr={s_corr:.5f}")
    print(f"  UNK tokens in test set:        {unk_count}")
    print(f"  semantic triplets:             {triplet_pass}/{len(TRIPLETS)}")

    if not threshold_ok or triplet_pass < len(TRIPLETS):
        sys.exit(1)


if __name__ == "__main__":
    main()
