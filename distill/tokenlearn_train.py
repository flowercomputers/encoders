"""
tokenlearn_train.py — Tokenlearn-quality distillation for ese.

Implements plan 017 G3: featurize a corpus with a teacher, run tokenlearn
training on the raw model2vec distill, measure NanoBEIR lift.

Usage:
    uv run python tokenlearn_train.py \\
        --teacher sentence-transformers/all-MiniLM-L6-v2 \\
        --corpus gooaq \\
        --n-texts 100000 \\
        --out /path/to/output-dir \\
        --dims 256

The output is an ese-compatible artifact dir (model.safetensors + tokenizer.json)
identical in format to what distill.py produces, but with Tokenlearn-quality embeddings.

Corpus options (loaded from HuggingFace):
    gooaq    — sentence-transformers/gooaq (fast, English QA)
    wikitext — wikitext-103-raw-v1 (clean Wikipedia prose)
    c4       — allenai/c4 en (streamed, very large — use with small --n-texts)

COMPUTE NOTES:
  - Featurize: ~1k sent/s on CPU with MiniLM — 100k texts ≈ 100s
  - Train: tokenlearn uses early stopping; on CPU with 90k train + 10k val,
    convergence typically in 2–5 epochs ≈ 5–15 min
  - Total budget: well under 1h for 100k texts on M4 Pro CPU
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Corpus loading helpers
# ---------------------------------------------------------------------------

def load_corpus(corpus: str, n_texts: int) -> list[str]:
    """Load up to n_texts from the specified corpus."""
    print(f"\n[corpus] loading {n_texts:,} texts from {corpus!r} ...")
    t0 = time.time()

    texts: list[str] = []

    if corpus == "gooaq":
        from datasets import load_dataset
        ds = load_dataset(
            "sentence-transformers/gooaq",
            split="train",
            streaming=True,
        )
        for row in ds:
            q = row.get("question", "")
            a = row.get("answer", "")
            if q:
                texts.append(q)
            if len(texts) >= n_texts:
                break
            if a:
                texts.append(a)
            if len(texts) >= n_texts:
                break

    elif corpus == "wikitext":
        from datasets import load_dataset
        ds = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split="train",
            streaming=True,
        )
        for row in ds:
            t = row.get("text", "").strip()
            if len(t) > 20:
                texts.append(t)
            if len(texts) >= n_texts:
                break

    elif corpus == "c4":
        from datasets import load_dataset
        ds = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for row in ds:
            t = row.get("text", "").strip()
            # Use first sentence or first 256 chars
            sent = t.split(".")[0].strip()
            if len(sent) > 20:
                texts.append(sent[:512])
            if len(texts) >= n_texts:
                break

    else:
        raise ValueError(f"Unknown corpus: {corpus!r}. Use gooaq, wikitext, or c4.")

    texts = texts[:n_texts]
    elapsed = time.time() - t0
    print(f"  loaded {len(texts):,} texts in {elapsed:.1f}s")
    return texts


# ---------------------------------------------------------------------------
# Featurize: teacher encode → mean vectors
# ---------------------------------------------------------------------------

def featurize(teacher_id: str, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode texts with the teacher model, return [N, teacher_dim] float32."""
    from sentence_transformers import SentenceTransformer

    print(f"\n[featurize] loading teacher {teacher_id!r} ...")
    teacher = SentenceTransformer(teacher_id)
    teacher.eval()

    print(f"  encoding {len(texts):,} texts with batch_size={batch_size} ...")
    t0 = time.time()
    vecs = teacher.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # tokenlearn does its own PCA/normalization
    )
    elapsed = time.time() - t0
    print(f"  featurized {len(texts):,} texts → shape {vecs.shape} in {elapsed:.1f}s "
          f"({len(texts)/elapsed:.0f} sent/s)")
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Raw distill (model2vec) + Tokenlearn training
# ---------------------------------------------------------------------------

def run_tokenlearn(
    teacher_id: str,
    texts: list[str],
    vecs: np.ndarray,
    out_dir: Path,
    dims: int = 256,
    device: str = "cpu",
) -> None:
    """
    1. model2vec.distill() → raw StaticModel
    2. tokenlearn.train.train_model() → trained StaticModel
    3. Save to out_dir in ese format (model.safetensors F32 + tokenizer.json)
    """
    from model2vec.distill import distill as m2v_distill
    from tokenlearn.train import train_model

    print(f"\n[tokenlearn] step 1: raw model2vec distill of {teacher_id!r} ...")
    t0 = time.time()
    static_model = m2v_distill(
        model_name=teacher_id,
        pca_dims=dims,
    )
    print(f"  distill done in {time.time()-t0:.1f}s  "
          f"vocab={static_model.embedding.shape[0]}  dims={static_model.embedding.shape[1]}")

    # model2vec >= 0.4 saves embeddings in F16; tokenlearn expects F32 for
    # correct dtype in StaticModelFineTuner (bmm dtype mismatch otherwise).
    if static_model.embedding.dtype != np.float32:
        print(f"  converting embedding F16 → F32 for tokenlearn compatibility")
        static_model.embedding = static_model.embedding.astype(np.float32)

    print(f"\n[tokenlearn] step 2: tokenlearn training on {len(texts):,} texts ...")
    t0 = time.time()
    trained_model = train_model(
        model=static_model,
        train_txt=list(texts),
        train_vec=vecs,
        device=device,
        pca_dims=dims,
    )
    elapsed = time.time() - t0
    print(f"  tokenlearn training done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save to temp, then repair and move to out_dir
    print(f"\n[tokenlearn] step 3: saving to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "model"
        trained_model.save_pretrained(str(tmp_path))

        files = sorted(f.name for f in tmp_path.iterdir())
        print(f"  tokenlearn save_pretrained → {files}")

        # Copy + convert model.safetensors to F32
        model_src = tmp_path / "model.safetensors"
        if not model_src.exists():
            candidates = list(tmp_path.glob("*.safetensors"))
            if not candidates:
                print(f"STOP: no .safetensors in {files}", file=sys.stderr)
                sys.exit(1)
            model_src = candidates[0]

        _copy_model_as_f32(model_src, out_dir / "model.safetensors")

        # Copy tokenizer.json
        tokenizer_src = tmp_path / "tokenizer.json"
        if not tokenizer_src.exists():
            print(f"STOP: no tokenizer.json in {files}", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(tokenizer_src, out_dir / "tokenizer.json")

        # Copy config.json if present
        config_src = tmp_path / "config.json"
        if config_src.exists():
            cfg = json.loads(config_src.read_text())
            cfg["embedding_dtype"] = "float32"
            (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Write info file
    info = f"""# TOKENLEARN_INFO

| Key | Value |
|-----|-------|
| teacher | `{teacher_id}` |
| dims | {dims} |
| n_texts_train | {len(texts)} |
| device | {device} |
| training | tokenlearn 0.2.1 (MSE, early stopping) |
| base_distill | model2vec (raw distill → tokenlearn fine-tune) |

Generated by `distill/tokenlearn_train.py`.
These artifacts are ese-compatible (model.safetensors F32 + tokenizer.json).

Recommended build env:
```sh
ESE_MODEL_DIR=... ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 cargo test --features tests,dim-256
```
"""
    (out_dir / "TOKENLEARN_INFO.md").write_text(info)
    print(f"  saved to {out_dir}/")


# ---------------------------------------------------------------------------
# safetensors F32 copy (mirrored from distill.py)
# ---------------------------------------------------------------------------

def _copy_model_as_f32(src: Path, dst: Path) -> None:
    """Copy safetensors file to dst, converting tensor to F32."""
    with open(src, "rb") as f:
        header_len = struct.unpack_from("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        data_region = f.read()

    non_meta = {k: v for k, v in header.items() if not k.startswith("__")}
    tensor_name = list(non_meta)[0]
    meta = non_meta[tensor_name]
    dtype_str = meta["dtype"]
    shape = meta["shape"]
    offsets = meta["data_offsets"]

    if dtype_str == "F32":
        shutil.copy2(src, dst)
        print(f"  model dtype: F32 (no conversion needed), shape={shape}")
        return

    raw = data_region[offsets[0]:offsets[1]]
    if dtype_str == "BF16":
        arr = np.frombuffer(raw, dtype=np.uint16)
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    elif dtype_str == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype_str == "F64":
        arr = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    arr = arr.reshape(shape)
    print(f"  model dtype: {dtype_str} → F32, shape={list(shape)}")

    f32_bytes = arr.astype(np.float32).tobytes()
    tensor_meta = {
        tensor_name: {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [0, len(f32_bytes)],
        }
    }
    new_header_bytes = json.dumps(tensor_meta, separators=(",", ":")).encode("utf-8")
    pad_len = (8 - (len(new_header_bytes) % 8)) % 8
    new_header_bytes += b" " * pad_len
    new_header_len = struct.pack("<Q", len(new_header_bytes))

    with open(dst, "wb") as f:
        f.write(new_header_len)
        f.write(new_header_bytes)
        f.write(f32_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenlearn-quality distillation: featurize corpus + train static model."
    )
    parser.add_argument(
        "--teacher",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF teacher model ID (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--corpus",
        default="gooaq",
        choices=["gooaq", "wikitext", "c4"],
        help="Corpus to featurize (default: gooaq)",
    )
    parser.add_argument(
        "--n-texts",
        type=int,
        default=100_000,
        help="Number of texts to use (default: 100000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for trained ese artifact",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=256,
        help="Embedding dimensions (default: 256)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Featurization batch size (default: 256)",
    )
    args = parser.parse_args()

    total_t0 = time.time()

    # Step 1: load corpus
    texts = load_corpus(args.corpus, args.n_texts)

    # Step 2: featurize with teacher
    vecs = featurize(args.teacher, texts, batch_size=args.batch_size)

    # Step 3: tokenlearn training + save
    run_tokenlearn(
        teacher_id=args.teacher,
        texts=texts,
        vecs=vecs,
        out_dir=args.out,
        dims=args.dims,
        device=args.device,
    )

    total_elapsed = time.time() - total_t0
    print(f"\n[done] total wall-clock: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  artifact: {args.out}")
    print(f"\nNext steps:")
    print(f"  1. Check parity: cd ../distill && uv run python crosscheck.py {args.out} --dims {args.dims}")
    print(f"  2. NanoBEIR:    cd ../benches/py && uv run python run_nanobeir_ese.py tokenlearn-minilm")


if __name__ == "__main__":
    main()
