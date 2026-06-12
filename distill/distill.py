"""
distill.py — turn any HF sentence-transformers teacher into ese-compatible artifacts.

Usage:
    uv run python distill.py <teacher-model> --out <output-dir> [--dims N]
    uv run python distill.py --check <output-dir>

Output directory will contain:
    model.safetensors   — single F32 tensor named "embeddings", shape [vocab, dims]
    tokenizer.json      — the distilled (compacted) tokenizer
    DISTILL_INFO.md     — metadata: teacher, revision, dims, model2vec version, settings
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# safetensors F32 export helper
# ---------------------------------------------------------------------------

def _copy_model_as_f32(src: Path, dst: Path) -> None:
    """Copy a safetensors file to dst, converting the single tensor to F32.

    model2vec >= 0.4 saves in F16. ese requires F32. We read, convert, and
    re-write using the safetensors library so the header is valid.
    """
    import struct

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
        # Already F32, just copy
        shutil.copy2(src, dst)
        print(f"  model dtype: F32 (no conversion needed)")
        return

    # Read, decode, convert to F32
    raw = data_region[offsets[0]:offsets[1]]
    dtype_map = {"F16": np.float16, "BF16": np.uint16, "F64": np.float64}
    if dtype_str == "BF16":
        arr = np.frombuffer(raw, dtype=np.uint16)
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    elif dtype_str in dtype_map:
        arr = np.frombuffer(raw, dtype=dtype_map[dtype_str]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype for conversion: {dtype_str}")

    arr = arr.reshape(shape)
    print(f"  model dtype: {dtype_str} → converting to F32, shape={list(shape)}")

    # Write as F32 safetensors
    f32_bytes = arr.astype(np.float32).tobytes()
    tensor_meta = {
        tensor_name: {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [0, len(f32_bytes)],
        }
    }
    new_header_bytes = json.dumps(tensor_meta, separators=(",", ":")).encode("utf-8")
    # safetensors requires 8-byte alignment of the data region
    pad_len = (8 - (len(new_header_bytes) % 8)) % 8
    new_header_bytes += b" " * pad_len
    new_header_len = struct.pack("<Q", len(new_header_bytes))

    with open(dst, "wb") as f:
        f.write(new_header_len)
        f.write(new_header_bytes)
        f.write(f32_bytes)


# ---------------------------------------------------------------------------
# distill: run model2vec distillation and save to output dir
# ---------------------------------------------------------------------------

def distill(teacher: str, out_dir: Path, dims: int) -> None:
    import importlib.metadata
    from model2vec.distill import distill as m2v_distill

    print(f"Distilling {teacher} with dims={dims} ...")
    m = m2v_distill(model_name=teacher, pca_dims=dims)

    # Save to a temp dir, then move into out_dir
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "model"
        m.save_pretrained(str(tmp_path))

        print(f"  model2vec save_pretrained → {tmp_path}")
        print(f"  files: {sorted(f.name for f in tmp_path.iterdir())}")

        out_dir.mkdir(parents=True, exist_ok=True)

        # Identify and copy model.safetensors
        model_src = tmp_path / "model.safetensors"
        if not model_src.exists():
            # Some versions write embeddings.safetensors or similar
            candidates = list(tmp_path.glob("*.safetensors"))
            if not candidates:
                print(
                    f"STOP: model2vec save_pretrained did not produce a .safetensors file.\n"
                    f"      Files in {tmp_path}: {sorted(f.name for f in tmp_path.iterdir())}",
                    file=sys.stderr,
                )
                sys.exit(1)
            model_src = candidates[0]
            print(f"  (using {model_src.name} as model.safetensors)")

        # Copy model, converting to F32 if needed (model2vec >= 0.4 saves F16)
        _copy_model_as_f32(model_src, out_dir / "model.safetensors")

        # Copy tokenizer.json
        tokenizer_src = tmp_path / "tokenizer.json"
        if not tokenizer_src.exists():
            print(
                f"STOP: model2vec save_pretrained did not produce tokenizer.json.\n"
                f"      Files in {tmp_path}: {sorted(f.name for f in tmp_path.iterdir())}",
                file=sys.stderr,
            )
            sys.exit(1)
        shutil.copy2(tokenizer_src, out_dir / "tokenizer.json")

        # Copy config.json if present (needed by model2vec StaticModel.from_pretrained)
        config_src = tmp_path / "config.json"
        if config_src.exists():
            cfg = json.loads(config_src.read_text())
            # Update embedding_dtype to float32 to match our converted file
            cfg["embedding_dtype"] = "float32"
            (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Write DISTILL_INFO.md
    try:
        m2v_version = importlib.metadata.version("model2vec")
    except Exception:
        m2v_version = "unknown"

    info = f"""# DISTILL_INFO

| Key | Value |
|-----|-------|
| teacher | `{teacher}` |
| dims | {dims} |
| model2vec_version | {m2v_version} |
| pca_dims | {dims} |
| weighting | default (subword frequency weighting via model2vec) |

Generated by `distill.py`. These artifacts are ese-compatible.
Recommended build env for model2vec-distilled artifacts:

```sh
ESE_MODEL_DIR=... ESE_NORMALIZE_OUTPUT=1 ESE_SKIP_UNK=1 cargo test --features tests,dim-256
```

`ESE_SKIP_UNK=1` ensures UNK-id tokens are excluded from mean pooling,
matching `model2vec.StaticModel.tokenize()` which filters them before pooling.
"""
    (out_dir / "DISTILL_INFO.md").write_text(info)

    print(f"  Saved to {out_dir}/")

    # Run validate + repair in-place
    validate(out_dir, repair=True)


# ---------------------------------------------------------------------------
# safetensors helpers
# ---------------------------------------------------------------------------

def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("File too short to be a safetensors file")
        header_len = struct.unpack_from("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
    return json.loads(header_bytes)


def read_safetensors_tensor(path: Path, name: str) -> np.ndarray:
    with open(path, "rb") as f:
        header_len = struct.unpack_from("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        if name not in header:
            raise KeyError(f"Tensor {name!r} not found in {path}. Keys: {list(header)}")
        meta = header[name]
        offsets = meta["data_offsets"]
        dtype_str = meta["dtype"]
        shape = meta["shape"]
        data_start = 8 + header_len + offsets[0]
        data_len = offsets[1] - offsets[0]
        f.seek(data_start)
        raw = f.read(data_len)

    dtype_map = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64}
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype_str}")
    arr = np.frombuffer(raw, dtype=dtype_map[dtype_str])
    if dtype_str == "BF16":
        # BF16: zero-extend lower 16 bits to get f32 bit pattern
        arr = arr.astype(np.uint32) << 16
        arr = arr.view(np.float32)
    arr = arr.reshape(shape)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# validate: check artifact shapes and tokenizer consistency; repair where safe
# ---------------------------------------------------------------------------

def validate(out_dir: Path, repair: bool = False) -> None:
    model_path = out_dir / "model.safetensors"
    tokenizer_path = out_dir / "tokenizer.json"

    if not model_path.exists():
        _fail(f"model.safetensors not found in {out_dir}")
    if not tokenizer_path.exists():
        _fail(f"tokenizer.json not found in {out_dir}")

    # ── Check tensor ─────────────────────────────────────────────────────────
    header = read_safetensors_header(model_path)
    non_meta = {k: v for k, v in header.items() if not k.startswith("__")}
    if len(non_meta) != 1:
        _fail(
            f"Expected exactly 1 tensor in model.safetensors, found {len(non_meta)}: "
            f"{list(non_meta)}"
        )
    tensor_name = list(non_meta)[0]
    if tensor_name != "embeddings":
        _fail(
            f"Expected tensor named 'embeddings', found {tensor_name!r}. "
            f"ese expects the single tensor to be named 'embeddings'."
        )
    meta = non_meta[tensor_name]
    dtype = meta["dtype"]
    shape = meta["shape"]
    if dtype != "F32":
        _fail(f"Expected F32 dtype, found {dtype}. ese requires F32 embeddings.")
    if len(shape) != 2:
        _fail(f"Expected 2D tensor [vocab, dims], found shape {shape}.")
    vocab_size_tensor, dims = shape
    print(f"  tensor: {tensor_name}, dtype={dtype}, shape={shape}  [OK]")

    # ── Check tokenizer ───────────────────────────────────────────────────────
    tk = json.loads(tokenizer_path.read_text())
    model_type = tk.get("model", {}).get("type", "unknown")
    if model_type not in ("WordPiece", "Unigram"):
        _fail(
            f"model '{out_dir.name}' uses a '{model_type}' tokenizer; ese compiles WordPiece "
            f"and Unigram (SentencePiece) models only — BPE, WordLevel, and other types are "
            f"not supported (see docs/rfc-generalization.md)."
        )
    print(f"  tokenizer family: {model_type}  [OK]")

    # ── Vocab size consistency ────────────────────────────────────────────────
    vocab_size_tk = _get_vocab_size(tk)
    if vocab_size_tk != vocab_size_tensor:
        _fail(
            f"Vocab size mismatch: tokenizer has {vocab_size_tk} tokens but "
            f"model.safetensors has {vocab_size_tensor} rows. These must match."
        )
    print(f"  vocab_size={vocab_size_tk}, dims={dims}  [OK]")

    # ── Specials-id consistency (the model2vec compaction bug) ────────────────
    if model_type == "WordPiece":
        _check_wordpiece_specials(tk, out_dir, tokenizer_path, repair)

    print(f"  specials-id consistency  [OK]")
    print(f"validate: all checks passed for {out_dir}/")


def _get_vocab_size(tk: dict) -> int:
    model = tk.get("model", {})
    model_type = model.get("type", "unknown")
    if model_type == "WordPiece":
        vocab = model.get("vocab", {})
        return len(vocab)
    elif model_type == "Unigram":
        vocab = model.get("vocab", [])
        return len(vocab)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def _get_wordpiece_vocab(tk: dict) -> dict[str, int]:
    return {k: int(v) for k, v in tk["model"]["vocab"].items()}


def _check_wordpiece_specials(
    tk: dict, out_dir: Path, tokenizer_path: Path, repair: bool
) -> None:
    vocab = _get_wordpiece_vocab(tk)
    pp = tk.get("post_processor")
    if pp is None:
        return

    pp_type = pp.get("type", "")
    specials_map = pp.get("special_tokens", {})
    if not specials_map:
        return

    repaired = False
    for token in ("[CLS]", "[SEP]"):
        entry = specials_map.get(token)
        if entry is None:
            continue
        ids = entry.get("ids")
        if not ids:
            continue
        pp_id = ids[0]
        vocab_id = vocab.get(token)
        if vocab_id is None:
            continue  # token not in vocab; caught by build.rs separately
        if pp_id != vocab_id:
            if not repair:
                _fail(
                    f"post_processor stores id {pp_id} for {token} but vocab maps "
                    f"{token} to {vocab_id} — internally inconsistent (known model2vec "
                    f"compaction bug). Run distill.py --check to auto-repair, or re-distill."
                )
            # Repair: rewrite the stored id to the vocab's id
            print(
                f"  REPAIR: {token} post_processor id {pp_id} → {vocab_id} "
                f"(vocab compaction fix)"
            )
            entry["ids"][0] = vocab_id
            # Also fix the token string if it has a 'tokens' list
            if "tokens" in entry:
                entry["tokens"][0] = token
            repaired = True

    if repaired:
        tokenizer_path.write_text(json.dumps(tk, ensure_ascii=False, indent=2))
        print(f"  wrote repaired tokenizer.json to {tokenizer_path}")


def _fail(msg: str) -> None:
    print(f"VALIDATE ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# export_fidelity: G3 oracle — compare model2vec .encode() vs manual mean-pool
# ---------------------------------------------------------------------------

def export_fidelity(out_dir: Path) -> None:
    from model2vec import StaticModel

    SENTENCES = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can encode semantic meaning.",
        "Static embeddings are efficient for production use.",
        "The capital of France is Paris.",
        "Natural language processing is a subfield of AI.",
        "The model converts text to dense vector representations.",
        "Cosine similarity measures the angle between vectors.",
        "Information retrieval systems use embeddings for semantic search.",
        "The embedding dimension determines the vector space size.",
        "Pre-tokenization splits text into words before subword segmentation.",
        "Hello world",
        "café résumé naïve",
        "你好世界",
        "UPPER CASE text",
        "multiple   spaces   between",
        "a quick brown cat",
        "semantic search at scale",
        "rust programming language",
        "open source software",
        "embeddings capture word relationships",
    ]

    print(f"\nG3 export fidelity check ({len(SENTENCES)} sentences) ...")
    model = StaticModel.from_pretrained(str(out_dir))
    tokenizer_path = out_dir / "tokenizer.json"
    model_path = out_dir / "model.safetensors"

    # Get model2vec reference encodings
    ref_embeddings = model.encode(SENTENCES, show_progressbar=False)

    # Get the raw embedding matrix from safetensors
    W = read_safetensors_tensor(model_path, "embeddings")  # [vocab, dims]

    # Load the tokenizer
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(tokenizer_path))

    tk_json = json.loads(tokenizer_path.read_text())
    model_type = tk_json.get("model", {}).get("type", "unknown")

    # Determine add_special_tokens — model2vec always encodes without specials
    # (its tokenize() method passes add_special_tokens=False)
    add_special_tokens = False

    # Get UNK token id — model2vec filters UNK tokens from the id list
    unk_token_id = None
    if model_type == "WordPiece":
        unk_token = tk_json.get("model", {}).get("unk_token", "[UNK]")
        vocab = tk_json.get("model", {}).get("vocab", {})
        unk_token_id = vocab.get(unk_token)

    min_cosine = float("inf")
    all_ok = True
    for i, sentence in enumerate(SENTENCES):
        # Manual mean-pool + L2 normalize
        # Mirror model2vec's tokenize(): add_special_tokens=False, filter UNK
        encoding = tk.encode(sentence, add_special_tokens=False)
        ids = list(encoding.ids)
        if unk_token_id is not None:
            ids = [tid for tid in ids if tid != unk_token_id]
        if ids:
            rows = W[ids]  # [n_tokens, dims]
            vec = rows.mean(axis=0)
        else:
            vec = np.zeros(W.shape[1], dtype=np.float32)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        ref = ref_embeddings[i].astype(np.float32)
        ref_norm = np.linalg.norm(ref)
        if ref_norm > 0:
            ref = ref / ref_norm

        cosine = float(np.dot(vec, ref) / (np.linalg.norm(vec) * np.linalg.norm(ref) + 1e-12))
        if cosine < 0.999999:
            all_ok = False
        min_cosine = min(min_cosine, cosine)
        status = "OK" if cosine >= 0.999999 else "FAIL"
        print(f"  [{status}] cosine={cosine:.8f}  {sentence[:60]!r}")

    print(f"\n  min cosine = {min_cosine:.8f}  (threshold: 0.999999)")
    if not all_ok:
        print("G3 FAILED: some sentences below cosine threshold 0.999999", file=sys.stderr)
        sys.exit(1)
    print("G3 PASSED: all sentences cosine >= 0.999999")


def _derive_add_special_tokens(tk: dict) -> bool:
    pp = tk.get("post_processor")
    if pp is None:
        return False
    pp_type = pp.get("type", "")
    if pp_type == "BertProcessing":
        return True
    if pp_type == "TemplateProcessing":
        def has_cls_sep(arr):
            if not arr:
                return False
            for item in arr:
                st = item.get("SpecialToken")
                if st and st.get("id") in ("[CLS]", "[SEP]"):
                    return True
            return False
        single = pp.get("single", [])
        pair = pp.get("pair", [])
        return has_cls_sep(single) or has_cls_sep(pair)
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill an HF sentence-transformer into ese-compatible artifacts."
    )
    parser.add_argument(
        "teacher",
        nargs="?",
        help="HF model ID to distill (e.g. BAAI/bge-base-en-v1.5)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory for model.safetensors + tokenizer.json",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=256,
        help="PCA output dimensions (default: 256)",
    )
    parser.add_argument(
        "--check",
        type=Path,
        metavar="DIR",
        help="Validate an existing artifact directory (no distillation).",
    )
    parser.add_argument(
        "--fidelity",
        type=Path,
        metavar="DIR",
        help="Run G3 export fidelity oracle on an existing artifact directory.",
    )

    args = parser.parse_args()

    if args.check is not None:
        print(f"Validating {args.check} ...")
        validate(args.check, repair=True)
        return

    if args.fidelity is not None:
        export_fidelity(args.fidelity)
        return

    if args.teacher is None:
        parser.error("Provide a teacher model ID or use --check / --fidelity.")
    if args.out is None:
        parser.error("--out is required when distilling.")

    distill(args.teacher, args.out, args.dims)


if __name__ == "__main__":
    main()
