"""
ese_encoder.py — Python EseEncoder for NanoBEIR quality evaluation.

Reuses the faithful pooling proven in:
  - spikes/specials-parity/check.py  (WordPiece mean-pool mirror)
  - distill/crosscheck.py            (Unigram mean-pool mirror, ESE_SKIP_UNK semantics)

The EseEncoder is sentence-transformers-compatible (.encode() returns numpy arrays)
and is validated against the compiled ese binary (embed_stdin) at cosine ≥ 0.999
before being used for NanoBEIR scoring (G1 gate).

Model config:
    Each EseModelConfig specifies:
      - artifact_dir: path to model.safetensors + tokenizer.json
      - dims: int — embedding dimension (must match artifact)
      - normalize: bool — L2-normalize output (matches ESE_NORMALIZE_OUTPUT=1)
      - skip_unk: bool — exclude UNK-id tokens from mean pool (ESE_SKIP_UNK=1,
                        used for model2vec distilled artifacts)
      - add_special_tokens: bool — include [CLS]/[SEP] in mean pool
                             (ESE_ADD_SPECIAL_TOKENS=1; default is False)
      - tokenizer_type: "WordPiece" | "Unigram"

G1 validation (run via ese_encoder_validate.py or inline below):
    Compare EseEncoder.encode() for 10 sentences against embed_stdin output.
    All per-sentence cosines must be ≥ 0.999.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Safetensors reader (single-tensor files)
# ---------------------------------------------------------------------------

def _read_embedding_matrix(path: Path) -> np.ndarray:
    """Read the single embedding tensor from a safetensors file as float32.

    Handles tensor names 'embeddings' (model2vec artifacts) and
    'embedding.weight' (static-retrieval-mrl / NIFE).
    Converts F16/BF16 to F32.
    """
    data = path.read_bytes()
    mlen = struct.unpack("<Q", data[:8])[0]
    meta = json.loads(data[8: 8 + mlen])
    non_meta = {k: v for k, v in meta.items() if not k.startswith("__")}
    assert len(non_meta) == 1, f"Expected 1 tensor, got {list(non_meta)}"
    key = list(non_meta)[0]
    t = non_meta[key]
    dtype = t["dtype"]
    rows, dims = t["shape"]
    start = 8 + mlen + t["data_offsets"][0]
    end = 8 + mlen + t["data_offsets"][1]
    raw = data[start:end]

    if dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4").reshape(rows, dims).copy()
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").reshape(rows, dims).astype(np.float32)
    elif dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").reshape(rows, dims)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F64":
        arr = np.frombuffer(raw, dtype="<f8").reshape(rows, dims).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return arr


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class EseModelConfig:
    key: str                          # short identifier, e.g. "default"
    display_name: str                 # human-readable name for table
    artifact_dir: Path                # directory with model.safetensors + tokenizer.json
    dims: int                         # embedding dimension
    normalize: bool = False           # ESE_NORMALIZE_OUTPUT
    skip_unk: bool = False            # ESE_SKIP_UNK
    add_special_tokens: bool = False  # ESE_ADD_SPECIAL_TOKENS
    # Optional cargo feature string for G1 validation (e.g. "dim-1024")
    cargo_feature: str = "dim-512"
    # Model family
    family: str = "ese"
    # Vocab size (auto-filled)
    vocab: int = 0
    # tokenizer type (auto-filled)
    tokenizer_type: str = ""
    # NanoBEIR teacher model id (for G3 retention)
    teacher_model_id: Optional[str] = None
    # Teacher NanoBEIR score (from model card or measured)
    teacher_nanobeir: Optional[float] = None
    # Whether teacher score is from model card
    teacher_score_source: str = ""


# ---------------------------------------------------------------------------
# Pooling helpers (mirrors spikes/specials-parity/check.py + distill/crosscheck.py)
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row. Zero vectors stay zero."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x / norms


def _mean_pool(ids_list: list[list[int]], matrix: np.ndarray) -> np.ndarray:
    """Mean-pool rows from matrix for each list of token ids."""
    out = np.zeros((len(ids_list), matrix.shape[1]), dtype=np.float64)
    for i, ids in enumerate(ids_list):
        if len(ids) == 0:
            continue
        out[i] = matrix[ids].mean(axis=0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# EseEncoder: sentence-transformers-compatible wrapper
# ---------------------------------------------------------------------------

class EseEncoder:
    """Python reimplementation of ese's pooling for quality evaluation.

    Validated against the compiled ese binary at cosine ≥ 0.999 (G1).
    Used for NanoBEIR scoring (G2) without requiring per-model Rust wheels.
    """

    def __init__(self, cfg: EseModelConfig):
        self.cfg = cfg
        artifact_dir = cfg.artifact_dir

        # Load embedding matrix (truncated to cfg.dims, mirroring build.rs trunc_dims())
        _full_matrix = _read_embedding_matrix(artifact_dir / "model.safetensors")
        if cfg.dims < _full_matrix.shape[1]:
            self._matrix = _full_matrix[:, : cfg.dims]
        else:
            self._matrix = _full_matrix

        # Load tokenizer
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))

        # Parse tokenizer metadata
        tk_json = json.loads((artifact_dir / "tokenizer.json").read_text())
        self._tokenizer_type = tk_json.get("model", {}).get("type", "WordPiece")
        cfg.tokenizer_type = self._tokenizer_type

        # Determine UNK id (for skip_unk)
        self._unk_id: Optional[int] = None
        if cfg.skip_unk:
            if self._tokenizer_type == "WordPiece":
                unk_token = tk_json.get("model", {}).get("unk_token", "[UNK]")
                vocab = tk_json.get("model", {}).get("vocab", {})
                self._unk_id = vocab.get(unk_token)
            elif self._tokenizer_type == "Unigram":
                self._unk_id = tk_json.get("model", {}).get("unk_id", 0)

        # Vocab size
        if self._tokenizer_type == "WordPiece":
            cfg.vocab = len(tk_json.get("model", {}).get("vocab", {}))
        elif self._tokenizer_type == "Unigram":
            cfg.vocab = len(tk_json.get("model", {}).get("vocab", []))

        # Dimensions
        self._dims = self._matrix.shape[1]
        # NanoBEIR evaluator uses this
        self.max_seq_length = 512
        self._dimensions = cfg.dims
        self.similarity_fn_name = "cosine"
        self.model_card_data = type(
            "CardData", (), {"set_evaluation_metrics": lambda *a, **k: None}
        )()

    def _encode_sentences(self, sentences: list[str]) -> np.ndarray:
        """Core pooling: tokenize + mean pool. Mirrors ese's build-time logic."""
        cfg = self.cfg
        # Tokenize
        encodings = self._tokenizer.encode_batch(
            sentences, add_special_tokens=cfg.add_special_tokens
        )
        ids_list = [list(e.ids) for e in encodings]

        # Skip UNK tokens if configured (model2vec behavior)
        if cfg.skip_unk and self._unk_id is not None:
            ids_list = [
                [tid for tid in ids if tid != self._unk_id]
                for ids in ids_list
            ]

        # Mean pool
        embs = _mean_pool(ids_list, self._matrix)

        # L2 normalize if configured
        if cfg.normalize:
            embs = _l2_normalize(embs)

        return embs.astype(np.float32)

    def encode(
        self,
        sentences,
        batch_size: int = 256,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """sentence-transformers-compatible encode()."""
        if isinstance(sentences, str):
            sentences = [sentences]

        # Process in batches to avoid memory issues on large corpora
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = list(sentences[i: i + batch_size])
            results.append(self._encode_sentences(batch))

        embs = np.concatenate(results, axis=0)

        # NanoBEIREvaluator may request normalization on its own
        if normalize_embeddings:
            embs = _l2_normalize(embs)

        return embs

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimensions

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix, used by NanoBEIREvaluator."""
        import torch
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b)
        a = torch.nn.functional.normalize(a.float(), p=2, dim=1)
        b = torch.nn.functional.normalize(b.float(), p=2, dim=1)
        return a @ b.T

    def encode_query(self, query, **kwargs):
        return self.encode(query, **kwargs)

    def encode_document(self, document, **kwargs):
        return self.encode(document, **kwargs)


# ---------------------------------------------------------------------------
# G1 validation helper
# ---------------------------------------------------------------------------

def validate_g1(cfg: EseModelConfig, probe: list[str], threshold: float = 0.999) -> bool:
    """Validate EseEncoder matches compiled embed_stdin binary at cosine >= threshold.

    Returns True if all per-sentence cosines meet the threshold.
    Raises RuntimeError if the binary fails to run.
    """
    encoder = EseEncoder(cfg)
    py_embs = encoder._encode_sentences(probe)

    # Build and run embed_stdin
    import os
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
    if cfg.add_special_tokens:
        env["ESE_ADD_SPECIAL_TOKENS"] = "1"
    else:
        env.pop("ESE_ADD_SPECIAL_TOKENS", None)

    inp = "\n".join(probe) + "\n"
    cmd = [
        "cargo", "run", "--quiet", "--example", "embed_stdin",
        "--features", cfg.cargo_feature,
    ]
    result = subprocess.run(cmd, cwd=str(REPO), input=inp,
                            capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"embed_stdin failed:\n{result.stderr}")

    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    if len(lines) != len(probe):
        raise RuntimeError(f"Expected {len(probe)} lines, got {len(lines)}")

    bin_embs = np.array(
        [[float(x) for x in line.split()] for line in lines], dtype=np.float32
    )

    # Compute per-sentence cosines
    py_n = _l2_normalize(py_embs.astype(np.float64))
    bin_n = _l2_normalize(bin_embs.astype(np.float64))
    cosines = (py_n * bin_n).sum(axis=1)

    print(f"\n[G1] EseEncoder vs embed_stdin ({cfg.key}, {cfg.cargo_feature}):")
    all_pass = True
    for i, (c, sent) in enumerate(zip(cosines, probe)):
        status = "OK" if c >= threshold else "FAIL"
        if c < threshold:
            all_pass = False
        print(f"  [{status}] cosine={c:.6f}  {sent[:60]!r}")
    print(f"  min cosine = {cosines.min():.6f}  threshold = {threshold}  "
          f"{'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ESE_CACHE = REPO / "target/ese-cache"
DISTILL_OUT = REPO / "target/distilled"

def _cache(name: str) -> Path:
    """Find a cache directory by partial name match."""
    for p in ESE_CACHE.iterdir():
        if p.is_dir() and name in p.name:
            return p
    raise FileNotFoundError(f"No cache dir matching {name!r} in {ESE_CACHE}")


def get_model_configs() -> dict[str, EseModelConfig]:
    """Return all ese model configs for the NanoBEIR run."""
    configs = {}

    # 1. Default: static-retrieval-mrl-en-v1
    #    WordPiece, dim=1024 (or 512 with default feature), no normalize, no skip_unk
    #    The default compiled binary uses dim-512 (Cargo.toml default feature)
    #    We use dim-512 to match the default binary for G1.
    try:
        default_dir = _cache("sentence-transformers-static-retrieval-mrl-en-v1")
        configs["default"] = EseModelConfig(
            key="default",
            display_name="static-retrieval-mrl-en-v1",
            artifact_dir=default_dir,
            dims=512,
            normalize=False,
            skip_unk=False,
            add_special_tokens=False,
            cargo_feature="dim-512",
            family="ese/static",
        )
    except FileNotFoundError:
        print("WARNING: default model cache not found", file=sys.stderr)

    # 2. NIFE-mxbai-embed-large-v1
    #    WordPiece, dim=1024, normalize=True (the model card specifies normalized)
    try:
        nife_dir = _cache("stephantulkens-NIFE-mxbai-embed-large-v1")
        configs["nife"] = EseModelConfig(
            key="nife",
            display_name="NIFE-mxbai-embed-large-v1",
            artifact_dir=nife_dir,
            dims=1024,
            normalize=True,
            skip_unk=False,
            add_special_tokens=False,
            cargo_feature="dim-1024",
            family="ese/NIFE",
        )
    except FileNotFoundError:
        print("WARNING: NIFE model cache not found", file=sys.stderr)

    # 3. potion-multilingual-128M
    #    Unigram, dim=256, normalize=True, skip_unk=True
    try:
        potion_ml_dir = _cache("minishlab-potion-multilingual-128M")
        configs["potion-multilingual"] = EseModelConfig(
            key="potion-multilingual",
            display_name="potion-multilingual-128M",
            artifact_dir=potion_ml_dir,
            dims=256,
            normalize=True,
            skip_unk=True,
            add_special_tokens=False,
            cargo_feature="dim-256",
            family="ese/potion",
        )
    except FileNotFoundError:
        print("WARNING: potion-multilingual-128M cache not found", file=sys.stderr)

    # 3b. potion-base-8M (plan 017 G1 — Tokenlearn'd English model)
    #     WordPiece, dim=256, normalize=True, skip_unk=True
    try:
        potion8m_dir = _cache("minishlab-potion-base-8M")
        configs["potion-base-8m"] = EseModelConfig(
            key="potion-base-8m",
            display_name="potion-base-8M",
            artifact_dir=potion8m_dir,
            dims=256,
            normalize=True,
            skip_unk=True,
            add_special_tokens=False,
            cargo_feature="dim-256",
            family="ese/potion",
        )
    except FileNotFoundError:
        print("WARNING: potion-base-8M cache not found", file=sys.stderr)

    # 4. tokenlearn-minilm (plan 017 G3 — Tokenlearn-trained MiniLM distill)
    #    WordPiece, dim=256, normalize=True, skip_unk=True
    tl_minilm_dir = DISTILL_OUT / "tokenlearn-minilm"
    if tl_minilm_dir.exists():
        configs["tokenlearn-minilm"] = EseModelConfig(
            key="tokenlearn-minilm",
            display_name="tokenlearn-minilm (gooaq 100k)",
            artifact_dir=tl_minilm_dir,
            dims=256,
            normalize=True,
            skip_unk=True,
            add_special_tokens=False,
            cargo_feature="dim-256",
            family="ese/tokenlearn",
            teacher_model_id="sentence-transformers/all-MiniLM-L6-v2",
        )
    else:
        print(f"INFO: tokenlearn-minilm not found at {tl_minilm_dir}", file=sys.stderr)

    # 5-8. Distilled models (in target/distilled/)
    distill_models = [
        ("bge-base", "BAAI/bge-base-en-v1.5", "bge-base-en-v1.5", "ese/distill"),
        ("arctic-xs", "Snowflake/snowflake-arctic-embed-xs", "arctic-embed-xs", "ese/distill"),
        ("minilm", "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", "ese/distill"),
        ("multilingual-e5", "intfloat/multilingual-e5-base", "multilingual-e5-base", "ese/distill"),
    ]
    for key, teacher_id, short_name, family in distill_models:
        distill_dir = DISTILL_OUT / short_name
        if distill_dir.exists():
            configs[key] = EseModelConfig(
                key=key,
                display_name=f"distilled-{short_name}",
                artifact_dir=distill_dir,
                dims=256,
                normalize=True,
                skip_unk=True,
                add_special_tokens=False,
                cargo_feature="dim-256",
                family=family,
                teacher_model_id=teacher_id,
            )
        else:
            print(f"INFO: distilled {short_name} not found at {distill_dir}", file=sys.stderr)

    return configs
