"""
G1 validation: EseEncoder (Python) matches embed_stdin (compiled binary)
at per-sentence cosine >= 0.999 on a 10-sentence probe.

Run from benches/py/:
    uv run --with tokenizers python validate_g1.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from ese_encoder import validate_g1, get_model_configs, EseModelConfig

# 10-sentence probe (varied English)
PROBE = [
    "The cat sat quietly on the warm windowsill.",
    "Machine learning models can encode semantic meaning.",
    "The stock market fell sharply this afternoon.",
    "She baked a loaf of sourdough bread on Sunday.",
    "Mount Everest is the tallest mountain on Earth.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Quantum computers exploit superposition and entanglement.",
    "Electric vehicles are becoming increasingly affordable.",
    "The committee voted to approve the new budget.",
    "Volcanic eruptions can alter the global climate.",
]

if __name__ == "__main__":
    cfgs = get_model_configs()
    cfg = cfgs["default"]
    print(f"G1 validation for: {cfg.display_name}")
    print(f"  artifact_dir: {cfg.artifact_dir}")
    print(f"  dims={cfg.dims}, normalize={cfg.normalize}, skip_unk={cfg.skip_unk}, "
          f"add_special_tokens={cfg.add_special_tokens}")
    print(f"  cargo_feature: {cfg.cargo_feature}")
    ok = validate_g1(cfg, PROBE, threshold=0.999)
    if ok:
        print("\nG1: PASS")
        sys.exit(0)
    else:
        print("\nG1: FAIL")
        sys.exit(1)
