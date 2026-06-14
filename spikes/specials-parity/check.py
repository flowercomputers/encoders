"""Specials-parity spike — does ese's default model match the real reference?

Prompted by model2vec maintainer feedback (MinishLab/model2vec#335): the
sentence-transformers static reference "explicitly does not insert special
tokens", and the right way to measure the impact of a pooling choice is the
similarity matrix over a corpus, not single-pair cosine.

This script answers two questions for `static-retrieval-mrl-en-v1` (ese's
DEFAULT compiled model):

  Q1 (correctness): which pooling reproduces real SentenceTransformer.encode()
      — mean WITH [CLS]/[SEP] (what ese does today) or mean WITHOUT them?
  Q2 (impact, maintainer's method): how different are the upper-triangular
      similarity matrices S of the two choices, and which is closer to the
      reference's S?

It also validates the numpy "with specials" pooling against the actual ese
binary (examples/embed_stdin built at dim-1024) so the comparison is not
self-referential.

Run:  cd spikes/specials-parity && \
      uv run --with sentence-transformers --with safetensors --with numpy \
              --with tokenizers python check.py
"""

import json
import struct
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/static-retrieval-mrl-en-v1"
CACHE = REPO / "target/ese-cache/sentence-transformers-static-retrieval-mrl-en-v1-f60985c706f192d45d218078e49e5a8b6f15283a"
SAFETENSORS = CACHE / "model.safetensors"
TOKENIZER = CACHE / "tokenizer.json"

# A small, varied English corpus (deterministic, no randomness).
CORPUS = [
    "The cat sat quietly on the warm windowsill.",
    "A dog barked loudly at the passing mail truck.",
    "Machine learning models can encode semantic meaning.",
    "Neural networks learn representations from raw data.",
    "The stock market fell sharply this afternoon.",
    "Investors worried about rising interest rates.",
    "She baked a loaf of sourdough bread on Sunday.",
    "The recipe calls for flour, water, salt, and yeast.",
    "Mount Everest is the tallest mountain on Earth.",
    "The Pacific Ocean is the largest body of water.",
    "He repaired the leaking faucet in the kitchen.",
    "The plumber tightened the valve under the sink.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Plants absorb carbon dioxide and release oxygen.",
    "The orchestra performed a symphony by Beethoven.",
    "Violins and cellos filled the concert hall with sound.",
    "The spacecraft entered orbit around the red planet.",
    "Astronomers detected water ice near the lunar pole.",
    "The chef seasoned the soup with fresh basil.",
    "A pinch of salt brings out the flavor of tomatoes.",
    "The marathon runner crossed the finish line first.",
    "Training for an ultramarathon takes months of effort.",
    "Quantum computers exploit superposition and entanglement.",
    "Researchers built a fifty-qubit processor last year.",
    "The library closes at nine o'clock on weekdays.",
    "Students borrowed textbooks before the exam period.",
    "Rainforests host an enormous diversity of species.",
    "Deforestation threatens countless tropical animals.",
    "The novel explores themes of memory and loss.",
    "Her poetry captures the quiet ache of nostalgia.",
    "Electric vehicles are becoming increasingly affordable.",
    "Battery costs have dropped dramatically over a decade.",
    "The committee voted to approve the new budget.",
    "Lawmakers debated the proposal late into the night.",
    "Volcanic eruptions can alter the global climate.",
    "Ash clouds grounded flights across the continent.",
    "The toddler stacked colorful wooden blocks.",
    "Children laughed as the puppet show began.",
    "Cryptographers rely on hard mathematical problems.",
    "A strong password resists brute-force attacks.",
    "The river flooded the valley after heavy rain.",
    "Engineers reinforced the levee with sandbags.",
    "Bees pollinate a third of the crops we eat.",
    "Pollinator decline endangers global food supplies.",
    "The museum unveiled a rare Renaissance painting.",
    "Art historians dated the canvas to the fifteenth century.",
    "He debugged the program until well past midnight.",
    "A single off-by-one error crashed the server.",
    "The glacier retreated several meters this summer.",
    "Melting ice raises sea levels worldwide.",
]


def read_embedding_matrix(path: Path) -> np.ndarray:
    data = path.read_bytes()
    mlen = struct.unpack("<Q", data[:8])[0]
    meta = json.loads(data[8 : 8 + mlen])
    key = next(k for k in meta if not k.startswith("__"))
    t = meta[key]
    assert t["dtype"] == "F32", t["dtype"]
    rows, dims = t["shape"]
    start = 8 + mlen + t["data_offsets"][0]
    buf = data[start : start + rows * dims * 4]
    return np.frombuffer(buf, dtype="<f4").reshape(rows, dims).copy()


def mean_pool(ids_list, matrix) -> np.ndarray:
    out = np.zeros((len(ids_list), matrix.shape[1]), dtype=np.float64)
    for i, ids in enumerate(ids_list):
        if len(ids) == 0:
            continue
        out[i] = matrix[ids].mean(axis=0)
    return out


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def per_row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = l2(a), l2(b)
    return (a * b).sum(axis=1)


def upper_tri_sim(x: np.ndarray) -> np.ndarray:
    xn = l2(x)
    S = xn @ xn.T
    iu = np.triu_indices(S.shape[0], k=1)
    return S[iu]


def ese_binary_embeddings() -> np.ndarray:
    """Run the corpus through the actual ese binary (dim-1024 default build)."""
    inp = "\n".join(CORPUS) + "\n"
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "embed_stdin", "--features", "dim-1024"],
        cwd=str(REPO),
        input=inp,
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [list(map(float, line.split())) for line in proc.stdout.strip().split("\n")]
    return np.array(rows, dtype=np.float64)


def main():
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer

    matrix = read_embedding_matrix(SAFETENSORS)
    print(f"embedding matrix: {matrix.shape}  (rows x dims)")

    tok = Tokenizer.from_file(str(TOKENIZER))
    ids_with = [tok.encode(s, add_special_tokens=True).ids for s in CORPUS]
    ids_without = [tok.encode(s, add_special_tokens=False).ids for s in CORPUS]

    pooled_with = mean_pool(ids_with, matrix)        # ese's current behavior
    pooled_without = mean_pool(ids_without, matrix)  # the maintainer's claim

    # --- Reference: the real sentence-transformers model ---
    st = SentenceTransformer(MODEL, device="cpu")
    ref = st.encode(CORPUS, convert_to_numpy=True, normalize_embeddings=False).astype(np.float64)
    print(f"reference (ST.encode) shape: {ref.shape}, mean L2 norm: {np.linalg.norm(ref, axis=1).mean():.4f}")

    # --- Validation: numpy 'with specials' must mirror the ese binary ---
    print("\n[validate] numpy(with-specials) vs ese binary (dim-1024):")
    ese_bin = ese_binary_embeddings()
    val = per_row_cosine(pooled_with, ese_bin)
    print(f"  per-sentence cosine: min={val.min():.6f} mean={val.mean():.6f}  "
          f"(>=0.9999 confirms the numpy mirror is faithful)")

    # --- Q1: which pooling reproduces the reference? ---
    print("\n[Q1 correctness] cosine(candidate, ST.encode reference), per sentence:")
    for name, cand in [("mean WITH [CLS]/[SEP]", pooled_with),
                       ("mean WITHOUT specials", pooled_without)]:
        c = per_row_cosine(cand, ref)
        print(f"  {name:24s}: min={c.min():.4f} mean={c.mean():.4f} median={np.median(c):.4f}")

    # --- Q2: maintainer's similarity-matrix method ---
    print("\n[Q2 impact] upper-triangular similarity matrix S over the corpus:")
    s_ref = upper_tri_sim(ref)
    s_with = upper_tri_sim(pooled_with)
    s_without = upper_tri_sim(pooled_without)
    print(f"  mean|S_ref - S_with|    = {np.abs(s_ref - s_with).mean():.5f}   "
          f"(ese today vs reference geometry)")
    print(f"  mean|S_ref - S_without| = {np.abs(s_ref - s_without).mean():.5f}   "
          f"(specials-off vs reference geometry)")
    print(f"  mean|S_with - S_without|= {np.abs(s_with - s_without).mean():.5f}   "
          f"(impact of the specials choice itself)")
    # Spearman-style: how much does the choice reorder neighbor similarities?
    print(f"  corr(S_with, S_without) = {np.corrcoef(s_with, s_without)[0,1]:.5f}")
    print(f"  corr(S_ref,  S_with)    = {np.corrcoef(s_ref, s_with)[0,1]:.5f}")
    print(f"  corr(S_ref,  S_without) = {np.corrcoef(s_ref, s_without)[0,1]:.5f}")

    print("\n[verdict] whichever candidate has higher Q1 cosine AND smaller "
          "Q2 |S_ref - S| is the pooling ese SHOULD use for this model.")


if __name__ == "__main__":
    main()
