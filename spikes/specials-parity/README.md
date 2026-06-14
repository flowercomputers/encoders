# specials-parity spike

One-off diagnostic prompted by model2vec maintainer feedback
([MinishLab/model2vec#335](https://github.com/MinishLab/model2vec/issues/335)):
*"the sentence-transformers static reference explicitly does not insert
special tokens"* and *"to measure the impact, look at the upper-triangular
similarity matrix S with and without the special tokens."*

`check.py` tests ese's **default** model (`static-retrieval-mrl-en-v1`)
against the real `SentenceTransformer.encode()` reference. Frozen result in
`results.txt`; not part of the build/test workflow.

## Verdict (2026-06-13): the maintainer is correct

- The numpy "with specials" pooling matches the actual ese binary at cosine
  **1.000000** → the comparison is independent, not self-referential.
- Mean pooling **WITHOUT** special tokens reproduces the real reference at
  cosine **1.0000** (every sentence). ese's current **WITH**-specials pooling
  lands at **0.9994** (min 0.9987).
- Similarity-matrix geometry: `mean|S_ref − S_without| = 0.00000` vs
  `mean|S_ref − S_with| = 0.00141`; correlation to the reference 1.00000 vs
  0.99970.

**Conclusion**: sentence-transformers `StaticEmbedding` does not pool CLS/SEP.
ese has pooled them since the initial commit — a small (retrieval corr
0.9997) but real divergence from the reference, invisible to the parity
suite because the build-time oracle assumed `add_special_tokens=true` and the
runtime faithfully matched the wrong assumption. Fix tracked in
`plans/012-specials-off-by-default.md`.

To re-run:
`uv run --with sentence-transformers --with safetensors --with numpy --with tokenizers python check.py`
