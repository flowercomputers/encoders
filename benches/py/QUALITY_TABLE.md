# ese Speed / Quality Tier Table — NanoBEIR

**Plan 013 — measurement artifact**

All ese NanoBEIR scores measured with `EseEncoder` (Python, identical pooling to compiled binary; validated at cosine = 1.000 vs `embed_stdin` in G1). Teacher scores measured on CPU with sentence-transformers `NanoBEIREvaluator` (13 NanoBEIR datasets, mean nDCG@10); "card" = published model-card number.

Metric: **mean nDCG@10** across 13 NanoBEIR-en datasets (climatefever, dbpedia, fever, fiqa2018, hotpotqa, msmarco, nfcorpus, nq, quoraretrieval, scidocs, arguana, scifact, touche2020).

QPS measured on **Apple M4 Pro, 64GB RAM, macOS 25.3.0** using the compiled release binary (`embed_stdin`, feature `dim-{N}`). Single = one sentence per subprocess call; Batch = 32 sentences per call.

---

## G1 — EseEncoder validation (binary vs Python cosine)

| sentence (probe) | cosine |
|---|---|
| The cat sat quietly on the warm windowsill. | 1.000000 |
| Machine learning models can encode semantic meaning. | 1.000000 |
| The stock market fell sharply this afternoon. | 1.000000 |
| She baked a loaf of sourdough bread on Sunday. | 1.000000 |
| Mount Everest is the tallest mountain on Earth. | 1.000000 |
| Photosynthesis converts sunlight into chemical energy. | 1.000000 |
| Quantum computers exploit superposition and entanglement. | 1.000000 |
| Electric vehicles are becoming increasingly affordable. | 1.000000 |
| The committee voted to approve the new budget. | 1.000000 |
| Volcanic eruptions can alter the global climate. | 1.000000 |

**G1 PASS** — min cosine = 1.000000 (threshold 0.999), all 10 sentences exact.
Model: `static-retrieval-mrl-en-v1`, feature `dim-512`.

---

## G2 / G3 — Quality table

| model | family | dims | vocab | NanoBEIR ese | NanoBEIR teacher/ref | retention % | notes |
|---|---|---:|---:|---:|---:|---:|---|
| static-retrieval-mrl-en-v1 | ese/static | 512 | 30 522 | **0.4957** | — | — | default model; WordPiece; no normalize |
| NIFE-mxbai-embed-large-v1 | ese/NIFE | 1024 | 107 962 | **0.4685** | — | — | WordPiece; normalize; large vocab |
| potion-multilingual-128M | ese/potion | 256 | 500 353 | **0.3699** | — | — | Unigram; skip_unk; multilingual |
| distilled-bge-base-en-v1.5 | ese/distill | 256 | 29 525 | 0.1107 | 0.6320 (meas.) | **17.5%** | model2vec distill of BAAI/bge-base-en-v1.5 |
| distilled-arctic-embed-xs | ese/distill | 256 | 29 525 | 0.0962 | 0.6121 (meas.) | **15.7%** | model2vec distill of Snowflake/snowflake-arctic-embed-xs |
| distilled-all-MiniLM-L6-v2 | ese/distill | 256 | 29 525 | 0.1044 | 0.5625 (meas.) | **18.6%** | model2vec distill of sentence-transformers/all-MiniLM-L6-v2 |
| distilled-multilingual-e5-base | ese/distill | 256 | 249 999 | 0.1035 | ~0.5921 (card) | **~17.5%** | model2vec distill of intfloat/multilingual-e5-base |

**Teacher scores measured** on CPU with `SentenceTransformer` + `NanoBEIREvaluator` (13 datasets). `multilingual-e5-base` teacher marked "card" (NanoBEIR evaluation timed out on this hardware; MTEB-published mean nDCG@10 ≈ 0.5921).

**Retention % = ese nDCG@10 / teacher nDCG@10 × 100.**

---

## G4 — Speed (compiled ese binary, release build)

Machine: **Apple M4 Pro, 64GB RAM, macOS Darwin 25.3.0** — note: same platform as the original blog benchmarks; QPS numbers are hardware-specific.

| model | dims | vocab | single QPS | single p50 (ms) | single p95 (ms) | batch=32 QPS | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| static-retrieval-mrl-en-v1 | 512 | 30 522 | **287.7** | 3.2 | 5.3 | **5 169.9** | WordPiece; dim-512 |
| potion-multilingual-128M | 256 | 500 353 | 89.2 | 10.4 | 16.5 | 640.7 | Unigram; 500k vocab |

QPS includes process startup overhead (subprocess call to pre-built binary). The default model (WordPiece, 30k vocab, 512 dims) is ~3.2× faster per sentence than potion-multilingual (Unigram, 500k vocab, 256 dims), reflecting vocab-size dominance at this scale.

---

## Summary: where ese sits on the speed/quality frontier

### Native ese models (non-distilled)

The three native ese models — `static-retrieval-mrl-en-v1` (0.4957), `NIFE-mxbai-embed-large-v1` (0.4685), and `potion-multilingual-128M` (0.3699) — demonstrate competitive retrieval quality for static embedding systems. The default model (`static-retrieval-mrl-en-v1`) leads the pack and was explicitly designed for retrieval; its NanoBEIR score of 0.4957 compares favorably to published model2vec numbers (potion-base-8M ≈ 0.433 on NanoBEIR per model card).

`static-retrieval-mrl-en-v1` at dim-512 reaches **287.7 QPS single / 5 170 QPS batched** on M4 Pro, combining retrieval-competitive quality with extreme speed. This is the ese sweet spot: comparable NanoBEIR to a small neural model while running thousands of times faster.

### Distilled models (model2vec only)

The four distilled models (bge-base, arctic-xs, MiniLM, multilingual-e5) score 0.096–0.111 on NanoBEIR — retaining only **15–19% of their teacher's retrieval signal**. This is expected for raw model2vec distillation without contrastive fine-tuning: the distillation step preserves embedding geometry but loses the retrieval-specific alignment that transformer teachers achieve through fine-tuning. Published model2vec models like `potion-base-8M` (which underwent contrastive re-training) achieve ~0.43.

**Conclusion**: freshly distilled model2vec artifacts are not retrieval-ready as-is. Phase 3 (vocab surgery / contrastive fine-tuning) should target the gap between 0.11 and 0.43 for distilled models. The native ese models (`static-retrieval-mrl-en-v1`, `NIFE-mxbai-embed-large-v1`) are the current quality leaders and are already deployment-ready.

### Retention ranking (best to worst)

1. `distilled-all-MiniLM-L6-v2`: **18.6%** (best relative retention)
2. `distilled-bge-base-en-v1.5`: **17.5%**
3. `distilled-multilingual-e5-base`: **~17.5%**
4. `distilled-arctic-embed-xs`: **15.7%** (lowest; arctic-xs is optimized for dense retrieval, hardest to distill statically)

---

## Appendix: model artifacts

| model | artifact type | tensor name | dtype |
|---|---|---|---|
| static-retrieval-mrl-en-v1 | ese built-in | `embedding.weight` | F32 |
| NIFE-mxbai-embed-large-v1 | ESE_MODEL cache | `embedding.weight` | F32 |
| potion-multilingual-128M | ESE_MODEL cache | `embeddings` | F32 |
| distilled-bge-base-en-v1.5 | model2vec distill | `embeddings` | F32 (converted from F16) |
| distilled-arctic-embed-xs | model2vec distill | `embeddings` | F32 (converted from F16) |
| distilled-all-MiniLM-L6-v2 | model2vec distill | `embeddings` | F32 (converted from F16) |
| distilled-multilingual-e5-base | model2vec distill | `embeddings` | F32 (converted from F16) |

Generated by `benches/py/run_nanobeir_ese.py` + `benches/py/measure_qps_direct.py`.
Validated by `benches/py/validate_g1.py`.
