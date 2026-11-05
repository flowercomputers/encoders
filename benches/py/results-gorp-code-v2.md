# gorp-code-v2 vs potion-code-16M-v2

Both compiled with this crate (`ENCODERS_LOCAL_MODEL`, 256 dimensions, `quant-16`), the settings gorp ships. Measured on a 2-core x86-64 cloud VM; throughput is only comparable within this table.

| | gorp-code-v2 | potion-code-16M-v2 | Δ |
|---|---|---|---|
| CoIR mean nDCG@10 (10 tasks) | **0.424** | 0.400 | +0.024 |
| NanoBEIR mean nDCG@10 (13 tasks) | **0.452** | 0.407 | +0.045 |
| Single-query QPS (p50) | 210k (4.2 µs) | 535k (1.6 µs) | |
| Batched QPS (GooAQ, batch 65,536) | 327k | 638k | |
| Embedded tables | 130.3 MiB | 32.6 MiB | |

## CoIR (nDCG@10, test qrels, full corpus)

| task | gorp-code-v2 | potion-code-16M-v2 | Δ |
|---|---|---|---|
| apps | 0.060 | 0.052 | +0.008 |
| cosqa | 0.236 | 0.242 | -0.006 |
| synthetic-text2sql | 0.546 | 0.543 | +0.003 |
| codesearchnet | 0.491 | 0.464 | +0.027 |
| codesearchnet-ccr | 0.444 | 0.437 | +0.007 |
| codetrans-contest | 0.499 | 0.437 | +0.062 |
| codetrans-dl | 0.314 | 0.318 | -0.004 |
| stackoverflow-qa | 0.679 | 0.596 | +0.083 |
| codefeedback-st | 0.581 | 0.532 | +0.049 |
| codefeedback-mt | 0.392 | 0.380 | +0.013 |

CodeSearchNet and CodeSearchNet-CCR are means over go, java, javascript, php, python and ruby.

## NanoBEIR (nDCG@10)

| dataset | gorp-code-v2 | potion-code-16M-v2 | Δ |
|---|---|---|---|
| ArguAna | 0.376 | 0.298 | +0.078 |
| ClimateFEVER | 0.241 | 0.238 | +0.003 |
| DBPedia | 0.520 | 0.448 | +0.072 |
| FEVER | 0.579 | 0.511 | +0.067 |
| FiQA2018 | 0.347 | 0.300 | +0.047 |
| HotpotQA | 0.525 | 0.506 | +0.019 |
| MSMARCO | 0.384 | 0.361 | +0.023 |
| NFCorpus | 0.245 | 0.223 | +0.022 |
| NQ | 0.391 | 0.292 | +0.098 |
| QuoraRetrieval | 0.893 | 0.867 | +0.026 |
| SCIDOCS | 0.303 | 0.275 | +0.028 |
| SciFact | 0.581 | 0.515 | +0.066 |
| Touche2020 | 0.492 | 0.458 | +0.034 |

## Fidelity

The gorp-code-v2 build matches Python `model2vec` (`StaticModel.from_pretrained(dir).encode`, default `max_length=512`) on 4,920 texts (NanoBEIR corpora and queries, CoIR code and questions up to 19,497 characters, and edge cases): minimum cosine 1.000000, zero vectors identical.
