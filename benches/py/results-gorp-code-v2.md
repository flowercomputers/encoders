# gorp-code-v2 results

Compiled with this crate (`ENCODERS_LOCAL_MODEL`, 256 dimensions, `quant-16`), the settings gorp ships. Throughput was measured on a 2-core x86-64 cloud VM.

| | gorp-code-v2 |
|---|---|
| CoIR mean nDCG@10 (10 tasks) | 0.424 |
| NanoBEIR mean nDCG@10 (13 tasks) | 0.452 |
| Single-query QPS (p50 latency) | 210k (4.2 µs) |
| Batched QPS (GooAQ, batch 65,536) | 327k |
| Embedded tables | 130.3 MiB |

## CoIR (nDCG@10, test qrels, full corpus)

| task | nDCG@10 |
|---|---|
| apps | 0.060 |
| cosqa | 0.236 |
| synthetic-text2sql | 0.546 |
| codesearchnet | 0.491 |
| codesearchnet-ccr | 0.444 |
| codetrans-contest | 0.499 |
| codetrans-dl | 0.314 |
| stackoverflow-qa | 0.679 |
| codefeedback-st | 0.581 |
| codefeedback-mt | 0.392 |

CodeSearchNet and CodeSearchNet-CCR are means over go, java, javascript, php, python and ruby.

## NanoBEIR (nDCG@10)

| dataset | nDCG@10 |
|---|---|
| ArguAna | 0.376 |
| ClimateFEVER | 0.241 |
| DBPedia | 0.520 |
| FEVER | 0.579 |
| FiQA2018 | 0.347 |
| HotpotQA | 0.525 |
| MSMARCO | 0.384 |
| NFCorpus | 0.245 |
| NQ | 0.391 |
| QuoraRetrieval | 0.893 |
| SCIDOCS | 0.303 |
| SciFact | 0.581 |
| Touche2020 | 0.492 |

## Fidelity

The build matches Python `model2vec` (`StaticModel.from_pretrained(dir).encode`, default `max_length=512`) on 4,920 texts (NanoBEIR corpora and queries, CoIR code and questions up to 19,497 characters, and edge cases): minimum cosine 1.000000, zero vectors identical.
