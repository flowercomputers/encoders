# CoIR (10 tasks) nDCG@10 for the currently installed `ese` wheel.
# Protocol as in the CoIR benchmark: test-split qrels, full corpus, cosine
# similarity, nDCG@10 averaged over queries. CodeSearchNet and
# CodeSearchNet-CCR are each averaged over their six languages.
# Usage: uv run --with <wheel> coir.py <out.json>
import json
import sys
import time

import numpy as np
from datasets import load_dataset

import ese

LANGS = ["go", "java", "javascript", "php", "python", "ruby"]
TASKS = {
    "apps": ["apps"],
    "cosqa": ["cosqa"],
    "synthetic-text2sql": ["synthetic-text2sql"],
    "codesearchnet": [f"CodeSearchNet-{l}" for l in LANGS],
    "codesearchnet-ccr": [f"CodeSearchNet-ccr-{l}" for l in LANGS],
    "codetrans-contest": ["codetrans-contest"],
    "codetrans-dl": ["codetrans-dl"],
    "stackoverflow-qa": ["stackoverflow-qa"],
    "codefeedback-st": ["codefeedback-st"],
    "codefeedback-mt": ["codefeedback-mt"],
}
K = 10
DISCOUNT = 1.0 / np.log2(np.arange(2, K + 2))


def embed(texts):
    v = np.asarray(ese.encode(list(texts)), dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def ndcg10(name):
    qc = load_dataset(f"CoIR-Retrieval/{name}-queries-corpus")
    qrels = load_dataset(f"CoIR-Retrieval/{name}-qrels")["test"]
    rel = {}
    for q, c, s in zip(qrels["query_id"], qrels["corpus_id"], qrels["score"]):
        if s > 0:
            rel.setdefault(q, {})[c] = s
    corpus, queries = qc["corpus"], qc["queries"]
    qtext = dict(zip(queries["_id"], queries["text"]))
    qids = [q for q in rel if q in qtext]
    cids = corpus["_id"]
    t0 = time.perf_counter()
    C = embed(corpus["text"])
    Q = embed([qtext[q] for q in qids])
    t_embed = time.perf_counter() - t0
    scores = []
    for s in range(0, len(qids), 512):
        sims = Q[s : s + 512] @ C.T
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        order = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, 1), 1), 1)
        for i, row in enumerate(order):
            r = rel[qids[s + i]]
            gains = np.array([r.get(cids[j], 0) for j in row], dtype=np.float64)
            dcg = ((2**gains - 1) * DISCOUNT).sum()
            ideal = np.sort(np.array(list(r.values()), dtype=np.float64))[::-1][:K]
            idcg = ((2**ideal - 1) * DISCOUNT[: len(ideal)]).sum()
            scores.append(dcg / idcg)
    return float(np.mean(scores)), len(qids), len(cids), t_embed


out = {"dimensions": ese.DIMENSIONS, "tasks": {}, "subsets": {}}
for task, subsets in TASKS.items():
    vals = []
    for name in subsets:
        score, nq, nc, te = ndcg10(name)
        out["subsets"][name] = {"ndcg@10": score, "queries": nq, "corpus": nc, "embed_s": te}
        print(f"  {name:28s} nDCG@10 {score:.4f}  ({nq} queries, {nc} docs, embed {te:.1f}s)", flush=True)
        vals.append(score)
    out["tasks"][task] = float(np.mean(vals))
    print(f"{task:30s} {out['tasks'][task]:.4f}", flush=True)
out["mean"] = float(np.mean(list(out["tasks"].values())))
print(f"CoIR mean nDCG@10 {out['mean']:.4f}")
json.dump(out, open(sys.argv[1], "w"), indent=2)
