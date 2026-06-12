# RFC: ese Generalization — Compile Any Static Embedding Model

**Status**: Draft  
**Plan**: 005-generalization-roadmap.md  
**Authors**: advisor/improvements branch  
**Date**: 2026-06-12  
**Revision**: v2 (after cofounder design conversation; tokenizer facts empirically verified 2026-06-12)

---

## 1. The IR Triple and Provenance-Independence Thesis

Every static embedding model — however it was trained (model2vec distillation, contrastive training à la the HF static-embeddings recipe, GloVe co-occurrence counting, or NIFE fine-tuning) — is the same deployable artifact:

```
(TokenizerSpec, EmbeddingMatrix, PoolingSpec)
```

Training provenance is upstream and invisible at deployment. This is the **provenance-independence thesis**: ese v2 is an ahead-of-time (AOT) compiler whose intermediate representation (IR) is that triple. The frontend loads it; optional optimization passes transform it; the backend emits a self-contained artifact; a verifier enforces fidelity.

**TokenizerSpec** encodes the full HF `tokenizer.json` typed representation: model algorithm (WordPiece / Unigram / BPE / WordLevel), normalizer chain, pre-tokenizer chain, post-processor/special tokens, and the vocabulary (with Unigram log-probability scores where present).

**EmbeddingMatrix** is the float32 (or quantized) row table mapping token ids to embedding vectors.

**PoolingSpec** describes how token embeddings are combined: `mean | weighted(zipf/idf) | sif`, plus which special tokens contribute to the denominator. Plan 004 fixed the denominator semantics for the default model; the IR must carry these explicitly so optimization passes (e.g., normalization folding) preserve them.

ese's existing `build.rs` is already a single-target, single-pass version of this compiler. This RFC fixes the architecture so implementation plans compose instead of accreting one-off hacks.

---

## 2. Five-Stage Compiler Architecture

```
[HF artifacts]
     |
  FRONTEND
  --------
  weight loader:     safetensors → EmbeddingMatrix
  tokenizer loader:  tokenizer.json → TypedTokenizerSpec (validated)
     |
     v
     IR: (TokenizerSpec, EmbeddingMatrix, PoolingSpec)
     |
  OPTIMIZATION PASSES (IR → IR, all optional, composable)
  ---------------------------------------------------------
  E1. vocab prune          — drop rows absent from a target corpus
  E2. vocab extend         — add domain tokens; init rows by subword mean-pool
  E3. vocab transplant     — swap tokenizers; re-derive rows over new decompositions
  E4. vocab reweight       — bake Zipf/IDF into rows
  E5. normalization fold   — bake lowercasing into keys; shrink runtime normalizer
  E6. quantize             — upgrade global affine → per-dimension scales; binary
  E7. MRL truncate         — shorten embedding dimension for tiered recall
  E8. canonicalize         — greedy-ify (Unigram/BPE → WordPiece-compatible);
                             Python phase via skeletoken (see §5)
     |
     v
  BACKEND
  -------
  PHF builder:   CHD over vocab keys → packed table
  tokenizer codegen: emit runtime tokenizer artifact (see §matcher-decision)
  emitters:      Rust crate (today), wasm, C header, mmap .esepack (see §7)
     |
     v
  VERIFIER
  --------
  cosine parity gate:     runtime reproduces reference SentenceTransformer.encode()
  magnitude parity gate:  raw vector magnitude matches (post plan 004)
  PHF OOV note:           documented below (see §8)
```

**How today's `build.rs` maps onto this**: build.rs is a minimal frontend (fetches one model URL pair, parses WordPiece vocab) + no optimization passes + a single backend (CHD PHF + Rust const codegen) + a golden-vector verifier (the `feature = "tests"` harness). Generalizing ese means expanding each stage without coupling them.

---

## 3. Tokenizer Support Matrix

Verified against HF Hub, 2026-06-12. "Blocked on" indicates which roadmap layer is the gate.

| Model | `model.type` | Normalizer | Pre-tokenizer | Post-processor | Vocab repr | Unigram scores | Blocked on |
|---|---|---|---|---|---|---|---|
| sentence-transformers/static-retrieval-mrl-en-v1 | WordPiece | BertNormalizer | BertPreTokenizer | TemplateProcessing [CLS]/[SEP] | dict{token:id} | — | Layer B (plan 006) |
| minishlab/potion-base-2M | WordPiece | BertNormalizer | BertPreTokenizer | TemplateProcessing [CLS]/[SEP] | dict{token:id} | — | Layer B |
| minishlab/potion-base-8M | WordPiece | BertNormalizer | BertPreTokenizer | TemplateProcessing [CLS]/[SEP] | dict{token:id} | — | Layer B |
| minishlab/potion-base-32M | WordPiece | BertNormalizer | BertPreTokenizer | TemplateProcessing [CLS]/[SEP] | dict{token:id} | — | Layer B |
| minishlab/M2V_base_output | WordPiece | BertNormalizer | BertPreTokenizer | TemplateProcessing [CLS]/[SEP] | dict{token:id} | — | Layer B |
| sentence-transformers/all-MiniLM-L6-v2 | WordPiece | BertNormalizer | BertPreTokenizer | TemplateProcessing [CLS]/[SEP] | dict{token:id} | — | Layer B |
| sentence-transformers/all-mpnet-base-v2 | WordPiece | BertNormalizer (lowercase) | BertPreTokenizer | RobertaProcessing [CLS]/[SEP] | dict{token:id} | — | Layer B (note: RobertaProcessing post-proc differs from TemplateProcessing) |
| BAAI/bge-base-en-v1.5 | WordPiece | BertNormalizer | BertPreTokenizer | TemplateProcessing [CLS]/[SEP] | dict{token:id} | — | Layer B |
| MongoDB/mdbr-leaf-ir | WordPiece | BertNormalizer | BertPreTokenizer | — | dict{token:id} | — | Layer B |
| minishlab/potion-multilingual-128M | **Unigram** | Sequence(Sequence(Precompiled,…),Replace×N,Strip) | Metaspace | TemplateProcessing | list[[token,score]] | yes | Layer A (Unigram Viterbi + Metaspace pre-tok) |
| intfloat/multilingual-e5-base | **Unigram** | **Precompiled** | Sequence(WhitespaceSplit,Metaspace) | TemplateProcessing [CLS]/[SEP] | list[[token,score]] | yes | Layer A + Precompiled-normalizer decision (see §open-questions) |
| average_word_embeddings_glove.6B.300d | WordLevel-equivalent | none (no tokenizer.json) | whitespace split | none | plain vocab file | — | Layer B (WordLevel; trivial dict lookup; no tokenizer.json on HF) |
| stephantulkens/NIFE-mxbai-embed-large-v1 | WordPiece | BertNormalizer | BertPreTokenizer | no specials added (post_processor: null) | dict{token:id} | — | Layer B |
| google/embeddinggemma-300m | — | — | — | — | — | — | access-gated (401); not yet catalogued |

**Notes:**
- Layer B alone (plan 006: manifest ingestion, configurable model) already unlocks 10 of the 14 rows above (all WordPiece/BERT-family, including the now-verified NIFE row).
- The Unigram bucket (potion-multilingual, e5/xlm-r family) is the gate for Layer A.
- `potion-multilingual-128M`'s normalizer is a nested Sequence containing a Precompiled charsmap sub-normalizer, meaning it shares the hard case with e5 (Precompiled at compile time; see §open-questions).
- GloVe average_word_embeddings has no `tokenizer.json` — it is WordLevel-equivalent and requires a trivial dict lookup at runtime; Layer B can handle it with a WordLevel path.

---

## 4. The Matcher Decision

### The question

Two candidate architectures for tokenizer generality:

1. **Per-family runtimes**: implement Unigram Viterbi (and later BPE merge-rank) in Rust alongside the existing WordPiece matcher. Faithful; more runtime code; more codegen paths; Precompiled normalizers still unsolved.

2. **Greedy-ification at compile time**: use [skeletoken](https://github.com/stephantul/skeletoken) (Python, v0.3.3 verified) to convert Unigram/BPE models into greedy longest-match form at compile time (`TokenizerModel.from_string(json).make_model_greedy()`). The runtime then needs exactly one matcher — ese's existing greedy WordPiece engine — plus ~2 pre-tokenizer dialects (Metaspace, ByteLevel). Fidelity cost is measured by the spike (Appendix A).

### Spike results (Appendix A summary)

Target model: `minishlab/potion-multilingual-128M` (Unigram, Metaspace, vocab=500 353, dim=256).  
Input set: 200 sentences across English (60), Chinese (30), Japanese (30), Thai (25), code/identifiers (40), German (15).  
Reference: `model2vec.StaticModel.from_pretrained(…).encode(…)` — mean-pool all token rows (including specials), then L2-normalize.  
Greedy-ified: `TokenizerModel.from_string(tok_json).make_model_greedy().to_tokenizer()` → tokenize → mean-pool same rows → L2-normalize.

| Language | N | Min | P5 | Median | P95 | Max | Pass? |
|---|---|---|---|---|---|---|---|
| en | 60 | 0.7756 | 0.9007 | 0.9867 | 0.9926 | 0.9948 | **FAIL** (p5 < 0.97) |
| zh | 30 | 0.8534 | 0.8954 | 1.0000 | 1.0000 | 1.0000 | **FAIL** (p5 < 0.97) |
| ja | 30 | 0.8814 | 0.9071 | 1.0000 | 1.0000 | 1.0000 | **FAIL** (p5 < 0.97) |
| th | 25 | 0.8674 | 0.8926 | 1.0000 | 1.0000 | 1.0000 | **FAIL** (p5 < 0.97) |
| code | 40 | 0.7233 | 0.7338 | 0.9358 | 1.0000 | 1.0000 | **FAIL** (both) |
| de | 15 | 0.7178 | 0.7508 | 0.9419 | 0.9932 | 0.9957 | **FAIL** (both) |

**Observation on CJK/Thai medians**: median=1.0 occurs because CJK ideographs and Thai characters each occupy individual Unigram vocabulary entries — both the Viterbi segmenter and the greedy segmenter arrive at the same single-character tokenization for those inputs, producing identical token sequences and thus cosine=1.0. The p5 failures arise from the minority of inputs where multi-character tokens (compounds, loanwords, romanized spans) are segmented differently.

**Decision thresholds** (from the plan): median ≥ 0.99 AND p5 ≥ 0.97 per language → viable. No language passes on p5. The worst cases are Code (p5=0.73) and German (p5=0.75), where compound words and identifier strings are fragmented differently under greedy longest-match vs. Viterbi.

### Recommendation

**Layer A must implement Unigram Viterbi natively in Rust.** Greedy-ification at default-path quality is not viable for this model family: a p5 cosine of 0.73 for code inputs represents an unacceptable worst-case retrieval quality regression.

Greedy-ification is retained as an **optional optimization pass** (E8) for use cases where:
- Speed of compilation matters more than perfect fidelity.
- The model family has been empirically verified to tolerate the fidelity cost (e.g., a future English-only Unigram model with a non-compound vocabulary might pass the thresholds).
- Vocab transplant (E3) re-derives rows after greedy-ification to recover fidelity (measuring this recovery is future work).

skeletoken v0.3.3's API note: `make_model_greedy()` is a method on `TokenizerModel`, not a module-level function. The correct call is `TokenizerModel.from_string(json_str).make_model_greedy()`. The result is a `TokenizerModel` with `model.type = WordPiece`. `to_tokenizer()` round-trips successfully to a `tokenizers.Tokenizer` object.

**Per-model Cargo features were rejected** as a model-selection mechanism: Cargo features must be additive, so per-model features collide under dependency unification (two crates depending on ese with different model features produce a build that silently uses the union, not the intended model). The correct interface is runtime env vars (`ESE_MODEL_DIR` from plan 006) read at compile time by `build.rs`.

---

## 5. Two-Phase Compile Boundary

The Python canonicalization phase (skeletoken) and the Rust build phase (`build.rs`) are separated at the **`ESE_MODEL_DIR` seam** (introduced by plan 006):

```
Phase 1 (Python, run once per new model):
  [raw HF artifacts: tokenizer.json + safetensors]
      |
  skeletoken canonicalize:
    - parse TokenizerModel
    - for Unigram/BPE: apply make_model_greedy() if in optimization mode
    - emit canonical_tokenizer.json (WordPiece-compatible if greedy-ified,
      or full Unigram/BPE spec for the Viterbi path)
    - emit vocab.json (token → id map, Unigram log-probs stripped or retained)
      |
  write to ESE_MODEL_DIR/

Phase 2 (Rust, cargo build):
  [ESE_MODEL_DIR/canonical_tokenizer.json + *.safetensors]
      |
  build.rs frontend → IR → optimization passes → PHF + codegen → artifact
```

**Hard constraint**: pure-Rust consumers never run Python. The default model ships pre-canonicalized (its canonical artifacts are checked in or bundled). `ESE_MODEL_DIR` is the only seam — if not set, build.rs falls back to the bundled default model artifacts.

**Skeletoken hashing is roadmap-only**: as of 2026-06-12, skeletoken does not implement tokenizer equivalence hashing or cache-key functionality. Do not make the artifact cache depend on it.

---

## 6. Optimization-Pass Layer E

All passes operate on the IR triple and are optional/composable. Sequencing note: passes that reduce the vocabulary (prune, transplant) should run before quantization (to avoid quantizing rows that will be dropped).

### E1. Vocab Prune
Drop embedding rows whose tokens are absent from a target corpus. Reduces artifact size; enables higher-quality quantization of the retained rows.

### E2. Vocab Extend
Add domain-specific tokens (single words or phrases) not in the base vocabulary. Initialize new rows by mean-pooling the embedding rows of the token's subword decomposition under the current tokenizer. This directly attacks fragmentation: a phrase like `"machine_learning"` becomes a single lookup instead of `["machine", "##_", "##learning"]`.

### E3. Vocab Transplant
Swap the tokenizer (e.g., replace Unigram with a greedy WordPiece vocab) while re-deriving embedding rows: for each token in the new vocabulary, decompose it under the old tokenizer, mean-pool the old rows. Under mean pooling, transplant is exact in expectation. This is the recovery mechanism for greedy-ification: apply E8 (canonicalize) then E3 (transplant) to recover fidelity.

### E4. Vocab Reweight
Bake per-token IDF or Zipf weights into the embedding rows. Replaces the SIF/weighted pooling runtime with plain mean-pool of pre-weighted rows.

### E5. Normalization Fold (Decase)
If the model is uncased, bake lowercasing into vocabulary keys at compile time: merge `"Apple"` and `"apple"` rows, remove the lowercase step from the runtime normalizer. Shrinks the vocabulary and removes a runtime normalizer step. Use `TokenizerModel.decase_vocabulary()` in skeletoken for the tokenizer side; apply corresponding row merge to the embedding matrix.

### E6. Quantize (Upgrade)
The current quantization is a global affine scale (8-bit). Measured data (plan 002):
- Global 8-bit quantization step ≈ 1.25 (measured on the default model).
- CLS/SEP token embedding norms ≈ 22 vs. typical token norms ≈ 424.

This large norm variance means global quantization over-quantizes CLS/SEP and under-quantizes typical tokens. Upgrade path: per-row (per-token) affine scales. Each row gets its own `(scale, zero_point)` pair, stored as a parallel float array. The runtime dequantizes on lookup rather than at query time. Binary quantization (1-bit + Hamming distance) is a further option for high-throughput retrieval.

### E7. MRL Truncate
Matryoshka Representation Learning (MRL) models can be truncated: use only the first `k` dimensions of each embedding vector. This reduces the artifact size by `dim/k` and allows tiered recall (fast coarse search at small `k`, precise rerank at full `k`).

### E8. Canonicalize (Greedy-ify)
Convert Unigram or BPE tokenizer to greedy longest-match WordPiece format via skeletoken's `make_model_greedy()`. See §4 for the fidelity data and when this is appropriate. Pairs with E3 (transplant) for fidelity recovery.

---

## 7. Emitters

### Rust crate (today)
PHF-packed embedding data + const-evaluated lookup table + `encode()` function. No changes needed for the static API.

### WebAssembly (headline target)
Target: `wasm32-unknown-unknown` via `wasm-pack`. Constraints: no std IO in the wasm module; all model data embedded at compile time; the lookup function is pure. The PHF+const approach compiles cleanly to wasm today; the blocker is the build-time Python dependency (plan 006's ESE_MODEL_DIR seam isolates it).

### C header emitter
Emit a `ese.h` + `ese.c` pair (or `ese.h` + object file) for FFI consumers. The embedding data becomes a C array literal or a `.data` section blob.

### mmap `.esepack` (flat pack)
A binary format for memory-mapped loading without deserialization. Design constraints:
- `src/lookup.rs` currently transmutes to `#[repr(C, align(8))]` statics — the embedding matrix is a flat `&[u8]` that is transmuted to `&[f32]`. An mmap loader must **copy into 8-byte-aligned memory** before the transmute (mmap base alignment is page-level but the offset within the page may not be 8-byte aligned for the matrix).
- A bounds check on the mmap slice is required before transmute.
- Keep the static API (compile-time embedding); add a **separate dynamic module** for the `bake`/`load` deployment shape. The two shapes must not share a single module because `include_bytes!` and mmap cannot coexist cleanly.

---

## 8. Verifier Contract

Every compiled artifact must satisfy:

1. **Cosine parity**: for a fixed set of golden inputs, `runtime.encode(input)` must achieve cosine ≥ threshold vs. reference `SentenceTransformer.encode(input)`. Current practice: the `feature = "tests"` build computes these at build time via the `tokenizers` crate and stores them as test vectors.

2. **Magnitude parity** (post plan 004): for unquantized builds, the raw vector magnitude must match the reference within tolerance. Quantized builds track cosine only (magnitude compression is expected).

3. **PHF OOV false positives**: the PHF table uses a 64-bit verify hash to reject OOV tokens. False positives (an OOV token matches a stored token's verify hash) are astronomically rare (probability ≈ 2⁻⁶⁴ per token per lookup) but nonzero. The verifier does not test for this; it is documented here as a known probabilistic edge case. For paranoid applications, a full string comparison can be added after the verify-hash check.

4. **Greedy-ification fidelity**: if pass E8 is applied, the verifier must measure cosine between greedy-ified output and reference Unigram output (not just between the artifact and a pre-greedy-ified reference). The per-language table from §4 is the baseline for this gate.

---

## 9. Non-Goals

- **No transformer inference at runtime.** ese produces embedding vectors; it does not run attention layers or other transformer components. If a user needs transformer-quality contextual embeddings, they should use a different tool.
- **No training framework.** ese compiles existing models; it does not train or fine-tune them.
- **No vector database.** ese produces embeddings; storing, indexing, and querying them is out of scope.
- **Word-order and negation blindness are inherent**, not bugs. Mean-pooling over a token multiset discards positional information by construction. The answer to "my retrieval quality is poor on negations/ordering-sensitive queries" is: tiers (use a re-ranker), better distillation quality, or vocab surgery (phrase rows reduce fragmentation). This cannot be fixed by changing the ese architecture without abandoning static embeddings.

---

## 10. Open Questions

### 10.1 BPE demand timeline
No BPE model in the current benchmark set requires immediate support. The main BPE consumer would be RoBERTa/GPT-derived models and ModernBERT-based embedders. Until there is a concrete model request, BPE support (Layer A BPE merge-rank engine) is deferred. When it lands, skeletoken's `make_model_greedy()` applies to BPE as well — the fidelity measurement should be repeated for the specific BPE model before recommending greedy-ification vs. native BPE.

### 10.2 Precompiled normalizer handling
Both `minishlab/potion-multilingual-128M` and `intfloat/multilingual-e5-base` use a `Precompiled` normalizer (SentencePiece charsmap). This normalizer encodes hundreds of Unicode normalization rules as a compiled blob; it is effectively unreimplementable faithfully at runtime in Rust.

Options:
- **Fold at compile time**: during Phase 1 (Python/skeletoken), apply the Precompiled normalizer to every vocabulary token offline, emit a pre-normalized vocabulary, and replace the Precompiled normalizer in the canonical spec with a simpler equivalent (e.g., a Lowercase + NFC chain). This is the preferred path if the fold is lossless for the vocabulary.
- **Reject**: refuse to compile models with Precompiled normalizers, with a clear error message. Accept quality loss or use a greedy-ified/transplanted version.
- **Stub**: at runtime, apply a best-effort Unicode normalization (NFC + NFKC) that approximates the Precompiled normalizer for the common case.

This decision is left open because it requires measuring the delta between `Precompiled` and its approximations on the actual vocabularies of the affected models.

### 10.3 Who hosts canonicalized and distilled artifacts
The Python canonicalization phase (Phase 1) produces artifacts that can be checked in, uploaded to HF, or distributed via a CDN. The question is who owns these artifacts:
- The ese project could host canonical versions of popular models on HF under a dedicated org.
- Users could run Phase 1 themselves and point `ESE_MODEL_DIR` at the output.
- A CI job could regenerate canonical artifacts on a schedule.

This decision affects the default-model update workflow and the public API contract for `ESE_MODEL_DIR`.

---

## Appendix A: Spike Results — Greedy-Ification Fidelity on potion-multilingual-128M

**Script**: `spikes/greedy-fidelity/measure_fidelity.py`  
**Model**: `minishlab/potion-multilingual-128M` (Unigram, Metaspace, 500 353-token vocab, 256-dim)  
**Reference**: `model2vec.StaticModel.from_pretrained(…).encode(…)` — mean-pools all token rows (including CLS/SEP) then L2-normalizes output vectors (verified: reference output is unit-norm)  
**Greedy path**: `skeletoken.TokenizerModel.from_string(tok_json).make_model_greedy().to_tokenizer()` → tokenize → mean-pool same EmbeddingMatrix rows → L2-normalize  
**Input set**: 200 sentences — English (60), Chinese (30), Japanese (30), Thai (25), code/identifiers (40), German (15)  
**skeletoken version**: 0.3.3  
**Date run**: 2026-06-12  

| Language | N | Min | P5 | Median | P95 | Max |
|---|---|---|---|---|---|---|
| en | 60 | 0.7756 | 0.9007 | 0.9867 | 0.9926 | 0.9948 |
| zh | 30 | 0.8534 | 0.8954 | 1.0000 | 1.0000 | 1.0000 |
| ja | 30 | 0.8814 | 0.9071 | 1.0000 | 1.0000 | 1.0000 |
| th | 25 | 0.8674 | 0.8926 | 1.0000 | 1.0000 | 1.0000 |
| code | 40 | 0.7233 | 0.7338 | 0.9358 | 1.0000 | 1.0000 |
| de | 15 | 0.7178 | 0.7508 | 0.9419 | 0.9932 | 0.9957 |

**Sanity check passed**: cosine(Paris-sentence, Tokyo-sentence) = 0.605 > cosine(Paris-sentence, code-snippet) = -0.10. The baseline correctly ranks topically similar sentences above unrelated ones.

**Verdict**: greedy-ification is NOT viable as the default path (p5 < 0.97 for all languages; worst case p5 = 0.73 for code/German). Recommendation: implement Unigram Viterbi in layer A. Greedy-ification retained as optional pass E8 (see §4 and §6).

**skeletoken API note**: In v0.3.3, `make_model_greedy()` is an instance method on `TokenizerModel`, not a module-level function. Correct usage:
```python
from skeletoken.base import TokenizerModel
greedy = TokenizerModel.from_string(open("tokenizer.json").read()).make_model_greedy()
tokenizer = greedy.to_tokenizer()  # returns tokenizers.Tokenizer
```
The converted model's type is `WordPiece` (greedy longest-match over the full Unigram vocabulary). `decase_vocabulary()` is also an instance method on `TokenizerModel`; apply only for uncased models. The multilingual Unigram (SentencePiece-derived) is cased — `decase_vocabulary()` was not applied in this spike.
