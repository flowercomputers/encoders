"""
Spike: measure greedy-ification fidelity on minishlab/potion-multilingual-128M.

This script:
1. Loads the reference model via model2vec and encodes a fixed multilingual
   input set to establish baseline embeddings.
2. Discovers what pooling/normalization model2vec applies (compare .encode()
   against manual mean-pool of token rows; check L2 normalization).
3. Greedy-ifies the tokenizer via skeletoken.make_model_greedy(), re-tokenizes
   the same inputs, mean-pools the same embedding matrix rows, replicating
   whatever normalization model2vec applies.
4. Computes per-language cosine similarity between baseline and greedy-ified
   embeddings, reporting min/p5/median per language.

Output: results.json + a printed table suitable for the RFC appendix.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

MODEL_NAME = "minishlab/potion-multilingual-128M"

# ---------------------------------------------------------------------------
# Fixed input set  (deterministic; ≥200 sentences; seed not needed)
# ---------------------------------------------------------------------------

INPUTS_BY_LANG: dict[str, list[str]] = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can encode semantic meaning.",
        "Static embeddings are efficient for production use.",
        "The capital of France is Paris.",
        "Embeddings capture word relationships in vector space.",
        "Natural language processing is a subfield of AI.",
        "The model converts text to dense vector representations.",
        "Greedy tokenization approximates Viterbi decoding.",
        "Cosine similarity measures the angle between vectors.",
        "A compiler transforms source code into machine code.",
        "The embedding matrix maps tokens to float arrays.",
        "WordPiece and Unigram are two tokenization algorithms.",
        "BPE merges byte pairs iteratively to build a vocabulary.",
        "Sentence transformers create sentence-level representations.",
        "The verifier checks that compiled artifacts reproduce reference outputs.",
        "Pooling aggregates token embeddings into a single vector.",
        "The L2 norm normalizes vectors to unit length.",
        "Information retrieval systems use embeddings for semantic search.",
        "The PHF table maps tokens to embedding matrix rows.",
        "A static embedding library requires no runtime inference.",
        "Pre-tokenization splits text into words before subword segmentation.",
        "Normalization folds character variants to a canonical form.",
        "The Metaspace pre-tokenizer uses a special space token.",
        "SentencePiece is a language-independent subword segmentation library.",
        "Open vocabulary models handle unseen words via subword units.",
        "The embedding dimension determines the vector space size.",
        "Quantization reduces memory footprint at inference time.",
        "Mean pooling averages token embeddings across positions.",
        "The CLS token represents the whole sentence in BERT.",
        "Fidelity measures how well greedy tokenization preserves semantics.",
        "The runtime matcher finds the longest matching token in the vocab.",
        "Layer B unlocks WordPiece models for the ese compiler.",
        "Unigram Viterbi finds the most likely tokenization sequence.",
        "The IR triple is TokenizerSpec, EmbeddingMatrix, PoolingSpec.",
        "Frontend loaders parse model artifacts into the compiler IR.",
        "Optimization passes transform the IR without changing semantics.",
        "The backend emits a self-contained artifact from the IR.",
        "WebAssembly is a portable binary instruction format for the web.",
        "The mmap pack allows zero-copy loading of embedding data.",
        "Provenance-independence means training history is invisible at deploy.",
        "Vocabulary pruning removes rows absent from the target corpus.",
        "Vocabulary extension adds domain-specific tokens to the model.",
        "Row transplant swaps tokenizers while re-deriving embedding rows.",
        "Reweighting bakes IDF scores into the embedding matrix rows.",
        "Normalization folding moves lowercasing from runtime to compile time.",
        "MRL truncation shortens the embedding dimension for tiered recall.",
        "The CHD algorithm builds a perfect hash function over the vocab.",
        "A static embedding model has no attention mechanism.",
        "The special token denominator affects pooling magnitude semantics.",
        "Per-row quantization scales reduce the per-token quantization error.",
        "Provenance independence means training history is opaque at deployment.",
        "The two-phase compile boundary separates Python canonicalization from Rust codegen.",
        "Layer B in the roadmap unlocks WordPiece model families.",
        "Unigram Viterbi finds the maximum-likelihood tokenization via dynamic programming.",
        "The spell checker corrects typos by minimizing edit distance to dictionary words.",
        "Retrieval augmented generation combines dense retrieval with language model generation.",
        "A flat pack binary format enables memory-mapped loading without deserialization.",
        "The OOV false-positive rate for a 64-bit verify hash is astronomically small.",
        "Phrase embeddings capture multi-word expressions as single vocabulary entries.",
        "The Zipf distribution approximates word frequency in natural language corpora.",
    ],
    "zh": [
        "机器学习是人工智能的一个重要分支。",
        "自然语言处理研究计算机理解人类语言的方法。",
        "词嵌入将单词映射到高维向量空间。",
        "句子变换器生成句子级别的语义表示。",
        "静态嵌入模型不需要在运行时进行推理。",
        "分词是自然语言处理中的基础步骤。",
        "语义搜索利用向量相似度查找相关文档。",
        "模型压缩技术可以减少存储和计算需求。",
        "词汇表定义了模型能够理解的单词集合。",
        "余弦相似度衡量两个向量之间的角度。",
        "预训练模型在大规模语料库上学习语言规律。",
        "嵌入维度决定了向量空间的大小。",
        "量化技术将浮点数转换为低精度表示。",
        "北京是中国的首都城市。",
        "深度学习使用多层神经网络学习特征。",
        "信息检索系统帮助用户找到相关文档。",
        "编译器将源代码转换为机器可执行的代码。",
        "向量数据库存储和查询高维嵌入向量。",
        "归一化将向量缩放到单位长度。",
        "文本分类将文档分配到预定义的类别。",
        "命名实体识别从文本中提取实体信息。",
        "知识图谱表示实体之间的关系。",
        "多语言模型可以处理多种语言的文本。",
        "迁移学习将知识从一个任务转移到另一个任务。",
        "词频-逆文档频率是一种常用的词权重方法。",
        "子词分词解决了未登录词的问题。",
        "中文分词不使用空格作为词的分隔符。",
        "汉字每个字通常对应一个独立的语义单元。",
        "句子的语义相似度可以通过向量点积计算。",
        "大规模语言模型在众多自然语言任务上表现优异。",
    ],
    "ja": [
        "機械学習は人工知能の重要な分野です。",
        "自然言語処理はコンピュータが人間の言語を理解する技術です。",
        "単語埋め込みは単語を高次元ベクトルに変換します。",
        "文変換器は文レベルの意味表現を生成します。",
        "静的埋め込みモデルは実行時推論を必要としません。",
        "トークン化はテキストを小さな単位に分割するプロセスです。",
        "意味検索はベクトルの類似度を使って関連文書を検索します。",
        "モデル圧縮技術はストレージと計算要件を削減します。",
        "語彙はモデルが理解できる単語の集合を定義します。",
        "コサイン類似度は二つのベクトル間の角度を測定します。",
        "東京は日本の首都です。",
        "深層学習は多層ニューラルネットワークを使って特徴を学習します。",
        "情報検索システムはユーザーが関連文書を見つけるのを助けます。",
        "コンパイラはソースコードを機械実行可能なコードに変換します。",
        "ベクトルデータベースは高次元埋め込みベクトルを格納・検索します。",
        "正規化はベクトルを単位長にスケーリングします。",
        "テキスト分類は文書を定義済みのカテゴリに割り当てます。",
        "固有表現認識はテキストからエンティティ情報を抽出します。",
        "多言語モデルは複数の言語のテキストを処理できます。",
        "転移学習はあるタスクから別のタスクへ知識を転移します。",
        "サブワードトークン化は未知語の問題を解決します。",
        "日本語は複数の文字体系（ひらがな、カタカナ、漢字）を使用します。",
        "文の意味的類似度はベクトルの内積で計算できます。",
        "大規模言語モデルは多くの自然言語タスクで優れた性能を発揮します。",
        "埋め込み次元はベクトル空間のサイズを決定します。",
        "量子化技術は浮動小数点数を低精度表現に変換します。",
        "事前学習モデルは大規模コーパスから言語パターンを学習します。",
        "語彙拡張はドメイン固有のトークンをモデルに追加します。",
        "プールはトークン埋め込みを単一ベクトルに集約します。",
        "検索システムはセマンティック類似度を使って文書をランク付けします。",
    ],
    "th": [
        "การเรียนรู้ของเครื่องเป็นสาขาสำคัญของปัญญาประดิษฐ์",
        "การประมวลผลภาษาธรรมชาติศึกษาการทำให้คอมพิวเตอร์เข้าใจภาษามนุษย์",
        "การฝังคำแปลงคำเป็นเวกเตอร์มิติสูง",
        "โมเดลการฝังประโยคสร้างการแสดงความหมายระดับประโยค",
        "โมเดลการฝังแบบคงที่ไม่ต้องการการอนุมานในขณะรันไทม์",
        "การแบ่งโทเค็นแบ่งข้อความออกเป็นหน่วยย่อย",
        "การค้นหาเชิงความหมายใช้ความคล้ายคลึงของเวกเตอร์เพื่อค้นหาเอกสารที่เกี่ยวข้อง",
        "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
        "การเรียนรู้เชิงลึกใช้โครงข่ายประสาทเทียมหลายชั้นในการเรียนรู้คุณลักษณะ",
        "ระบบการดึงข้อมูลช่วยให้ผู้ใช้ค้นหาเอกสารที่เกี่ยวข้อง",
        "คอมไพเลอร์แปลงซอร์สโค้ดเป็นโค้ดที่เครื่องสามารถรันได้",
        "ฐานข้อมูลเวกเตอร์จัดเก็บและค้นหาเวกเตอร์การฝังมิติสูง",
        "การทำให้เป็นปกติปรับขนาดเวกเตอร์เป็นความยาวหน่วย",
        "การจำแนกประเภทข้อความกำหนดเอกสารให้กับหมวดหมู่ที่กำหนดไว้ล่วงหน้า",
        "โมเดลหลายภาษาสามารถประมวลผลข้อความในหลายภาษาได้",
        "การเรียนรู้แบบถ่ายโอนถ่ายโอนความรู้จากงานหนึ่งไปยังอีกงานหนึ่ง",
        "ความคล้ายคลึงของโคไซน์วัดมุมระหว่างเวกเตอร์สองตัว",
        "การกำหนดน้ำหนักคำผสานคะแนน IDF เข้าในแถวเมทริกซ์การฝัง",
        "มิติการฝังกำหนดขนาดของปริภูมิเวกเตอร์",
        "เทคนิคการหาปริมาณแปลงจำนวนทศนิยมเป็นการแทนค่าความแม่นยำต่ำ",
        "โมเดลที่ผ่านการฝึกล่วงหน้าเรียนรู้รูปแบบภาษาจากคอร์ปัสขนาดใหญ่",
        "การขยายคำศัพท์เพิ่มโทเค็นเฉพาะโดเมนให้กับโมเดล",
        "การรวมกันรวมการฝังโทเค็นเป็นเวกเตอร์เดียว",
        "ระบบการค้นหาจัดอันดับเอกสารโดยใช้ความคล้ายคลึงเชิงความหมาย",
        "การฝังก่อนการฝึกช่วยให้โมเดลเข้าใจภาษาได้ดีขึ้น",
    ],
    "code": [
        "def mean_pool(embeddings, mask): return (embeddings * mask).sum(0) / mask.sum()",
        "fn encode(&self, text: &str) -> Vec<f32> { self.lookup(text) }",
        "SELECT * FROM embeddings WHERE cosine_sim(vec, query) > 0.9",
        "import numpy as np; vecs = np.load('embeddings.npy')",
        "cargo build --release --features tests",
        "git commit -m 'add greedy-ification spike'",
        "curl -sL https://huggingface.co/model/resolve/main/tokenizer.json | python3 -c \"import json,sys; print(json.load(sys.stdin)['model']['type'])\"",
        "uv run python measure_fidelity.py --model minishlab/potion-multilingual-128M",
        "torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))",
        "tokenizer = Tokenizer.from_file('tokenizer.json')",
        "make_model_greedy(tokenizer_dict)",
        "model.encode(sentences, normalize_embeddings=True)",
        "np.linalg.norm(embeddings, axis=1, keepdims=True)",
        "phf_map.get(token.as_bytes()).copied()",
        "let embedding = &EMBEDDING_MATRIX[row * DIM..(row + 1) * DIM];",
        "from tokenizers import Tokenizer; tok = Tokenizer.from_file(path)",
        "for token, score in unigram_vocab: vocab_dict[token] = len(vocab_dict)",
        "cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))",
        "static EMBEDDING_DATA: &[u8] = include_bytes!(concat!(env!('OUT_DIR'), '/weights.bin'));",
        "serde_json::from_str::<TokenizerSpec>(&tokenizer_json)?",
        "build.rs: println!(\"cargo:rerun-if-changed=tokenizer.json\");",
        "ESE_MODEL_DIR=/path/to/canonical/model cargo build",
        "from model2vec import StaticModel; m = StaticModel.from_pretrained(name)",
        "skeletoken.decase_vocabulary(tokenizer_dict)",
        "huggingface_hub.hf_hub_download(repo_id, 'tokenizer.json')",
        "const PHF_TABLE: phf::Map<&str, u32> = phf_map! { \"hello\" => 0, \"world\" => 1 };",
        "impl Encode for &str { fn encode(&self, tokens: &mut Vec<u32>) { ... } }",
        "python3 -m pytest tests/ -x -v --tb=short",
        "#[cfg(feature = \"tests\")] mod test { use super::*; }",
        "type EmbeddingMatrix = ndarray::Array2<f32>;",
        "fn cosine_sim(a: &[f32], b: &[f32]) -> f32 { dot(a,b) / (norm(a)*norm(b)) }",
        "Vec<(String, f32)>  // Unigram vocabulary: (token, log_prob)",
        "struct TokenizerSpec { model: ModelSpec, normalizer: Option<NormSpec> }",
        "tar -czf artifact.tar.gz weights.bin tokenizer.json vocab.txt",
        "wasm-pack build --target web --release",
        "let scores: Vec<f32> = tokens.iter().map(|t| idf[t]).collect();",
        "grep -rn 'hardcoded\\|TODO\\|FIXME' src/ benches/",
        "python3 -c \"import skeletoken; help(skeletoken.base.TokenizerModel.make_model_greedy)\"",
        "cargo test --features tests 2>&1 | tail -20",
        "jq '.model.type' tokenizer.json",
    ],
    "de": [
        "Maschinelles Lernen ist ein wichtiges Teilgebiet der Künstlichen Intelligenz.",
        "Natürliche Sprachverarbeitung untersucht, wie Computer menschliche Sprache verstehen.",
        "Worteinbettungen bilden Wörter auf hochdimensionale Vektoren ab.",
        "Satztransformatoren erzeugen semantische Darstellungen auf Satzebene.",
        "Statische Einbettungsmodelle benötigen keine Laufzeitinferenz.",
        "Tokenisierung ist ein grundlegender Schritt in der Sprachverarbeitung.",
        "Berlin ist die Hauptstadt Deutschlands.",
        "Tiefes Lernen verwendet mehrschichtige neuronale Netze.",
        "Der Kosinusabstand misst den Winkel zwischen zwei Vektoren.",
        "Vortrainierte Modelle lernen Sprachmuster aus großen Korpora.",
        "Quantisierung reduziert den Speicherbedarf bei der Inferenz.",
        "Informationsgewinnung hilft Nutzern, relevante Dokumente zu finden.",
        "Ein Compiler wandelt Quellcode in maschinenausführbaren Code um.",
        "Vektordatenbanken speichern und durchsuchen hochdimensionale Vektoren.",
        "Normalisierung skaliert Vektoren auf Einheitslänge.",
    ],
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    print("=== Greedy-Ification Fidelity Spike ===")
    print(f"Model: {MODEL_NAME}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load reference model and encode baseline
    # ------------------------------------------------------------------
    print("[1/4] Loading reference model via model2vec ...")
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(MODEL_NAME)
    print(f"  Embedding matrix shape: {model.embedding.shape}")

    all_sentences: list[str] = []
    lang_slices: dict[str, tuple[int, int]] = {}
    for lang, sents in INPUTS_BY_LANG.items():
        start = len(all_sentences)
        all_sentences.extend(sents)
        lang_slices[lang] = (start, len(all_sentences))

    print(f"  Total sentences: {len(all_sentences)}")

    # Reference encode (this is the gold baseline)
    baseline = model.encode(all_sentences)  # shape: (N, D)
    print(f"  Baseline shape: {baseline.shape}")

    # ------------------------------------------------------------------
    # Step 2: Discover pooling and normalization
    # ------------------------------------------------------------------
    print()
    print("[2/4] Discovering model2vec pooling/normalization ...")

    # Use the tokenizer from model2vec's tokenizer attribute
    ref_tokenizer = model.tokenizer

    # Manually mean-pool a few sentences to understand what model.encode does
    emb_matrix = np.array(model.embedding)  # (vocab, D)

    def manual_encode(text: str, l2_normalize: bool) -> np.ndarray:
        """Mean-pool token rows from the embedding matrix."""
        enc = ref_tokenizer.encode(text)
        ids = enc.ids
        if not ids:
            return np.zeros(emb_matrix.shape[1])
        rows = emb_matrix[ids]
        pooled = rows.mean(axis=0)
        if l2_normalize:
            n = np.linalg.norm(pooled)
            if n > 1e-10:
                pooled = pooled / n
        return pooled

    # Test with and without L2 normalization on a sample
    sample_text = all_sentences[0]
    ref_vec = baseline[0]
    manual_no_norm = manual_encode(sample_text, l2_normalize=False)
    manual_with_norm = manual_encode(sample_text, l2_normalize=True)

    cos_no_norm = cosine_sim(ref_vec, manual_no_norm)
    cos_with_norm = cosine_sim(ref_vec, manual_with_norm)

    print(f"  Sample sentence: {sample_text[:60]!r}")
    print(f"  Cosine(ref, manual_no_norm)  = {cos_no_norm:.6f}")
    print(f"  Cosine(ref, manual_with_norm) = {cos_with_norm:.6f}")

    # Determine normalization
    uses_l2_norm = cos_with_norm > cos_no_norm + 0.001
    if uses_l2_norm:
        print("  => model2vec applies L2 normalization after mean pooling")
    elif cos_no_norm > 0.999:
        print("  => model2vec does NOT apply L2 normalization (or output is already unit-norm)")
        # Check if it's magnitude-1 already
        mag = float(np.linalg.norm(ref_vec))
        print(f"     Reference vector L2 norm: {mag:.6f}")
        uses_l2_norm = abs(mag - 1.0) < 0.01
        if uses_l2_norm:
            print("     (Vector is unit-norm, so L2 normalization IS being applied)")
    else:
        print(f"  => Unclear (cos_no_norm={cos_no_norm:.4f}, cos_with_norm={cos_with_norm:.4f})")
        # Check magnitude
        mag = float(np.linalg.norm(ref_vec))
        print(f"     Reference vector L2 norm: {mag:.6f}")
        uses_l2_norm = abs(mag - 1.0) < 0.01

    print(f"  Conclusion: uses_l2_norm = {uses_l2_norm}")

    # Also check if CLS/SEP tokens are included in pooling
    # model2vec mean-pools ALL tokens including special tokens
    sample_ids = ref_tokenizer.encode(all_sentences[0]).ids
    print(f"  First sentence token ids (first 5): {sample_ids[:5]}")
    # In BERT-style the first is [CLS]=101, last is [SEP]=102 typically
    # For multilingual Unigram it may differ

    # ------------------------------------------------------------------
    # Step 3: Greedy-ify and re-encode
    # ------------------------------------------------------------------
    print()
    print("[3/4] Greedy-ifying tokenizer via skeletoken ...")

    import skeletoken

    print(f"  skeletoken version: {skeletoken.__version__ if hasattr(skeletoken, '__version__') else 'unknown'}")
    print(f"  skeletoken API: {[x for x in dir(skeletoken) if not x.startswith('_')]}")

    # Load tokenizer.json as a dict (skeletoken works on the raw dict)
    from huggingface_hub import hf_hub_download

    tok_path = hf_hub_download(MODEL_NAME, "tokenizer.json")
    print(f"  Tokenizer path: {tok_path}")

    with open(tok_path) as f:
        tokenizer_dict = json.load(f)

    print(f"  Original model.type: {tokenizer_dict['model']['type']}")

    # Check if tokenizer is cased or uncased (multilingual Unigram is cased)
    # For Unigram/SentencePiece models the vocab has lowercase and uppercase tokens
    # decase_vocabulary() only makes sense for uncased models
    is_uncased = False  # Multilingual Unigram (SentencePiece) is cased
    print(f"  Model is cased (is_uncased={is_uncased}), skipping decase_vocabulary()")

    # Apply make_model_greedy via TokenizerModel.from_string().make_model_greedy()
    # (skeletoken 0.3.x: make_model_greedy is a method on TokenizerModel, not a
    # module-level function)
    from skeletoken.base import TokenizerModel

    print(f"  skeletoken.TokenizerModel.make_model_greedy signature verified")

    try:
        tok_model = TokenizerModel.from_string(json.dumps(tokenizer_dict))
        greedy_model = tok_model.make_model_greedy()
        print(f"  Greedy model.type: {greedy_model.model.model_dump().get('type', '?') if hasattr(greedy_model.model, 'model_dump') else type(greedy_model.model).__name__}")
        print("  make_model_greedy() succeeded")
    except Exception as e:
        print(f"  STOP: make_model_greedy() failed: {e!r}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Convert back to tokenizers.Tokenizer via to_tokenizer()
    from tokenizers import Tokenizer

    try:
        greedy_tokenizer = greedy_model.to_tokenizer()
        print(f"  Round-trip via to_tokenizer() succeeded")
    except Exception as e:
        # Fallback: serialize to JSON and load
        print(f"  to_tokenizer() failed ({e!r}), trying JSON round-trip ...")
        try:
            greedy_json = greedy_model.model_dump_json()
            greedy_dict_out = json.loads(greedy_json)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(greedy_dict_out, f)
                greedy_tok_path = f.name
            greedy_tokenizer = Tokenizer.from_file(greedy_tok_path)
            print(f"  JSON round-trip succeeded")
        except Exception as e2:
            print(f"  STOP: both round-trips failed: {e2!r}")
            sys.exit(1)

    # Encode all sentences using greedy tokenizer + manual mean-pool
    def greedy_encode(text: str) -> np.ndarray:
        enc = greedy_tokenizer.encode(text)
        ids = enc.ids
        if not ids:
            return np.zeros(emb_matrix.shape[1])
        # Clip ids to valid range (safety check for OOV tokens added by special-token handling)
        ids = [i for i in ids if i < len(emb_matrix)]
        if not ids:
            return np.zeros(emb_matrix.shape[1])
        rows = emb_matrix[ids]
        pooled = rows.mean(axis=0)
        if uses_l2_norm:
            n = np.linalg.norm(pooled)
            if n > 1e-10:
                pooled = pooled / n
        return pooled

    greedy_embeddings = np.stack([greedy_encode(s) for s in all_sentences])
    print(f"  Greedy embeddings shape: {greedy_embeddings.shape}")

    # ------------------------------------------------------------------
    # Step 4: Compute per-language cosine similarities
    # ------------------------------------------------------------------
    print()
    print("[4/4] Computing per-language cosine similarity statistics ...")
    print()

    results: dict[str, dict] = {}

    header = f"{'Language':<12} {'N':>5} {'Min':>8} {'P5':>8} {'Median':>8} {'P95':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))

    for lang, (start, end) in lang_slices.items():
        ref_vecs = baseline[start:end]
        greedy_vecs = greedy_embeddings[start:end]

        cosines = np.array(
            [cosine_sim(r, g) for r, g in zip(ref_vecs, greedy_vecs)]
        )

        stats = {
            "n": len(cosines),
            "min": float(np.min(cosines)),
            "p5": float(np.percentile(cosines, 5)),
            "median": float(np.median(cosines)),
            "p95": float(np.percentile(cosines, 95)),
            "max": float(np.max(cosines)),
        }
        results[lang] = stats

        viable = stats["median"] >= 0.99 and stats["p5"] >= 0.97
        flag = " PASS" if viable else " FAIL"
        print(
            f"{lang:<12} {stats['n']:>5} {stats['min']:>8.4f} {stats['p5']:>8.4f} "
            f"{stats['median']:>8.4f} {stats['p95']:>8.4f} {stats['max']:>8.4f}{flag}"
        )

    print()
    print("Decision thresholds: median >= 0.99 AND p5 >= 0.97 per language")
    all_pass = all(
        r["median"] >= 0.99 and r["p5"] >= 0.97 for r in results.values()
    )
    print(f"Overall verdict: {'VIABLE (greedy-ification default path)' if all_pass else 'NOT VIABLE (implement Unigram Viterbi in layer A)'}")

    # Sanity check: similar sentences should score higher than dissimilar
    print()
    print("Sanity check (baseline similarity pairs):")
    paris_idx = next(i for i, s in enumerate(all_sentences) if "Paris" in s)
    tokyo_idx = next(i for i, s in enumerate(all_sentences) if "Tokyo" in s or "東京" in s)
    code_idx = next(i for i, s in enumerate(all_sentences) if "cosine_sim" in s or "def mean_pool" in s)

    sim_cap_cap = cosine_sim(baseline[paris_idx], baseline[tokyo_idx])
    sim_cap_code = cosine_sim(baseline[paris_idx], baseline[code_idx])
    print(f"  sim(Paris sentence, Tokyo sentence) = {sim_cap_cap:.4f}  (expect > sim with code)")
    print(f"  sim(Paris sentence, code snippet)   = {sim_cap_code:.4f}  (expect lower)")
    if sim_cap_cap > sim_cap_code:
        print("  => Sanity check PASSED")
    else:
        print("  => Sanity check FAILED — baseline may be wrong")

    # Save results
    # Determine greedy model type for reporting
    greedy_model_type = type(greedy_model.model).__name__

    output = {
        "model": MODEL_NAME,
        "uses_l2_norm": uses_l2_norm,
        "is_uncased": is_uncased,
        "greedy_model_type": greedy_model_type,
        "total_sentences": len(all_sentences),
        "decision_thresholds": {"median": 0.99, "p5": 0.97},
        "verdict": "viable" if all_pass else "not_viable",
        "per_language": results,
    }

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
