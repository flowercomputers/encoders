# From One Model to Any Model

### How we generalized the world's fastest static embedding model into a compiler — and what we learned from the people who corrected us along the way.

A while ago we wrote about [the fastest static embedding model in the world](https://www.flowercomputer.com/news/fast-static-embedding/): a Rust crate that bakes an embedding table and tokenizer directly into your binary, replaces the hash map with a perfect hash, and turns text embedding into something that feels like an ordinary function call — millions of embeddings per second, no model load, no runtime, no GPU.

It had one limitation we were honest about and quietly embarrassed by: it compiled exactly **one** model. The pipeline was hard-wired to a single set of weights. If you wanted a different model, you edited a URL in a build script and hoped its tokenizer happened to match the one we'd hand-written.

This is the story of teaching it to compile *anything* — and of how the most valuable course-corrections came from outside our own heads.

## The reframe: it was always a compiler

The work started not with code but with a conversation. Our cofounder Sam sat down with Claude and asked a deceptively simple question: what would it take to turn *any* model into one of these?

The answer that emerged reframed the whole project. `ese` was never really "a fast model." It was a **compiler** — it just had one compilable program. And like any compiler, it has an intermediate representation. For a static embedding model, that IR is a triple:

> **(tokenizer, embedding matrix, pooling)**

Everything else — whether the model was distilled from a transformer, contrastively trained from scratch, or built from word co-occurrence counts — is *provenance*. It happens upstream and is invisible at deployment. Once you see the artifact as that triple, generalization stops being a research problem and becomes an engineering one: make each stage of the compiler read its configuration instead of hard-coding it.

That conversation also flagged something we'd shipped without noticing — a subtle discrepancy in how we pooled tokens. We filed it away. It would come back.

## Making each stage general

With the IR as our north star, the build-out was methodical. Model selection became configuration — `ESE_MODEL=repo@revision` — with pinned revisions and checksums so a build is reproducible and tamper-evident. The model's facts (dimensions, special tokens, normalization) are now *derived* from its artifacts, not assumed.

The hard part, as predicted, was the tokenizers. There are three families that cover essentially the entire modern embedding landscape, and each needs a genuinely different runtime algorithm:

- **WordPiece** (BERT-family) — greedy longest-match. What we already had.
- **Unigram** (SentencePiece, the multilingual world) — a Viterbi search over a scored lattice.
- **BPE** (RoBERTa, GPT-2-derived, ModernBERT) — ordered merge-rank application over a byte-level alphabet.

We added all three. The crucial discovery was that they slot in *behind the same core* — the same perfect hash, the same baked-weight table, the same two-instruction lookup. A new tokenizer is a new front-end; the back-end that makes `ese` fast never changes. Each one was implemented by reading the reference tokenizer's source line by line and mirroring its exact semantics, then proving the match against it.

Then we closed the loop on "any model" with a distillation front-end: point it at any Hugging Face transformer, and [model2vec](https://github.com/MinishLab/model2vec) turns it into the static triple `ese` compiles. We verified it on a basket of real models — including Snowflake's `arctic-embed-xs`, the very transformer an early user had benchmarked us against.

## The correction that made our oldest claim true

While building the distillation tooling, we noticed that one published model's tokenizer was internally inconsistent, and we filed a friendly bug report against model2vec. The maintainer, Stéphan Tulkens, replied — and gently corrected us on two counts.

First: the inconsistency was harmless in practice, because neither model2vec nor the sentence-transformers reference ever actually inserts those tokens. Second, and more useful: the *test* we'd used to measure the impact was the wrong instrument. Comparing two single embeddings tells you almost nothing, because adding a token shifts the whole space. The right way to measure whether a pooling choice matters is the **similarity matrix over a corpus** — does the *relative* geometry, the thing retrieval actually depends on, change?

So we took his method and pointed it at our own flagship model. And there it was: `ese` had been pooling the `[CLS]` and `[SEP]` special tokens into every embedding **since the very first commit** — and the real reference doesn't. Our outputs matched the true reference at cosine `0.9994`; without the spurious tokens, they matched at `1.0000`. The similarity-matrix error went from `0.00141` to `0.00000`.

It was a real bug, hiding in the one place our own tests were structurally blind to — our oracle had been validating the runtime against the *same wrong assumption*. We fixed it. And in doing so, we made the central claim of our original blog post — "no degradation in accuracy" — finally, exactly true. The most valuable code review of the entire effort came from a stranger's two-paragraph GitHub comment. We replied, owned it, and thanked him.

## Fast is not the same as good

Generalizing the compiler raised an uncomfortable question we'd been able to ignore with one hand-tuned model: when you compile an *arbitrary* model, is it any good?

We built a measurement rig — NanoBEIR retrieval scores across everything we could now compile — and the answer was sobering. A *raw* distillation of a strong transformer retained only **15–19%** of its retrieval quality. The good static models — the one we shipped, the `potion` family — weren't raw distillations at all. They had been **contrastively post-trained**.

So we brought that training into the pipeline. Using model2vec's Tokenlearn recipe, a laptop CPU took a distilled model from `0.10` to `0.40` NanoBEIR — a **3.8× quality lift in 95 seconds**. We were suspicious of a result that good, so we checked it the hard way: the lift held just as strongly on scientific-retrieval datasets utterly unlike the training corpus. It was real. "Any model → a *fast* static model" had quietly become "any model → a fast *and good* static model."

## Anywhere

Finally, we put it in a browser. A compiled model runs as WebAssembly with no server and no download-at-runtime — and building for a new target immediately flushed out *another* latent bug: a hash that truncated to 32 bits, invisible on our 64-bit machines, that would have silently corrupted every lookup on the web or on embedded hardware. With the model quantized to 8 bits and Matryoshka-truncated, the whole thing — weights and all — is **4.6 MB**, down from 60, while keeping 93% of its retrieval quality. Sub-millisecond semantic search, in a tab.

## The throughline

The architecture — a compiler over a simple IR — is what let all of this compose. Adding Unigram didn't disturb WordPiece; adding a browser target didn't touch the runtime; better distillation required no compiler changes at all, because it only changes how the matrix gets *populated*, which the IR was designed to forget.

But the deeper lesson isn't architectural. It's that the findings that mattered most came from *outside our own confidence*. A cofounder's reframe turned a one-off into a compiler. A maintainer's correction found a bug our own tests were built to miss. A new compile target found another. The work itself — eighteen reviewed steps, every claim backed by a measurement — was done in close collaboration with Claude, which wrote the plans, ran the experiments, and, more than once, caught itself.

`ese` now compiles any model in any of the three major tokenizer families, turns any transformer into a fast and high-quality static model, and runs anywhere — from a server to a browser tab to a Raspberry Pi. The compiler, it turns out, was always the point. We just had to listen to the people who could see it more clearly than we could.
