# Plan 017: Re-validate the speed claim on the current pipeline with a fair, reproducible harness

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 8a4616c..HEAD -- benches/ api-py/Cargo.toml bench_out/`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.
>
> **Step 0 (dispatch only)**: agent worktrees are branched from `master`, NOT
> from `advisor/improvements` (where all this work lives). Before doing
> anything else run: `git reset --hard advisor/improvements`. The worktree
> branch is fresh, so this is safe.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW (harness + docs; no runtime/compiler code changes)
- **Depends on**: none. (Its output is the recommended gate for merging `advisor/improvements` into `master`.)
- **Category**: perf / tests
- **Planned at**: commit `8a4616c`, 2026-06-13

## Why this matters

The project's central public claim — "the world's fastest static embedding
model" ([flowercomputer.com blog](https://www.flowercomputer.com/news/fast-static-embedding/)) —
is currently **not substantiated by the repo's own evidence**:

1. **The numbers are stale.** `bench_out/machine-benchmark-report.md` is dated
   `2026-04-02`. The entire generalization rewrite (native Unigram/Viterbi,
   per-row quantization, specials-off-by-default, the exact-denominator pooling
   fix) landed `2026-06-12`/`2026-06-13`. The specials-off change is explicitly
   flagged `[output change]` in `readme.md` — so even the *quality* (NanoBEIR)
   numbers describe vectors the code no longer produces.
2. **The comparison is not apples-to-apples.** `ours` is benchmarked as an
   int8-quantized, dim-512 wheel against f32 baselines at their native dims, and
   the benchmark wrapper silently ignores `normalize_embeddings`. None of this
   is disclosed in the report.
3. **The repro is broken.** The report's "Repro Commands" call `./bench.sh`,
   which does not exist in the repository.

This plan makes the harness fair and reproducible, fixes the wrapper, adds the
new model families to the matrix, and regenerates the report stamped with the
commit it describes. After this lands, the speed claim is defensible and the
`master` merge can proceed on evidence.

This plan does **not** try to make `ese` faster (that is plan 019). It only
makes the measurement honest.

## Current state

Files:
- `benches/py/bench.py` — GooAQ throughput harness. Takes a model name argv,
  measures single-query and batched QPS, writes `benchmarks/<model>_bench.json`.
- `benches/py/model_loader.py` — loads `ours` (our wheel) or a
  `SentenceTransformer`. **The wrapper drops `normalize_embeddings`.**
- `benches/py/nanobeir.py` — NanoBEIR quality harness (read it before step 4).
- `benches/py/models.txt` — the comparison model list (13 entries).
- `api-py/Cargo.toml` — the wheel build config: `features = ["rayon", "dim-512", "quant-8"]`.
- `bench_out/machine-benchmark-report.md` — the human-readable report (stale).

The wrapper that silently drops the normalize flag, `benches/py/model_loader.py:18-27`:

```python
    def encode(
        self,
        sentences,
        batch_size=65536,
        show_progress_bar=False,
        normalize_embeddings=False,
        convert_to_numpy=True,
        **kwargs,
    ):
        return ese.encode(sentences)        # <-- ignores normalize_embeddings AND batch_size
```

The wheel config that makes `ours` int8 + dim-512, `api-py/Cargo.toml:11`:

```toml
ese_core = { package = "ese", path = "..", features = ["rayon", "dim-512", "quant-8"] }
```

The broken repro block, `bench_out/machine-benchmark-report.md:111-116`:

```sh
./bench.sh verify
./bench.sh rust
./bench.sh python --all
./bench.sh nanobeir
```

`find . -name bench.sh` returns nothing. The only shell script under benches is
`benches/py/all_nanobeir.sh`.

The headline claim being re-validated, `bench_out/machine-benchmark-report.md:70-71`:

> - `ours` is about 71x faster in batched throughput than the next-fastest public model in this run.
> - `ours` is about 43x faster in single-query throughput than the best non-`ours` model in this run.

**Known, already-decided tradeoff — do NOT "fix" this:** `ours` runs at batch
size 65536 (the wrapper default) while transformer baselines run at their own
default (32). This is deliberate (each model's preferred batch size; recorded in
`plans/README.md` "considered and rejected"). Your job is to **disclose** it in
the report, not change it.

Repo conventions:
- Python tooling uses `uv` (`benches/py/pyproject.toml`, `uv.lock` present).
- Wheel build: `cd api-py && RUSTFLAGS="-Ctarget-cpu=native" maturin build --release`
  → produces `target/wheels/ese_py-*.whl`.
- Python bench invocation pattern (from `readme.md`):
  `cd benches/py && uv run --with ../../target/wheels/ese_py-*.whl bench.py ours`.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Drift check | `git diff --stat 8a4616c..HEAD -- benches/ api-py/` | empty or known |
| Record SHA | `git rev-parse --short HEAD` | a short SHA (stamp it into the report) |
| Workspace builds | `cargo check --workspace` | exit 0 |
| Build wheel | `cd api-py && RUSTFLAGS="-Ctarget-cpu=native" maturin build --release` | a `.whl` in `target/wheels/` |
| Rust bench | `cargo bench --bench gooaq --features rayon` | criterion output, exit 0 |
| Python bench (ours) | `cd benches/py && uv run --with ../../target/wheels/ese_py-*.whl bench.py ours` | prints QPS, writes JSON |
| Python lint (syntax) | `cd benches/py && uv run python -c "import ast; ast.parse(open('model_loader.py').read())"` | exit 0 |

## Scope

**In scope** (the only files you should modify or create):
- `benches/py/model_loader.py` (fix the dropped flag; document the asymmetry)
- `benches/py/models.txt` (add a Unigram/multilingual entry if not present)
- `bench.sh` (create — the repro entry point) **or** edit the repro block in the report to real commands (see step 5)
- `bench_out/machine-benchmark-report.md` (regenerate)
- `benches/py/README.md` (create or update — document the fair-comparison caveats)

**Out of scope** (do NOT touch):
- Any file under `src/`, `build.rs`, or `api-py/src/` — this plan changes no
  runtime or compiler behavior. (Changing `api-py/Cargo.toml` features is also
  out of scope: the int8/dim-512 wheel is the shipped config; disclose it, don't
  change it.)
- `benches/gooaq.rs` / `benches/bench_encode.rs` Rust harness logic — re-run
  them, don't rewrite them.

## Git workflow

- Work on `advisor/improvements` (all plan work aggregates there; `master`
  stays untouched until the user merges).
- Commit per logical unit; message style is lowercase imperative, e.g.
  `fix bench wrapper: honor normalize_embeddings`,
  `regenerate machine benchmark report on HEAD`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Make the benchmark wrapper honor `normalize_embeddings`

In `benches/py/model_loader.py`, change `OurSentenceTransformerWrapper.encode`
so that when `normalize_embeddings=True` it L2-normalizes the returned array
(the baselines normalize per their own defaults; today `ours` never does, so the
two sides don't compute the same thing). Keep the default `False`.

Target shape:

```python
    def encode(self, sentences, batch_size=65536, show_progress_bar=False,
               normalize_embeddings=False, convert_to_numpy=True, **kwargs):
        vecs = ese.encode(sentences)
        if normalize_embeddings:
            vecs = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs
```

Also add a one-paragraph class docstring stating the two disclosed asymmetries:
`ours` is **int8-quantized (quant-8) at dim-512** (per `api-py/Cargo.toml`),
while baselines run f32 at native dims; and `ours` uses batch size 65536 vs
baselines' 32 (deliberate per-model batch sizing).

**Verify**: `cd benches/py && uv run python -c "import ast; ast.parse(open('model_loader.py').read())"` → exit 0. (Full import needs the wheel; that happens in step 3.)

### Step 2: Re-run the Rust throughput benchmarks on HEAD

Run `cargo bench --bench gooaq --features rayon` and capture the single-input,
105-char, and batch-size table numbers. These replace the "Rust Throughput"
section of the report.

**Verify**: criterion completes, exit 0; you have fresh numbers for short-input
latency and the batch-size throughput table.

**If the bench cannot run in this environment** (no network for the GooAQ
download, or no perf-stable hardware): STOP and report — do not fabricate
numbers. See STOP conditions.

### Step 3: Rebuild the wheel and re-run the Python GooAQ sweep

Build the wheel (`cd api-py && RUSTFLAGS="-Ctarget-cpu=native" maturin build --release`),
then run `bench.py` for `ours` and for each downloadable baseline in
`models.txt`. `google/embeddinggemma-300m` is HF-gated — skip it and note the
skip (as the April report did).

**Verify**: `benches/py/benchmarks/ours_bench.json` and one JSON per baseline
exist and are newly dated; `ours` prints a single-query and batched QPS.

### Step 4: Add a Unigram/multilingual configuration to the matrix

The April report only measured the WordPiece default model as `ours`. The new
Unigram/Viterbi path (the headline generalization feature) is unbenchmarked.
Build a **second** wheel against a Unigram model and measure it separately so
the speed claim is scoped to configurations actually measured.

Use the CI acceptance Unigram model (already pinned + checksummed in
`.github/workflows/ci.yml`): `minishlab/potion-multilingual-128M` at dim-256.
Build it via env vars at wheel-build time, e.g.:

```sh
cd api-py
ESE_MODEL=minishlab/potion-multilingual-128M@73908c3438cf03b6a01bcb9611d62b23d0726f08 \
ESE_MODEL_SHA256=14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397 \
ESE_TOKENIZER_SHA256=19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1 \
ESE_NORMALIZE_OUTPUT=1 \
RUSTFLAGS="-Ctarget-cpu=native" maturin build --release
```

Then run `bench.py ours` against that wheel and record the result under a
distinct label (e.g. `ours-unigram-potion-ml-128M`) in the report.

**Verify**: a JSON result exists for the Unigram wheel; its QPS is recorded in
the report as a separate row, not merged into the WordPiece `ours` number.

**Note for the report**: a Unigram build is algorithmically heavier than
WordPiece (Viterbi vs greedy longest-match). If its QPS is materially lower,
that is expected and must be stated — the "fastest" claim should be scoped to
the configuration(s) where it holds.

### Step 5: Restore the repro entry point

The report references `./bench.sh`. Either:
- **(a)** create `bench.sh` at the repo root implementing the four subcommands
  (`verify`, `rust`, `python [--all]`, `nanobeir`) as thin wrappers over the
  real commands in this plan's "Commands you will need" table and `all_nanobeir.sh`; or
- **(b)** if you prefer not to add a script, edit `bench_out/machine-benchmark-report.md:109-116`
  to list the actual working commands instead of the non-existent `bench.sh`.

Prefer (a) — a single repro entry point is what the report promises.

**Verify**: `bash bench.sh verify` (or the edited commands) runs without "file
not found"; `test -f bench.sh || grep -q 'cargo bench' bench_out/machine-benchmark-report.md`.

### Step 6: Regenerate the report, stamped and disclosed

Rewrite `bench_out/machine-benchmark-report.md` with:
- **Run date** and **git SHA** (`git rev-parse --short HEAD`) it describes —
  both in the Environment section.
- The fresh Rust + Python numbers from steps 2–4.
- A new **"Comparison fairness"** subsection stating explicitly: `ours` is
  quant-8 / dim-512; baselines are f32 at native dims; batch sizes differ
  (65536 vs 32, deliberate); normalization handling.
- The separate Unigram row from step 4.
- Correct repro commands (step 5).

**Verify**: `grep -n "$(git rev-parse --short HEAD)" bench_out/machine-benchmark-report.md`
returns a match; the report contains a "fairness"/"quant-8"/"dim-512" disclosure;
the Unigram row is present.

## Test plan

This plan produces measurements, not unit-testable logic. Verification is:
- `cargo check --workspace` → exit 0 (nothing in `src/` changed, so this must
  remain green).
- `benches/py/model_loader.py` parses (step 1 verify) and imports against the
  built wheel without error (exercised in step 3).
- The regenerated report is internally consistent: every model in the ranking
  table has a corresponding JSON in `benches/py/benchmarks/` dated to this run.

No new Rust tests. If you touch nothing under `src/`, the existing
`cargo test --features tests` suite must still pass unchanged — run it once as a
guard: `cargo test --features tests` → all pass.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `cargo check --workspace` exits 0
- [ ] `benches/py/model_loader.py` honors `normalize_embeddings` (grep shows a `np.linalg.norm` or equivalent in the wrapper `encode`)
- [ ] `bench_out/machine-benchmark-report.md` contains the current HEAD short SHA and a run date later than 2026-06-13
- [ ] The report has an explicit fairness disclosure naming `quant-8` and `dim-512` for `ours`
- [ ] A Unigram/multilingual configuration is measured and reported as a separate row
- [ ] `bench.sh` exists and its subcommands run, OR the report's repro block lists only real, existing commands
- [ ] `git status` shows no files modified outside the in-scope list
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The benchmarks cannot be run in this environment (no network for dataset/model
  downloads, or no performance-stable hardware). **Do not fabricate or estimate
  numbers.** Deliver the harness/code fixes (steps 1, 4-code, 5) and report that
  the measurement run (steps 2, 3, 6) must be executed on the target machine.
- `ours` batched QPS on HEAD is more than ~2× slower than the April figure
  (~4.08M batched QPS on an M4 Pro) on comparable hardware. That would indicate
  a real performance regression from the generalization work and must be
  surfaced *before* the report is published — it changes the merge decision.
- The drift check shows `api-py/Cargo.toml` or the bench harness changed since
  `8a4616c` in a way that conflicts with the excerpts above.
- Honoring `normalize_embeddings` changes `ours`'s NanoBEIR quality numbers in a
  direction that contradicts the README's `[output change]` note — report the
  discrepancy rather than quietly publishing it.

## Maintenance notes

- This report should be regenerated whenever the runtime hot path changes
  (notably after plan 019 SIMD lands — that is the *point* of having a fair
  baseline now). Keep the SHA stamp so future readers know what code a number
  describes.
- When BPE/cased/WordLevel families land later, add them to `models.txt` and the
  matrix the same way step 4 adds the Unigram row.
- A reviewer should scrutinize: that no `src/`/`build.rs` file was touched (this
  is a measurement plan), and that the fairness disclosure is prominent rather
  than buried.
- Deferred out of this plan: a CI "benchmark smoke" job (run a tiny sweep on
  every PR to catch regressions). Worth doing once the numbers are trustworthy;
  out of scope here to keep this plan to measurement + fairness.
