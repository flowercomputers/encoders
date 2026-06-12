# Plan 001: Reconcile the five outstanding codex/* branches into master

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 5155c13..HEAD -- readme.md build.rs Cargo.toml license benches/py/models.txt`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S-M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tech-debt
- **Planned at**: commit `5155c13`, 2026-06-12

## Why this matters

Five unmerged remote branches contain useful, mostly-small fixes (license
metadata, a build-script portability fix, a full README rewrite, runtime-use
docs, and an experimental second-model feature). They overlap each other —
two branches edit the same README, and the model-support branch is built on
top of the README branch — so the longer they sit, the harder they merge.
All later plans in `plans/` edit `build.rs` and `readme.md`; landing or
explicitly rejecting these branches first means later work starts from a
stable master instead of colliding with five in-flight diffs.

## Current state

Master is at `5155c13`. The remote branches, each one commit ahead of some
base, are:

| Branch | Commit | What it does |
|---|---|---|
| `origin/codex/license-notice` | `fd19757` | Renames `license` → `LICENSE`, adds a `NOTICE` file (Apache-2.0 attribution), normalizes license metadata. |
| `origin/codex/ese-cache-dir-portability` | `0fa8a8d` | Rewrites `shared_model_dir()` in `build.rs` so the model cache no longer panics when `OUT_DIR` has no `target/` ancestor (custom `CARGO_TARGET_DIR`). |
| `origin/codex/readme-rewrite` | `8fd7827` | Replaces `readme.md` (21 lines) with a ~290-line onboarding doc: quickstart, feature table, semantic-search example, "what this is / is not". |
| `origin/codex/agent-tier-embedding-docs` | `319c511` | Adds an "Embedding as a local primitive" section (runtime/agent use case + rerank example) to the OLD short readme. **Conflicts with readme-rewrite.** |
| `origin/codex/nife-mxbai-support` | `c2f03fb` | Adds a second model (NIFE-mxbai) behind a `model-nife-mxbai` Cargo feature via a `ModelProfile` struct in `build.rs`. **Built on top of `8fd7827` (readme-rewrite).** |

Relevant master files:

- `build.rs:397-406` — `shared_model_dir()` walks `OUT_DIR` ancestors looking
  for a directory literally named `target` and panics otherwise:

  ```rust
  fn shared_model_dir(out: &Path) -> PathBuf {
      let target_dir = out
          .ancestors()
          .find(|p| p.file_name().map_or(false, |f| f == "target"))
          .unwrap_or_else(|| panic!("could not find target/ in OUT_DIR"))
          .to_path_buf();
      let dir = target_dir.join(concat!(env!("CARGO_PKG_NAME"), "-cache"));
  ```

- `Cargo.toml:7` — `license-file = "license"` (lowercase path; the
  license-notice branch renames the file, so this line must end up pointing
  at the new name or be replaced by `license = "Apache-2.0"`).
- `benches/py/models.txt` — currently lists `google/embeddinggemma-300m`
  twice (lines 6 and 8 area). Step 6 fixes this independently of the branches.

Repo conventions: commit messages are short, lowercase, imperative (see
`git log --oneline`: "improve benchmark harness", "add pyproject.toml").
There is no CI yet; verification is local `cargo test`.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Fetch branches | `git fetch origin` | exit 0 |
| Build + unit tests | `cargo test` | 1 unit test + 3 doctests pass |
| Parity tests | `cargo test --features tests` | 2 tests + 3 doctests pass |
| Package sanity | `cargo package --list --allow-dirty \| head -30` | file list, exit 0 |

Note: the first `cargo build` downloads ~100 MB of model weights from
Hugging Face into the model cache (currently `target/ese-cache/`). Network
is required once; afterwards builds are offline.

## Scope

**In scope** (the only files you should modify):
- `readme.md`, `LICENSE` / `license`, `NOTICE`, `Cargo.toml` (license
  metadata line only), `build.rs` (only the `shared_model_dir` hunk from the
  portability branch), `benches/py/models.txt`
- Merge commits for the branches listed below

**Out of scope** (do NOT touch):
- `origin/codex/nife-mxbai-support` — do **not** merge it. It is superseded
  by `plans/006-manifest-model-ingestion.md`, which absorbs its ideas
  (ModelProfile, special-token lookup by name, per-model cache key) without
  the per-model Cargo feature. Leave the branch in place as reference.
- Any other change to `build.rs` (pinning, checksums, special tokens — those
  belong to plans 003/004/006).
- `src/` — no source changes in this plan.

## Git workflow

- Work directly toward `master` merges on a branch named
  `advisor/001-branch-reconciliation`; the operator merges/pushes.
- Use real merges (`git merge --no-ff origin/<branch>`) so branch history is
  preserved; resolve conflicts in the merge commit.
- Commit messages: short lowercase imperative, e.g. `merge license notice
  branch`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Merge `codex/license-notice`

`git merge --no-ff origin/codex/license-notice`. After merging, verify the
`Cargo.toml` license metadata is consistent with the renamed file: either
`license-file = "LICENSE"` or `license = "Apache-2.0"` (the branch's choice
wins; just confirm it builds and packages).

**Verify**: `cargo package --list --allow-dirty | grep -i -E "license|notice"` →
lists `LICENSE` (and `NOTICE` if the branch added it to includes); `cargo test` → all pass.

### Step 2: Merge `codex/ese-cache-dir-portability`

`git merge --no-ff origin/codex/ese-cache-dir-portability`. This touches only
`shared_model_dir()` in `build.rs`. Read the merged function and note the new
cache location logic — record it in your report (plans 002/003 need to know
where the cache lives).

**Verify**: `cargo test` → all pass (build script reruns successfully).
Additionally: `CARGO_TARGET_DIR=/tmp/ese-alt-target cargo check` → exit 0
(this is the failure mode the branch fixes; on master it panics).

### Step 3: Merge `codex/readme-rewrite`

`git merge --no-ff origin/codex/readme-rewrite`. The branch replaces
`readme.md` wholesale; master's readme has drifted (typo fix `5155c13`,
python-wheel paragraph). Resolve in favor of the branch's structure, then
re-check that these master-only facts survive in the merged readme:
- the link to `./api-py/readme.md` for the Python wheel,
- the crate-features paragraph pointing at `Cargo.toml`.

**Verify**: `grep -c "api-py" readme.md` → at least 1;
`grep -i "quickstart" readme.md` → at least 1 match (rewrite content present).

### Step 4: Port `codex/agent-tier-embedding-docs` content into the new readme

Do NOT git-merge this branch (it would conflict pointlessly with the rewrite).
Instead cherry-pick its content by hand: `git show origin/codex/agent-tier-embedding-docs:readme.md`
and copy the "Embedding as a local primitive" section (the runtime/agent
use-case framing plus the email-reranking `cosine` example) into the merged
readme as its own section, after the quickstart examples. If the rewritten
readme already contains a semantic-search example, keep both but de-duplicate
any repeated `cosine` helper explanation — one code listing per idea.

Commit as `port runtime embedding docs into rewritten readme`. Then tell the
operator the branch can be deleted (do not delete it yourself).

**Verify**: `grep -c "local primitive" readme.md` → ≥ 1; readme renders
without broken fences: `grep -c '```' readme.md` → an even number.

### Step 5: Record the NIFE branch disposition

Append a line to `plans/README.md` under "Dependency notes":
`codex/nife-mxbai-support intentionally NOT merged — superseded by plan 006; kept as reference.`

**Verify**: `grep "nife" plans/README.md` → 1 match.

### Step 6: Fix the duplicate model line in `benches/py/models.txt`

The file lists `google/embeddinggemma-300m` twice. Remove one occurrence so
each model appears exactly once. (If the working tree already contains this
fix uncommitted, commit it as-is.)

**Verify**: `sort benches/py/models.txt | uniq -d` → empty output.

## Test plan

No new tests. The gate after every merge is the existing suite:
`cargo test` and, once after all merges, `cargo test --features tests`
(2 tests + 3 doctests, all pass).

## Done criteria

- [ ] `git log --oneline --merges -5` shows merges of license-notice,
      cache-dir-portability, and readme-rewrite
- [ ] `cargo test --features tests` exits 0
- [ ] `CARGO_TARGET_DIR=/tmp/ese-alt-target cargo check` exits 0
- [ ] `sort benches/py/models.txt | uniq -d` prints nothing
- [ ] `grep "local primitive" readme.md` matches
- [ ] `codex/nife-mxbai-support` is NOT in the merge history
      (`git log --oneline --merges | grep -ci nife` → 0)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- A merge produces conflicts outside `readme.md`, `Cargo.toml` metadata,
  `LICENSE`/`license`, or `build.rs`'s `shared_model_dir` — that means the
  branches contain more than this plan describes.
- `cargo test` fails after any individual merge (report which merge).
- The license-notice branch changes the license *terms* rather than just the
  file name/metadata — licensing decisions belong to the operator.
- `CARGO_TARGET_DIR=/tmp/ese-alt-target cargo check` still panics after
  Step 2.

## Maintenance notes

- Plans 002–006 assume post-merge master: in particular the model-cache
  directory may no longer be `target/ese-cache` after Step 2 — they instruct
  their executors to read `shared_model_dir()` rather than assume the path.
- Reviewer should scrutinize the merged readme for stale claims (it was
  written against pre-`5155c13` master).
- Deferred deliberately: merging nife-mxbai (superseded by plan 006).
