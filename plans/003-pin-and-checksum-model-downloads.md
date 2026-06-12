# Plan 003: Pin model downloads to a revision and verify checksums before baking weights into binaries

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 5155c13..HEAD -- build.rs`
> Changes from plan 001 (the `shared_model_dir` hunk) are expected. If the
> `MODEL_URL`/`TOKENIZER_URL` constants or `download_if_missing` changed,
> compare against the "Current state" excerpts; on a mismatch, STOP.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/001-branch-reconciliation.md (build.rs merge), plans/002-ci-and-quant-aware-parity.md (regression gate)
- **Category**: security
- **Planned at**: commit `5155c13`, 2026-06-12

## Why this matters

The build script downloads model weights and tokenizer from
`huggingface.co/.../resolve/main/...` — an unpinned, mutable ref — with no
integrity check, and then `include_bytes!`-bakes the result into every
consumer's binary. If the upstream repo is re-uploaded, force-pushed, or
compromised, every fresh build silently embeds different weights (different
search results at best, attacker-chosen bytes in shipped binaries at worst),
and no two builds are guaranteed reproducible. Pinning to an immutable
revision and verifying SHA-256 makes the build reproducible and tamper-evident.
The repo already has precedent: the unmerged `codex/nife-mxbai-support`
branch pins its model URL to revision `970a0151f3faeddbab5c68be38d52bbc5ea719ab`.

## Current state

- `build.rs:8-9` (line numbers from `5155c13`; plan 001 may shift them):

  ```rust
  static MODEL_URL: &'static str = "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/main/0_StaticEmbedding/model.safetensors";
  static TOKENIZER_URL: &'static str = "https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/main/0_StaticEmbedding/tokenizer.json";
  ```

- `build.rs:347-355` — the downloader, no integrity check:

  ```rust
  fn download_if_missing(path: &Path, url: &str) {
      if path.exists() {
          return;
      }
      eprintln!("cargo:warning=Downloading {}...", path.display());
      let resp = minreq::get(url).send().expect("download failed");
      assert!(resp.status_code == 200, "HTTP {}", resp.status_code);
      fs::write(path, resp.as_bytes()).unwrap();
  }
  ```

- HTTP client: `minreq` with `https-native` (TLS verified — transport is
  fine; the gap is content pinning).
- The model cache directory is returned by `shared_model_dir()` in
  `build.rs` — read it for the current path before testing.
- No new dependencies should be added for hashing: build-deps are
  deliberately minimal (`serde_json`, `minreq`, optional `tokenizers`).
  A self-contained SHA-256 implementation is NOT wanted either — see Step 2
  for the approved approach.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Resolve current revision | `curl -s https://huggingface.co/api/models/sentence-transformers/static-retrieval-mrl-en-v1 \| python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"` | a 40-char hex sha |
| Hash a file (macOS/Linux) | `shasum -a 256 <file>` | hex digest |
| Build (cold) | `cargo build` | exit 0, download warnings |
| Parity gate | `cargo test --features tests` | all pass |

## Scope

**In scope** (the only files you should modify):
- `build.rs`
- `Cargo.toml` / `Cargo.lock` (ONLY if Step 2's chosen hashing route
  requires adding the `sha2` build-dependency — see the decision rule there)
- `plans/README.md` (status row)

**Out of scope** (do NOT touch):
- `src/` — runtime is unaffected.
- Multi-model/env-var configuration — that is plan 006; this plan pins the
  single existing model only.
- The benches' dataset downloads (`benches/gooaq.rs`) — bench data integrity
  is not load-bearing for shipped binaries.

## Git workflow

- Branch: `advisor/003-pin-model-downloads`
- Commit style: short lowercase imperative, e.g. `pin model download to
  revision and verify sha256`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Resolve the revision and expected hashes

1. Resolve the current `main` commit sha of the upstream model repo (command
   table above); call it `REV`.
2. Download both files at that pinned revision to a temp dir:
   - `https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1/resolve/<REV>/0_StaticEmbedding/model.safetensors`
   - `.../resolve/<REV>/0_StaticEmbedding/tokenizer.json`
3. Compute `shasum -a 256` for both. Also compare the model file's hash with
   the one already in your local model cache (from `shared_model_dir()`) —
   they should match; if not, note it in the report (upstream drifted from
   what local builds have been using).

**Verify**: you have `REV` plus two hex digests recorded.

### Step 2: Pin URLs and verify integrity in `build.rs`

1. Replace `resolve/main/` with `resolve/<REV>/` in both URL constants, and
   add the expected digests as constants next to them, e.g.:

   ```rust
   // Pinned to upstream revision <REV> (resolved 2026-06-12).
   // To bump: update REV in both URLs, re-download, recompute the SHA-256s.
   static MODEL_SHA256: &str = "<digest>";
   static TOKENIZER_SHA256: &str = "<digest>";
   ```

2. Extend `download_if_missing(path, url, expected_sha256)` to hash the
   bytes after download and before `fs::write`, panicking with a clear
   message on mismatch (include expected vs actual and the URL). Also verify
   when the file already exists in cache (`path.exists()` branch): hash the
   existing file once and panic on mismatch — this catches a poisoned or
   truncated cache. (The cost — hashing ~100 MB per build-script rerun — is
   acceptable because the build script only reruns when its inputs change;
   cargo does not rerun it on every `cargo build`.)
3. Hashing implementation — decision rule, in order of preference:
   a. Add `sha2 = "0.10"` to `[build-dependencies]`. This is the standard,
      audited choice and is the approach to take by default.
   b. Only if the operator has said "no new dependencies": shell out is NOT
      acceptable (portability); in that case STOP and report instead.

**Verify**: `rm -rf <model-cache-dir> && cargo build` → exit 0 (downloads,
verifies, builds). Then corrupt the cached tokenizer
(`printf 'x' >> <cache>/tokenizer.json`) and run `cargo build` → build FAILS
with the checksum message. Restore by deleting the corrupted file and
rebuilding.

### Step 3: Confirm encoding is byte-identical

The pinned revision should serve the same content `main` did.

**Verify**: `cargo test --features tests` → all pass (parity against
reference outputs regenerated from the pinned files).

## Test plan

No new test functions — the integrity check lives in the build script and is
exercised by the corrupt-cache check in Step 2 (record its output in your
report). Regression safety comes from the existing parity suite plus plan
002's CI matrix.

## Done criteria

- [ ] `grep -c "resolve/main" build.rs` → 0
- [ ] `grep -c "SHA256" build.rs` → ≥ 2 (both expected digests present)
- [ ] Cold build succeeds: `rm -rf <model-cache-dir> && cargo build` → exit 0
- [ ] Corrupted cache file fails the build with a checksum error (manually
      verified in Step 2, noted in report)
- [ ] `cargo test --features tests` exits 0
- [ ] No files outside the in-scope list modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The hash of the freshly pinned download differs from the file already in
  the local cache AND `cargo test --features tests` fails afterwards —
  upstream changed the weights; the operator must decide which revision to
  pin.
- The HF API call cannot resolve a revision sha (offline/auth issues).
- Adding `sha2` to build-dependencies causes any dependency-resolution
  conflict.
- `download_if_missing` in the live code no longer matches the excerpt
  (someone else restructured downloads first).

## Maintenance notes

- Bumping the model now requires updating REV + two digests in `build.rs` —
  the comment from Step 2 documents the procedure. Plan 006 will move
  pin+digest into a per-model manifest; keep the verification function
  generic (`(path, url, expected_sha256)`) so it can be reused there.
- Plan 002's CI model-cache key should be updated to include `REV` for exact
  invalidation (one-line follow-up in `.github/workflows/ci.yml`).
- Reviewer: confirm the digests in the diff match an independent download
  (don't take the PR's word for it — recompute one of them).
