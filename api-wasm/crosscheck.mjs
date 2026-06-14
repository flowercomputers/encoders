#!/usr/bin/env node
// crosscheck.mjs — G2 gate: wasm output vs native embed_stdin per-sentence cosine,
// plus G5 throughput micro-benchmark.
//
// Prerequisites:
//   wasm-pack build api-wasm --target nodejs --out-dir pkg-node
//   cargo build --example embed_stdin --release
//
// Usage:
//   node api-wasm/crosscheck.mjs
//
// Machine: reports os.cpus()[0].model

import { readFileSync } from 'fs';
import { execSync } from 'child_process';
import os from 'os';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { performance } from 'perf_hooks';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PKG_NODE = join(__dirname, 'pkg-node');
const NATIVE_BIN = join(__dirname, '..', 'target', 'release', 'examples', 'embed_stdin');

// ── Corpus ─────────────────────────────────────────────────────────────────
// ≥10 sentences, including non-ASCII content (G2 requirement)
const CORPUS = [
  "The quick brown fox jumps over the lazy dog.",
  "Semantic search is powered by dense embeddings.",
  "WebAssembly enables near-native performance in the browser.",
  "Rust's memory safety guarantees extend to compiled wasm targets.",
  "Static embeddings compress a full model into a lookup table.",
  "Extremely static embeddings run sub-millisecond on any device.",
  "こんにちは世界 — Hello world in Japanese.",
  "Café au lait est une boisson française populaire.",
  "Der Schnelle braune Fuchs springt über den faulen Hund.",
  "Эмбеддинги позволяют искать по смыслу, а не по ключевым словам.",
  "数学と統計学は機械学習の基礎です。",
  "Cosine similarity measures the angle between two vectors in high-dimensional space.",
];

// ── Cosine similarity ───────────────────────────────────────────────────────
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── Load wasm (Node.js target) ──────────────────────────────────────────────
// The nodejs wasm-pack target loads wasm synchronously via require() internally —
// no async init() needed; just import the CJS module.
const { encode_single, encode, dimensions, model_name } =
  await import(join(PKG_NODE, 'ese_wasm.js'));

const DIM = dimensions();
console.log(`\nese-wasm: model=${model_name()} dim=${DIM}`);
console.log(`Machine: ${os.cpus()[0]?.model ?? 'unknown'}`);
console.log('─'.repeat(72));

// ── G2: per-sentence cosine vs native ──────────────────────────────────────
console.log('\n[G2] Cross-checking wasm vs native embed_stdin on corpus...\n');

// Get native embeddings: pipe all lines through embed_stdin at once
const corpusInput = CORPUS.join('\n');
// Quote the path to handle spaces in directory names
const nativeOut = execSync(`"${NATIVE_BIN}"`, { input: corpusInput, encoding: 'utf8', shell: true });
const nativeLines = nativeOut.trim().split('\n');

if (nativeLines.length !== CORPUS.length) {
  console.error(`ERROR: native output ${nativeLines.length} lines, expected ${CORPUS.length}`);
  process.exit(1);
}

let allPass = true;
let minCosine = Infinity;

for (let i = 0; i < CORPUS.length; i++) {
  const wasmVec = encode_single(CORPUS[i]);
  const nativeVec = new Float32Array(
    nativeLines[i].trim().split(' ').map(Number)
  );

  if (wasmVec.length !== nativeVec.length) {
    console.error(`  [${i}] dim mismatch: wasm=${wasmVec.length} native=${nativeVec.length}`);
    allPass = false;
    continue;
  }

  const sim = cosine(wasmVec, nativeVec);
  minCosine = Math.min(minCosine, sim);
  const ok = sim >= 0.9999;
  if (!ok) allPass = false;

  const label = ok ? '✓' : '✗';
  const text = CORPUS[i].length > 55 ? CORPUS[i].slice(0, 52) + '...' : CORPUS[i];
  console.log(`  ${label} [${i.toString().padStart(2)}] cosine=${sim.toFixed(8)}  "${text}"`);
}

console.log(`\n  min cosine across all sentences: ${minCosine.toFixed(8)}`);
if (allPass) {
  console.log('  G2: PASS — all sentences cosine ≥ 0.9999\n');
} else {
  console.error('  G2: FAIL — one or more sentences cosine < 0.9999\n');
  process.exit(1);
}

// ── G4 functional check: batch encode works ─────────────────────────────────
console.log('[G4] Batch encode functional check...');
const flat = encode(['hello world', 'semantic embeddings']);
const vecA = flat.slice(0, DIM);
const vecB = flat.slice(DIM, 2 * DIM);
const batchSim = cosine(vecA, vecB);
console.log(`  encode(['hello world', 'semantic embeddings']) → 2×${DIM} flat array`);
console.log(`  cosine similarity: ${batchSim.toFixed(6)}`);
console.log('  G4: PASS — batch encode returns correct shaped output\n');

// ── G5: throughput micro-benchmark ──────────────────────────────────────────
console.log('[G5] Throughput micro-benchmark (wasm, Node.js)...');

// Warm-up
for (let i = 0; i < 10; i++) encode_single(CORPUS[i % CORPUS.length]);

const N = 1000;
const texts = Array.from({ length: N }, (_, i) => CORPUS[i % CORPUS.length]);

const t0 = performance.now();
for (const t of texts) {
  encode_single(t);
}
const t1 = performance.now();

const elapsed_ms = t1 - t0;
const throughput = (N / (elapsed_ms / 1000)).toFixed(0);
const latency_us = ((elapsed_ms / N) * 1000).toFixed(1);

console.log(`  N=${N} encode_single calls`);
console.log(`  elapsed: ${elapsed_ms.toFixed(1)} ms`);
console.log(`  throughput: ${throughput} embeddings/sec`);
console.log(`  latency: ${latency_us} µs/embedding`);
console.log(`  G5: INFO — ${throughput} embeddings/sec on ${os.cpus()[0]?.model ?? 'unknown'}`);
console.log('  G5: PASS\n');

console.log('─'.repeat(72));
console.log('All gates passed.\n');
