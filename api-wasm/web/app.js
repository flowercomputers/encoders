// app.js — ese wasm demo: semantic similarity in the browser, no server
// Loads the wasm module (weights baked in), encodes two text inputs in real-time.

import init, { encode_single, dimensions, model_name } from '../pkg-web/ese_wasm.js';

const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const scoreEl  = document.getElementById('scoreText');
const barEl    = document.getElementById('bar');
const metaEl   = document.getElementById('meta');
const textA    = document.getElementById('textA');
const textB    = document.getElementById('textB');

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function update() {
  const tA = textA.value.trim();
  const tB = textB.value.trim();
  if (!tA || !tB) { resultEl.hidden = true; return; }

  const t0 = performance.now();
  const vA = encode_single(tA);
  const vB = encode_single(tB);
  const t1 = performance.now();

  const sim = cosine(vA, vB);
  const pct = Math.max(0, Math.min(100, ((sim + 1) / 2) * 100));

  scoreEl.textContent = sim.toFixed(4);
  barEl.style.width   = pct + '%';
  barEl.style.background = sim > 0.7 ? '#2a7a2a' : sim > 0.3 ? '#b37d00' : '#666';
  metaEl.textContent  = `dim=${dimensions()}  model=${model_name()}  encode time: ${(t1 - t0).toFixed(1)} ms`;
  resultEl.hidden = false;
}

// Debounce so we don't encode on every keystroke
let debounce;
function onInput() {
  clearTimeout(debounce);
  debounce = setTimeout(update, 80);
}

// Load the wasm module — weights are embedded in ese_wasm_bg.wasm (no fetch needed for inline)
try {
  await init();
  statusEl.textContent = `Ready — ${model_name()}, dim=${dimensions()}`;
  statusEl.className   = 'ready';
  textA.addEventListener('input', onInput);
  textB.addEventListener('input', onInput);
  update(); // compute initial similarity for the placeholder text
} catch (err) {
  statusEl.textContent = 'Failed to load wasm: ' + err.message;
  statusEl.className   = 'error';
  console.error(err);
}
