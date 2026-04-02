import init, { dimensions, encode, encode_repeated } from "./pkg/ese_wasm.js";

const benchmarkInput = document.getElementById("benchmark-input");
const iterationsInput = document.getElementById("iterations");
const benchmarkStatus = document.getElementById("benchmark-status");
const corpusInput = document.getElementById("corpus-input");
const corpusStatus = document.getElementById("corpus-status");
const queryInput = document.getElementById("query-input");
const searchStatus = document.getElementById("search-status");
const selectedStatus = document.getElementById("selected-status");
const neighborStatus = document.getElementById("neighbor-status");
const similarityStatus = document.getElementById("similarity-status");
const mapStatus = document.getElementById("map-status");
const mapHover = document.getElementById("map-hover");

const metricTotal = document.getElementById("metric-total");
const metricPer = document.getElementById("metric-per");
const metricRate = document.getElementById("metric-rate");
const metricChecksum = document.getElementById("metric-checksum");
const heroDims = document.getElementById("hero-dims");
const heroCorpus = document.getElementById("hero-corpus");

const embeddingCanvas = document.getElementById("embedding-canvas");
const mapCanvas = document.getElementById("map-canvas");
const searchResults = document.getElementById("search-results");
const neighborResults = document.getElementById("neighbor-results");
const similarityOutput = document.getElementById("similarity-output");

const fmt = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

const SAMPLE_CORPUS = `fresh apple slices with cinnamon
pear and citrus salad for lunch
banana smoothie with oat milk
semantic search for engineering notes
vector retrieval for product documentation
browser wasm benchmark for local embeddings
customer asks for invoice refund
billing issue on annual subscription renewal
bug report: upload button freezes on mobile
feature request for collaborative editing
docs page about rust toolchains
web assembly tutorial for search ranking
nearest neighbor lookup for help center articles
support ticket about password reset email
team lunch planning for Friday
benchmark results for a static retrieval model
help center article about changing email address
shipping status question for a delayed order
guide to compiling Rust to WebAssembly
roadmap note about offline browser search
internal note on semantic clustering
bug report: checkout page loops after submit`;

const appState = {
  corpus: [],
  embeddings: [],
  selectedIndex: null,
  queryEmbedding: null,
  searchResults: [],
  anchors: null,
  projectionBounds: null,
  mapPoints: [],
  hoverIndex: null,
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function shorten(text, maxLength = 72) {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength - 1)}...`;
}

function embeddingFromText(text) {
  return Array.from(encode(text));
}

function cosineSimilarity(left, right) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (let i = 0; i < left.length; i += 1) {
    dot += left[i] * right[i];
    leftNorm += left[i] * left[i];
    rightNorm += right[i] * right[i];
  }

  if (leftNorm === 0 || rightNorm === 0) {
    return 0;
  }

  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

function parseCorpusText(rawText) {
  return rawText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function renderEmbedding(embedding) {
  const context = embeddingCanvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = embeddingCanvas.clientWidth * ratio;
  const height = embeddingCanvas.clientHeight * ratio;
  embeddingCanvas.width = width;
  embeddingCanvas.height = height;
  context.clearRect(0, 0, width, height);

  const min = Math.min(...embedding);
  const max = Math.max(...embedding);
  const amplitude = Math.max(Math.abs(min), Math.abs(max), 1e-6);
  const midY = height / 2;

  context.fillStyle = "#fafafa";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "rgba(28, 22, 15, 0.08)";
  context.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = (height / 4) * i;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  const gradient = context.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#0f4c81");
  gradient.addColorStop(0.45, "#0b6e63");
  gradient.addColorStop(1, "#c46a2e");
  context.strokeStyle = gradient;
  context.lineWidth = Math.max(2, ratio * 1.6);
  context.beginPath();

  embedding.forEach((value, index) => {
    const x = (index / (embedding.length - 1)) * width;
    const y = midY - (value / amplitude) * (height * 0.38);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });

  context.stroke();

  context.strokeStyle = "rgba(28, 22, 15, 0.22)";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, midY);
  context.lineTo(width, midY);
  context.stroke();
}

function renderSimilarityMatrix(texts, embeddings) {
  const size = texts.length;
  const matrix = Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, col) =>
      cosineSimilarity(embeddings[row], embeddings[col]),
    ),
  );

  similarityOutput.replaceChildren();

  const grid = document.createElement("div");
  grid.className = "matrix-grid";
  grid.style.gridTemplateColumns = `180px repeat(${size}, minmax(56px, 1fr))`;

  const corner = document.createElement("div");
  corner.className = "matrix-label";
  corner.textContent = "text";
  grid.appendChild(corner);

  texts.forEach((text) => {
    const label = document.createElement("div");
    label.className = "matrix-label";
    label.textContent = shorten(text, 24);
    label.title = text;
    grid.appendChild(label);
  });

  matrix.forEach((row, rowIndex) => {
    const label = document.createElement("div");
    label.className = "matrix-label";
    label.textContent = shorten(texts[rowIndex], 24);
    label.title = texts[rowIndex];
    grid.appendChild(label);

    row.forEach((value) => {
      const cell = document.createElement("div");
      const normalized = clamp((value + 1) / 2, 0, 1);
      const hue = 210 - normalized * 170;
      const lightness = 92 - normalized * 42;
      cell.className = "matrix-cell";
      cell.style.background = `hsl(${hue} 58% ${lightness}%)`;
      cell.style.color = normalized > 0.62 ? "#fffaf5" : "#1c160f";
      cell.textContent = value.toFixed(2);
      cell.title = `${texts[rowIndex]} similarity ${value.toFixed(3)}`;
      grid.appendChild(cell);
    });
  });

  similarityOutput.appendChild(grid);
}

function updateEmbeddingView(statusText) {
  const text = benchmarkInput.value.trim();
  if (!text) {
    benchmarkStatus.textContent = "Enter text to inspect an embedding.";
    return;
  }

  const embedding = embeddingFromText(text);
  renderEmbedding(embedding);
  benchmarkStatus.textContent =
    statusText ||
    `Rendered ${embedding.length} dimensions from the current browser-side embedding.`;
}

function runBenchmark() {
  const text = benchmarkInput.value.trim();
  const iterations = Number.parseInt(iterationsInput.value, 10);

  if (!text) {
    benchmarkStatus.textContent = "Enter text before running the benchmark.";
    return;
  }

  if (!Number.isFinite(iterations) || iterations < 1) {
    benchmarkStatus.textContent = "Iterations must be a positive whole number.";
    return;
  }

  benchmarkStatus.textContent = "Running the encoder inside WASM...";

  const start = performance.now();
  const checksum = encode_repeated(text, iterations);
  const totalMs = performance.now() - start;
  const perEncodeMs = totalMs / iterations;
  const encodesPerSecond = 1000 / perEncodeMs;

  metricTotal.textContent = `${fmt.format(totalMs)} ms`;
  metricPer.textContent = `${fmt.format(perEncodeMs)} ms`;
  metricRate.textContent = `${fmt.format(encodesPerSecond)}`;
  metricChecksum.textContent = checksum.toFixed(4);

  updateEmbeddingView(`Completed ${fmt.format(iterations)} browser-side encodes.`);
}

function getNearestNeighbors(index, limit = 5) {
  const target = appState.embeddings[index];

  return appState.embeddings
    .map((embedding, candidateIndex) => ({
      index: candidateIndex,
      score: cosineSimilarity(target, embedding),
    }))
    .filter((entry) => entry.index !== index)
    .sort((left, right) => right.score - left.score)
    .slice(0, limit);
}

function renderSearchResults() {
  searchResults.replaceChildren();

  if (!appState.searchResults.length) {
    return;
  }

  const topResults = appState.searchResults.slice(0, 8);

  topResults.forEach((result, position) => {
    const listItem = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent =
      `${position + 1}. ${shorten(appState.corpus[result.index], 88)} (${result.score.toFixed(3)})`;
    button.addEventListener("click", () => {
      setSelectedIndex(result.index);
    });
    listItem.appendChild(button);
    searchResults.appendChild(listItem);
  });
}

function renderSelectedItem() {
  if (appState.selectedIndex == null) {
    selectedStatus.textContent = "Select a result or map point to inspect it.";
    return;
  }

  const text = appState.corpus[appState.selectedIndex];
  selectedStatus.textContent = `Selected item: ${text}`;
}

function renderNeighbors() {
  neighborResults.replaceChildren();

  if (appState.selectedIndex == null) {
    neighborStatus.textContent = "No selected item yet.";
    return;
  }

  const neighbors = getNearestNeighbors(appState.selectedIndex, 6);
  neighborStatus.textContent = `Nearest neighbors for the selected item.`;

  neighbors.forEach((neighbor, position) => {
    const listItem = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent =
      `${position + 1}. ${shorten(appState.corpus[neighbor.index], 88)} (${neighbor.score.toFixed(3)})`;
    button.addEventListener("click", () => {
      setSelectedIndex(neighbor.index);
    });
    listItem.appendChild(button);
    neighborResults.appendChild(listItem);
  });
}

function findFarthestPair(embeddings) {
  let best = [0, Math.min(1, embeddings.length - 1)];
  let bestDistance = -Infinity;

  for (let left = 0; left < embeddings.length; left += 1) {
    for (let right = left + 1; right < embeddings.length; right += 1) {
      const similarity = cosineSimilarity(embeddings[left], embeddings[right]);
      const distance = 1 - similarity;
      if (distance > bestDistance) {
        bestDistance = distance;
        best = [left, right];
      }
    }
  }

  return best;
}

function computeAnchors(embeddings) {
  if (embeddings.length < 2) {
    return [0, 0, 0, 0];
  }

  const [a, b] = findFarthestPair(embeddings);
  let c = a;
  let worstAverage = Infinity;

  embeddings.forEach((embedding, index) => {
    const average =
      (cosineSimilarity(embedding, embeddings[a]) +
        cosineSimilarity(embedding, embeddings[b])) /
      2;

    if (average < worstAverage) {
      worstAverage = average;
      c = index;
    }
  });

  let d = b;
  let farthestFromC = -Infinity;
  embeddings.forEach((embedding, index) => {
    const distance = 1 - cosineSimilarity(embedding, embeddings[c]);
    if (distance > farthestFromC) {
      farthestFromC = distance;
      d = index;
    }
  });

  return [a, b, c, d];
}

function projectEmbedding(embedding) {
  const [a, b, c, d] = appState.anchors;
  return {
    x:
      cosineSimilarity(embedding, appState.embeddings[a]) -
      cosineSimilarity(embedding, appState.embeddings[b]),
    y:
      cosineSimilarity(embedding, appState.embeddings[c]) -
      cosineSimilarity(embedding, appState.embeddings[d]),
  };
}

function computeMapPoints() {
  if (!appState.embeddings.length) {
    appState.mapPoints = [];
    appState.projectionBounds = null;
    return;
  }

  appState.anchors = computeAnchors(appState.embeddings);
  const rawPoints = appState.embeddings.map((embedding) => projectEmbedding(embedding));

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  rawPoints.forEach((point) => {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  });

  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  appState.projectionBounds = { minX, minY, spanX, spanY };

  appState.mapPoints = rawPoints.map((point, index) => ({
    index,
    x: (point.x - minX) / spanX,
    y: (point.y - minY) / spanY,
  }));
}

function nearestPointFromMouse(event) {
  const rect = mapCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;

  let bestIndex = null;
  let bestDistance = Infinity;

  appState.mapPoints.forEach((point) => {
    const dx = point.x - x;
    const dy = point.y - y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = point.index;
    }
  });

  if (bestDistance <= 0.05) {
    return bestIndex;
  }

  return null;
}

function drawMap() {
  const context = mapCanvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = mapCanvas.clientWidth * ratio;
  const height = mapCanvas.clientHeight * ratio;
  mapCanvas.width = width;
  mapCanvas.height = height;
  context.clearRect(0, 0, width, height);

  context.fillStyle = "#fafafa";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "rgba(0, 0, 0, 0.12)";
  context.beginPath();
  context.moveTo(width * 0.1, height / 2);
  context.lineTo(width * 0.9, height / 2);
  context.moveTo(width / 2, height * 0.1);
  context.lineTo(width / 2, height * 0.9);
  context.stroke();

  appState.mapPoints.forEach((point) => {
    const x = width * (0.1 + point.x * 0.8);
    const y = height * (0.1 + (1 - point.y) * 0.8);
    const isSelected = point.index === appState.selectedIndex;
    const isHovered = point.index === appState.hoverIndex;

    context.beginPath();
    context.arc(x, y, isSelected ? 7 * ratio : 5 * ratio, 0, Math.PI * 2);
    context.fillStyle = isSelected ? "#c46a2e" : "#0f4c81";
    context.fill();

    if (isHovered || isSelected) {
      context.strokeStyle = "#111";
      context.lineWidth = 1.5 * ratio;
      context.stroke();
      context.fillStyle = "#111";
      context.font = `${12 * ratio}px Times New Roman`;
      context.fillText(shorten(appState.corpus[point.index], 32), x + 8 * ratio, y - 8 * ratio);
    }
  });

  if (appState.queryEmbedding) {
    const point = projectEmbedding(appState.queryEmbedding);
    const bounds = appState.projectionBounds;
    const normalizedX = clamp((point.x - bounds.minX) / bounds.spanX, 0, 1);
    const normalizedY = clamp((point.y - bounds.minY) / bounds.spanY, 0, 1);
    const x = width * (0.1 + normalizedX * 0.8);
    const y = height * (0.1 + (1 - normalizedY) * 0.8);

    context.strokeStyle = "#0b6e63";
    context.lineWidth = 2 * ratio;
    context.beginPath();
    context.moveTo(x - 7 * ratio, y);
    context.lineTo(x + 7 * ratio, y);
    context.moveTo(x, y - 7 * ratio);
    context.lineTo(x, y + 7 * ratio);
    context.stroke();
  }
}

function renderSimilarityPanel() {
  if (!appState.embeddings.length) {
    similarityStatus.textContent = "Build a corpus to render a similarity snapshot.";
    similarityOutput.replaceChildren();
    return;
  }

  const query = queryInput.value.trim();

  if (query && appState.queryEmbedding && appState.searchResults.length) {
    const top = appState.searchResults.slice(0, 5);
    const texts = [`query: ${query}`, ...top.map((entry) => appState.corpus[entry.index])];
    const embeddings = [appState.queryEmbedding, ...top.map((entry) => appState.embeddings[entry.index])];
    renderSimilarityMatrix(texts, embeddings);
    similarityStatus.textContent = "Similarity snapshot for the current query and its top matches.";
    return;
  }

  if (appState.selectedIndex != null) {
    const neighbors = getNearestNeighbors(appState.selectedIndex, 5);
    const indices = [appState.selectedIndex, ...neighbors.map((entry) => entry.index)];
    const texts = indices.map((index) => appState.corpus[index]);
    const embeddings = indices.map((index) => appState.embeddings[index]);
    renderSimilarityMatrix(texts, embeddings);
    similarityStatus.textContent = "Similarity snapshot for the selected item and its nearest neighbors.";
  }
}

function setSelectedIndex(index) {
  if (index == null || !appState.corpus[index]) {
    return;
  }

  appState.selectedIndex = index;
  benchmarkInput.value = appState.corpus[index];
  renderSelectedItem();
  renderNeighbors();
  renderSimilarityPanel();
  updateEmbeddingView(`Loaded the selected corpus item into the embedding profile.`);
  drawMap();
}

function runSearch() {
  if (!appState.embeddings.length) {
    searchStatus.textContent = "Build the corpus first.";
    return;
  }

  const query = queryInput.value.trim();
  if (!query) {
    searchStatus.textContent = "Enter a search query.";
    return;
  }

  const queryEmbedding = embeddingFromText(query);
  appState.queryEmbedding = queryEmbedding;
  appState.searchResults = appState.embeddings
    .map((embedding, index) => ({
      index,
      score: cosineSimilarity(queryEmbedding, embedding),
    }))
    .sort((left, right) => right.score - left.score);

  renderSearchResults();
  renderSimilarityPanel();
  drawMap();

  const best = appState.searchResults[0];
  if (best) {
    searchStatus.textContent =
      `Top match score: ${best.score.toFixed(3)} for "${appState.corpus[best.index]}".`;
    setSelectedIndex(best.index);
  }
}

function buildCorpus() {
  const texts = parseCorpusText(corpusInput.value);

  if (texts.length < 4) {
    corpusStatus.textContent = "Enter at least four non-empty lines.";
    return;
  }

  corpusStatus.textContent = "Encoding corpus in the browser...";

  const start = performance.now();
  const embeddings = texts.map((text) => embeddingFromText(text));
  const elapsed = performance.now() - start;

  appState.corpus = texts;
  appState.embeddings = embeddings;
  appState.selectedIndex = 0;
  appState.queryEmbedding = null;
  appState.searchResults = [];
  appState.hoverIndex = null;

  computeMapPoints();
  heroCorpus.textContent = `${texts.length} texts indexed`;
  corpusStatus.textContent =
    `Indexed ${texts.length} texts in ${fmt.format(elapsed)} ms inside the browser.`;
  searchResults.replaceChildren();
  searchStatus.textContent = "Type a query to rank the corpus.";
  mapStatus.textContent =
    "The map uses browser-side similarities to place related texts near each other. Click any point to inspect it.";
  mapHover.textContent = "Hover or click a point to see its text.";

  setSelectedIndex(0);
  drawMap();
}

function loadSampleCorpus() {
  corpusInput.value = SAMPLE_CORPUS;
  buildCorpus();
}

async function main() {
  await init();
  heroDims.textContent = `${dimensions()} dimensions`;
  corpusInput.value = SAMPLE_CORPUS;

  document.getElementById("run-benchmark").addEventListener("click", runBenchmark);
  document.getElementById("refresh-visual").addEventListener("click", () => updateEmbeddingView());
  document.getElementById("build-corpus").addEventListener("click", buildCorpus);
  document.getElementById("load-sample-corpus").addEventListener("click", loadSampleCorpus);
  document.getElementById("run-search").addEventListener("click", runSearch);

  mapCanvas.addEventListener("mousemove", (event) => {
    const nextHover = nearestPointFromMouse(event);
    if (nextHover !== appState.hoverIndex) {
      appState.hoverIndex = nextHover;
      mapHover.textContent =
        nextHover == null
          ? "Hover or click a point to see its text."
          : appState.corpus[nextHover];
      drawMap();
    }
  });

  mapCanvas.addEventListener("mouseleave", () => {
    appState.hoverIndex = null;
    mapHover.textContent = "Hover or click a point to see its text.";
    drawMap();
  });

  mapCanvas.addEventListener("click", (event) => {
    const index = nearestPointFromMouse(event);
    if (index != null) {
      setSelectedIndex(index);
    }
  });

  window.addEventListener("resize", () => {
    updateEmbeddingView();
    drawMap();
  });

  queryInput.value = "refund for subscription invoice";
  runBenchmark();
  buildCorpus();
  runSearch();
}

main().catch((error) => {
  benchmarkStatus.textContent = `Failed to initialize WASM: ${error}`;
  console.error(error);
});
