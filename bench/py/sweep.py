"""How model choice, width and storage type move encoders' single-core speed.

Rebuilds `bench/examples/sweep_probe.rs` once per configuration and records
its JSON line plus the embedded table size from ese-build's build summary,
then renders three charts into `target/sweep/`:

- `sweep-dims.png`: bekko-100k at 32, 256 and 1024 dims, per dtype.
- `sweep-vocab.png`: the bekko slim 32k, 50k and 100k vocabularies at 256
  dims (int16).
- `sweep-rows.png`: bekko-100k against bytes per row (dims x element size),
  each point labelled with its width.

    uv run sweep.py              # measure missing configs, then render
    uv run sweep.py --no-bench   # render from target/sweep/results.jsonl
    uv run sweep.py --fresh      # discard previous results first

Results are appended as each configuration finishes, so an interrupted sweep
resumes where it stopped. Run it on an otherwise idle machine.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flowercharts import TOKENS, fonts, style_axes
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "target" / "sweep"
RESULTS = OUT / "results.jsonl"

DTYPES = ["float32", "int16", "int8"]
DTYPE_LABEL = {"float32": "f32", "int16": "int16", "int8": "int8"}
WIDTHS = [32, 256, 1024]
VOCABS = ["32k", "50k", "100k"]
TEXTS = OUT / "texts"
N = 4096


def configs():
    seen = set()
    for dtype in DTYPES:
        for dims in WIDTHS:
            seen.add(("100k", dims, dtype))
    for vocab in VOCABS:
        seen.add((vocab, 256, "int16"))
    order = {v: i for i, v in enumerate(VOCABS)}
    return sorted(seen, key=lambda c: (DTYPES.index(c[2]), order[c[0]], c[1]))


def write_texts():
    """The first N gooaq questions and answers, one per line, from the parquet
    the Rust benches already cached."""
    import pyarrow.parquet as pq

    if (TEXTS / "answers.txt").exists():
        return
    TEXTS.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(ROOT / "target" / "gooaq" / "gooaq_0.parquet").to_pydict()
    for name, cols in (("questions", ("question", "sentence1")), ("answers", ("answer", "sentence2"))):
        col = next(c for c in cols if c in table)
        lines = [" ".join(t.split()) for t in table[col][:N]]
        (TEXTS / f"{name}.txt").write_text("\n".join(lines) + "\n")


def key(c):
    return f"{c[0]}/{c[1]}/{c[2]}"


def parse_size(text: str) -> float:
    num, unit = re.fullmatch(r"([\d.]+)\s*(B|KiB|MiB|GiB)", text).groups()
    return float(num) * {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}[unit]


def measure(c) -> dict:
    vocab, dims, dtype = c
    features = ",".join(
        f"encoders/{f}"
        for f in (f"bekko-static-v1-slim-{vocab}", f"mrl-{dims}", dtype)
    )
    build = subprocess.run(
        ["cargo", "build", "--release", "-p", "encoders-bench",
         "--example", "sweep_probe", "--features", features],
        cwd=ROOT, capture_output=True, text=True,
    )
    if build.returncode != 0:
        sys.exit(build.stderr)
    summary = re.search(r"ese: model=.*", build.stderr)
    assert summary, "no ese-build summary in the build output"
    fields = dict(kv.split("=", 1) for kv in summary.group(0)[len("ese: "):].split(" ") if "=" in kv)
    embedded = re.search(r"embedded=([\d.]+ \w+)", summary.group(0)).group(1)
    run = subprocess.run(
        [ROOT / "target" / "release" / "examples" / "sweep_probe",
         TEXTS / "questions.txt", TEXTS / "answers.txt"],
        capture_output=True, text=True, check=True,
    )
    row = json.loads(run.stdout.strip().splitlines()[-1])
    row.update(
        key=key(c), vocab_label=vocab, dtype=dtype, quant=fields.get("quant"),
        table_slots=int(fields["table"]), lut_bytes=parse_size(embedded),
    )
    return row


def load() -> dict[str, dict]:
    if not RESULTS.exists():
        return {}
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    return {r["key"]: r for r in rows}


def sweep():
    done = load()
    todo = [c for c in configs() if key(c) not in done]
    OUT.mkdir(parents=True, exist_ok=True)
    write_texts()
    for i, c in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {key(c)} ...", flush=True)
        row = measure(c)
        with RESULTS.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"    {row['queries_per_s'] / 1e6:.2f}M queries/s, "
            f"{row['docs_per_s'] / 1e3:.0f}k docs/s, "
            f"table {row['lut_bytes'] / 2**20:.1f} MiB",
            flush=True,
        )


def si(v, _=None):
    for div, p in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            x = v / div
            return f"{x:.3g}{p}"
    return f"{v:.0f}"


METRICS = [
    ("queries_per_s", "Queries (GooAQ questions, ~43 B)", "queries per second"),
    ("docs_per_s", "Documents (GooAQ answers, ~252 B)", "documents per second"),
]


def panels(title: str, subtitle: str, out: Path, draw):
    t = TOKENS["light"]
    fams = fonts.register()
    plt.rcParams["font.family"] = [fams["text"], "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    for ax, (metric, name, ylabel) in zip(axes, METRICS):
        style_axes(ax, t)
        draw(ax, metric, t, fams)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(si))
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9.5)
        ax.set_title(name, loc="left", fontsize=10.5, color=t["ink"], pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=len(labels),
               fontsize=9, labelcolor=t["ink2"], bbox_to_anchor=(0.99, 0.985))
    fig.suptitle(title, x=0.012, y=0.975, ha="left", fontsize=14,
                 family=fams["title"], fontweight=400, color=t["ink"])
    fig.text(0.012, 0.905, subtitle, ha="left", fontsize=9, color=t["muted"])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out)


def line_panel(rows, xs, xkey, xlabels, xlabel):
    def draw(ax, metric, t, fams):
        for k, dtype in enumerate(DTYPES):
            pts = [(i, r[metric]) for i, x in enumerate(xs)
                   for r in rows if r["dtype"] == dtype and r[xkey] == x]
            if not pts:
                continue
            col = t["series"][k]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linewidth=2,
                    solid_capstyle="round", zorder=3, label=DTYPE_LABEL[dtype])
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=64, color=col,
                       edgecolors=t["surface"], linewidths=2, zorder=4)
            # Direct label at the line's end.
            ax.annotate(DTYPE_LABEL[dtype], pts[-1], textcoords="offset points",
                        xytext=(8, 0), va="center", fontsize=8.5, color=t["ink2"])
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xlabels, fontsize=9, color=t["ink2"])
        ax.set_xlim(-0.3, len(xs) - 0.4)
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9.5)
    return draw


ELEM_BYTES = {"float32": 4, "int16": 2, "int8": 1}


def row_bytes(r) -> int:
    return r["dims"] * ELEM_BYTES[r["dtype"]]


def bytes_label(v, _=None):
    return f"{v / 1024:g} KiB" if v >= 1024 else f"{v:g} B"


def row_panel(rows):
    """Rate against bytes read per token (dims x element size), labelled with
    each point's width, so equal-bytes configurations with different column
    counts (256-dim f32 vs 1024-dim int8) sit side by side."""
    def draw(ax, metric, t, fams):
        ax.set_xscale("log", base=2)
        for k, dtype in enumerate(DTYPES):
            rs = sorted((r for r in rows if r["dtype"] == dtype), key=row_bytes)
            xs, ys = [row_bytes(r) for r in rs], [r[metric] for r in rs]
            col = t["series"][k]
            ax.plot(xs, ys, color=col, linewidth=2, solid_capstyle="round", zorder=3,
                    label=DTYPE_LABEL[dtype])
            ax.scatter(xs, ys, s=64, color=col, edgecolors=t["surface"], linewidths=2, zorder=4)
            for r, x, y in zip(rs, xs, ys):
                ax.annotate(f"{r['dims']}d", (x, y), textcoords="offset points", xytext=(6, 5),
                            fontsize=7.5, family=fams["mono"], color=t["muted"])
        ticks = [2**e for e in range(5, 13)]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(bytes_label))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        ax.set_xlim(24, 5800)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_xlabel("bytes per row (dimensions x element size), log scale",
                      color=t["ink2"], fontsize=9.5)
    return draw


def render():
    rows = list(load().values())
    if not rows:
        sys.exit(f"no results in {RESULTS}")
    note = "single core, encode_single per text, median of repeated passes over 4096 texts; Apple M3 Pro"
    panels(
        "Width: bekko-static-v1-slim-100k at three Matryoshka widths",
        note, OUT / "sweep-dims.png",
        line_panel([r for r in rows if r["vocab_label"] == "100k"],
                   WIDTHS, "dims", [str(w) for w in WIDTHS], "dimensions"),
    )
    panels(
        "Vocabulary: bekko-static-v1-slim at 256 dimensions",
        note, OUT / "sweep-vocab.png",
        line_panel([r for r in rows if r["dims"] == 256],
                   VOCABS, "vocab_label", VOCABS, "vocabulary size"),
    )
    panels(
        "Bytes per row: bekko-static-v1-slim-100k, every width and dtype",
        note, OUT / "sweep-rows.png",
        row_panel([r for r in rows if r["vocab_label"] == "100k"]),
    )


if __name__ == "__main__":
    if "--fresh" in sys.argv and RESULTS.exists():
        RESULTS.unlink()
    if "--no-bench" not in sys.argv:
        sweep()
    render()
