"""Run the three benchmarks and render the readme charts in the Flower Computer style.

    uv run charts.py              # bench + render
    uv run charts.py --no-bench   # re-render from the last run
    uv run charts.py --no-rust    # only the python QPS bench, then render

Writes `.github/throughput.png` (bytes/s over gooaq text, log scale) and
`.github/qps.png` (sentences/s for the embedding libraries). Rust numbers are
read from criterion's `target/criterion/<group>/.../new/estimates.json`; the
python QPS numbers are measured by `qps.measure` and cached in
`benchmarks/qps.json`.
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flowercharts import TOKENS, fonts, style_axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CRITERION = ROOT / "target" / "criterion"
OUT = ROOT / ".github"
QPS_CACHE = HERE / "benchmarks" / "qps.json"

# criterion ids / model names -> label on the chart. Only ids listed here are
# charted (criterion keeps results of benches that no longer exist); bars are
# sorted by value.
THROUGHPUT_LABELS = {
    "encoders": "encoders",
    "encoders-rayon": "encoders\n(rayon)",
    "tokenize": "HF tokenizers\nWordPiece only",
    "sha256_soft": "SHA-256\nno hw accel",
    "gzip": "gzip\nflate2 default",
    "tantivy_index": "tantivy index\nin-memory",
    "json_decode": "JSON decode\nserde_json crate",
}
QPS_LABELS = {
    "encoders": "encoders",
    "encoders-rayon": "encoders\n(rayon)",
    "model2vec-rs": "model2vec-rs\npotion-base-8M",
    "candle/all-MiniLM-L6-v2": "candle\nall-MiniLM-L6-v2",
    "minishlab/potion-base-8M": "python\npotion-base-8M",
    "sentence-transformers/all-MiniLM-L6-v2": "python\nall-MiniLM-L6-v2",
}
PY_MODELS = ["minishlab/potion-base-8M", "sentence-transformers/all-MiniLM-L6-v2"]
HIGHLIGHT = {"encoders", "encoders-rayon"}
# Every bar other than `HIGHLIGHT` is re-cut in OKLCH: chroma scaled by
# OTHER_SATURATION (0 = gray, 1 = the palette's own chroma) and lightness
# pinned to OTHER_LIGHTNESS (0..1, perceptual) so dark slots such as green
# don't stand out. Each bar has its own categorical slot, fixed by its position
# in the label table (encoders is always slot 1), so colors follow the entity
# rather than the rank.
OTHER_SATURATION = 0.33
OTHER_LIGHTNESS = 0.9


def _srgb_to_oklab(rgb):
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    r, g, b = lin
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s_,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s_,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s_,
    )


def _oklab_to_srgb(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s_,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s_,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s_,
    )
    out = []
    for c in lin:
        c = min(max(c, 0.0), 1.0)
        out.append(12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)
    return tuple(out)


def recut(hex_color: str, chroma_keep: float, lightness: float) -> str:
    """Same hue, chroma scaled by `chroma_keep`, lightness set to `lightness`
    (OKLCH), clipped to sRGB."""
    L, a, b = _srgb_to_oklab(to_rgb(hex_color))
    return to_hex(_oklab_to_srgb((lightness, a * chroma_keep, b * chroma_keep)))


def bar_colors(keys: list[str], labels: dict[str, str], t) -> list[str]:
    slots = [k for k in labels if k not in HIGHLIGHT]
    out = []
    for k in keys:
        if k in HIGHLIGHT:
            out.append(t["series"][0])
        else:
            out.append(
                recut(
                    t["series"][1 + slots.index(k)], OTHER_SATURATION, OTHER_LIGHTNESS
                )
            )
    return out


def run_rust_benches():
    # Each bench twice: the default build, then `--features rayon`, which
    # records only the parallel encoders bar under a second id.
    for bench in ("throughput", "embedders"):
        for extra in ([], ["--features", "rayon"]):
            subprocess.run(
                ["cargo", "bench", "-p", "encoders-bench", "--bench", bench, *extra],
                cwd=ROOT,
                check=True,
            )


def run_python_bench() -> dict[str, float]:
    from qps import measure

    qps = measure(PY_MODELS)
    QPS_CACHE.parent.mkdir(exist_ok=True)
    QPS_CACHE.write_text(json.dumps(qps, indent=2))
    return qps


def criterion_results(group: str) -> dict[str, float]:
    """Mean throughput per bench id in the group: bytes/s or elements/s."""
    out = {}
    for bench_json in (CRITERION / group).rglob("new/benchmark.json"):
        meta = json.loads(bench_json.read_text())
        est = json.loads((bench_json.parent / "estimates.json").read_text())
        secs = est["mean"]["point_estimate"] / 1e9
        ((unit, count),) = meta["throughput"].items()
        assert unit in ("Bytes", "Elements"), unit
        out[meta["function_id"]] = count / secs
    if not out:
        print(
            f"no criterion results under {CRITERION / group}; skipping its chart",
            file=sys.stderr,
        )
    return out


def fmt_rate(v: float, unit: str) -> str:
    """Decimal units with three significant figures and no trailing zeros:
    `1 GB/s`, `37 MB/s`, `4.98 MB/s`, `241k`, `10.9M`."""
    scales = (
        ((1e9, "G"), (1e6, "M"), (1e3, "k"))
        if unit == "B/s"
        else ((1e6, "M"), (1e3, "k"))
    )
    suffix = " B/s" if unit == "B/s" else ""
    for div, prefix in scales:
        if v >= div:
            x = v / div
            num = f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"
            num = num.rstrip("0").rstrip(".") if "." in num else num
            return (
                f"{num} {prefix}{suffix.strip()}" if unit == "B/s" else f"{num}{prefix}"
            )
    return f"{v:.0f}{suffix}"


def bars(
    values: dict[str, float],
    labels: dict[str, str],
    title: str,
    ylabel: str,
    out: Path,
    *,
    log: bool,
    unit: str,
    mode: str = "light",
):
    """Vertical bars, lowest to highest. `HIGHLIGHT` takes categorical slot 1
    at full saturation; the rest get their own slots at `OTHER_SATURATION`.
    Values in mono above each bar; only the highlight label wears its color."""
    t = TOKENS[mode]
    fams = fonts.register()
    plt.rcParams["font.family"] = [fams["text"], "DejaVu Sans"]

    items = sorted(
        ((k, v) for k, v in values.items() if k in labels), key=lambda kv: kv[1]
    )
    if not items:
        return
    xs = range(len(items))
    cols = bar_colors([k for k, _ in items], labels, t)

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style_axes(ax, t)
    # Numeric ticks on the right so the eye reads bar height and value together;
    # the axis title stays on the left.
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("left")
    ax.bar(xs, [v for _, v in items], color=cols, width=0.62, zorder=3)
    lo, hi = min(v for _, v in items), max(v for _, v in items)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(lo / 4, hi * 2.2)
        decades = [
            10.0**e
            for e in range(
                int(np.floor(np.log10(lo / 4))), int(np.ceil(np.log10(hi * 2.2))) + 1
            )
        ]
        ax.yaxis.set_major_locator(FixedLocator(decades))
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_ylim(0, hi * 1.12)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: fmt_rate(v, unit) if v > 0 else "0")
    )
    for x, (k, v), col in zip(xs, items, cols):
        ax.annotate(
            fmt_rate(v, unit),
            (x, v),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            family=fams["mono"],
            color=col if k in HIGHLIGHT else t["ink2"],
            zorder=5,
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [labels.get(k, k) for k, _ in items], fontsize=8.5, color=t["ink2"]
    )
    for lab, (k, _) in zip(ax.get_xticklabels(), items):
        if k in HIGHLIGHT:
            lab.set_color(t["ink"])
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9.5)
    ax.set_title(
        title,
        loc="left",
        fontsize=14,
        family=fams["title"],
        fontweight=400,
        color=t["ink"],
        pad=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out)


def main():
    bench = "--no-bench" not in sys.argv
    if bench:
        if "--no-rust" not in sys.argv:
            run_rust_benches()
        qps_py = run_python_bench()
    else:
        qps_py = json.loads(QPS_CACHE.read_text()) if QPS_CACHE.exists() else {}

    throughput = criterion_results("throughput")
    bars(
        throughput,
        THROUGHPUT_LABELS,
        "Throughput (GooAQ corpus)",
        "bytes per second, log scale",
        OUT / "throughput.png",
        log=True,
        unit="B/s",
    )

    qps = criterion_results("embedders") | qps_py
    bars(
        qps,
        QPS_LABELS,
        "Queries embedded per second (GooAQ)",
        "queries per second",
        OUT / "qps.png",
        log=False,
        unit="qps",
    )


if __name__ == "__main__":
    main()
