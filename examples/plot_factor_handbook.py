#!/usr/bin/env python
"""Render the `factor_handbook` Rust example's long-format CSVs with matplotlib.

Two inputs (either or both):

- the IC CSV (`target/factor_handbook_ic.csv`, `series,timestamp_ns,value` =
  per-factor per-rebalance RankIC) → a cumulative-IC curve per factor;
- the NAV CSV (`target/factor_handbook_nav.csv`, series named `<factor>_g1..g10`
  and `<factor>_bench`) → per factor a decile-NAV panel (relative to the
  equal-weight benchmark) and a per-group annualized-return bar chart (分层回测).

Usage:
    python examples/plot_factor_handbook.py target/factor_handbook_nav.csv \
        [--ic target/factor_handbook_ic.csv] [--save out.png]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

YEAR_NS = 365.0 * 86_400.0 * 1e9


def load_long(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = np.genfromtxt(
        path, delimiter=",", names=True, dtype=["U40", "i8", "f8"], encoding="utf-8"
    )
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for name, ts, val in zip(rows["series"], rows["timestamp_ns"], rows["value"]):
        out[str(name)].append((int(ts), float(val)))
    series = {}
    for name, pts in out.items():
        pts.sort()
        t = np.array([p[0] for p in pts], dtype="i8")
        v = np.array([p[1] for p in pts], dtype="f8")
        series[name] = (t, v)
    return series


def cagr(t: np.ndarray, v: np.ndarray) -> float:
    m = np.isfinite(v) & (v > 0)
    if m.sum() < 2:
        return np.nan
    t, v = t[m], v[m]
    years = (t[-1] - t[0]) / YEAR_NS
    return (v[-1] / v[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nav", help="factor_handbook_nav.csv")
    ap.add_argument("--ic", default=None, help="factor_handbook_ic.csv (optional)")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    nav = load_long(args.nav)
    # Group series by factor: "<factor>_g<k>" / "<factor>_bench".
    factors: dict[str, dict[str, str]] = defaultdict(dict)
    for name in nav:
        factor, _, tail = name.rpartition("_")
        factors[factor][tail] = name

    plt.style.use(["fast"])
    nrows = len(factors) + (1 if args.ic else 0)
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 4 * nrows), squeeze=False)

    for r, (factor, parts) in enumerate(sorted(factors.items())):
        ax_nav, ax_bar = axes[r]
        bench_t, bench_v = nav[parts["bench"]] if "bench" in parts else (None, None)
        anns = []
        labels = []
        for k in range(1, 11):
            key = f"g{k}"
            if key not in parts:
                continue
            t, v = nav[parts[key]]
            base = v[np.isfinite(v)][0]
            rel = v / base
            ax_nav.plot(
                t.astype("datetime64[ns]"),
                rel,
                color=plt.cm.RdYlGn((k - 1) / 9),
                linewidth=1.0,
                label=f"g{k}",
            )
            anns.append(cagr(t, v) * 100.0)
            labels.append(f"g{k}")
        if bench_t is not None:
            base = bench_v[np.isfinite(bench_v)][0]
            ax_nav.plot(
                bench_t.astype("datetime64[ns]"),
                bench_v / base,
                color="black",
                linewidth=1.4,
                linestyle="--",
                label="bench",
            )
            bench_ann = cagr(bench_t, bench_v) * 100.0
        else:
            bench_ann = np.nan
        ax_nav.set_yscale("log")
        ax_nav.set_title(f"{factor}: decile NAV (normalized)")
        ax_nav.legend(fontsize=7, ncol=2)

        colors = ["C2" if a >= bench_ann else "C3" for a in anns]
        ax_bar.bar(labels, anns, color=colors)
        if np.isfinite(bench_ann):
            ax_bar.axhline(bench_ann, color="black", linestyle="--", linewidth=1)
        ax_bar.set_title(f"{factor}: group annualized return %")
        ax_bar.tick_params(axis="x", labelrotation=0)

    if args.ic:
        ic = load_long(args.ic)
        ax = axes[-1][0]
        ax.set_title("Cumulative RankIC by factor")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        for i, (name, (t, v)) in enumerate(sorted(ic.items())):
            cum = np.cumsum(np.where(np.isfinite(v), v, 0.0))
            ax.plot(t.astype("datetime64[ns]"), cum, label=name, color=f"C{i}")
        ax.legend(fontsize=8)
        axes[-1][1].axis("off")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
