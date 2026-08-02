#!/usr/bin/env python
"""Render a strategy NAV CSV (index + one column per variant) with matplotlib.

Shared by the `mean_strategy`, `mean_variance_strategy`, and
`benchmark_relative_strategy` Rust examples, which all write a wide CSV whose
first column is `timestamp_ns`, second is the cap-weighted `index` baseline, and
the rest are per-variant NAV curves. Columns are NaN-masked independently.

Usage:
    python examples/plot_strategy.py target/mean_variance_strategy.csv [--save out.png]
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    cols = list(data.dtype.names)
    assert cols[0] == "timestamp_ns", cols
    t = data["timestamp_ns"].astype("datetime64[ns]")
    value_cols = cols[1:]

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 7))
    title = os.path.splitext(os.path.basename(args.csv))[0]
    ax.set_title(f"Portfolio value vs benchmark — {title}")
    ax.set_ylabel("CNY (10k)")
    ax.set_xlabel("Date")

    n_variants = max(len(value_cols) - 1, 1)
    cmap = plt.colormaps.get("viridis")
    for i, c in enumerate(value_cols):
        v = data[c].astype(float)
        m = np.isfinite(v)
        if c == "index":
            ax.plot(t[m], v[m] / 1e4, color="gray", linewidth=1.4, label="Index")
        else:
            color = cmap((i - 1) / max(n_variants - 1, 1)) if cmap else f"C{i}"
            ax.plot(t[m], v[m] / 1e4, color=color, linewidth=1.0, label=c)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
