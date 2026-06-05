#!/usr/bin/env python
"""Render the `factor_ic` Rust example's long-format CSV with matplotlib.

Reads `series,timestamp_ns,value` rows (one per factor per rebalance) and plots
the cumulative IC of each factor over time — mirroring
`python/examples/factor_ic.py`'s cumulative-IC figure.

Usage:
    python examples/plot_factor_ic.py target/factor_ic.csv [--save out.png]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    rows = np.genfromtxt(
        args.csv, delimiter=",", names=True, dtype=["U40", "i8", "f8"], encoding="utf-8"
    )

    groups: dict[str, list[tuple[int, float]]] = {}
    for name, ts, val in zip(rows["series"], rows["timestamp_ns"], rows["value"]):
        groups.setdefault(str(name), []).append((int(ts), float(val)))

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("Cumulative IC by factor")
    ax.set_ylabel("Cumulative IC")
    ax.set_xlabel("Date")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    for i, (name, pts) in enumerate(groups.items()):
        pts.sort()
        t = np.array([p[0] for p in pts], dtype="datetime64[ns]")
        v = np.array([p[1] for p in pts])
        cum = np.cumsum(np.where(np.isfinite(v), v, 0.0))
        ax.plot(t, cum, label=name, color=f"C{i}", linestyle="--" if "rank" in name else "-")

    ax.legend(fontsize=9)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
