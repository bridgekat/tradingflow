#!/usr/bin/env python
"""Render the `plot_total_market_cap` Rust example's CSV with matplotlib.

Two stacked panels: the index circulating market cap (trillions) and the
cap-weighted total-return
NAV. Columns are NaN-masked independently (the two series tick on slightly
different cadences).

Usage:
    python examples/plot_total_market_cap.py target/plot_total_market_cap.csv [--save out.png]
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

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    t = data["timestamp_ns"].astype("datetime64[ns]")

    def col(name: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        v = data[name].astype(float) * scale
        m = np.isfinite(v)
        return t[m], v[m]

    plt.style.use(["fast"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.set_title("Index circulating market cap (cap-weighted)")
    ax.set_ylabel("CNY (trillion)")
    tt, vv = col("index_circ_market_cap", 1e-12)
    ax.plot(tt, vv, color="C0", linewidth=0.8)

    ax = axes[1]
    ax.set_title("Index returns (cap-weighted, total return, unit cash)")
    ax.set_ylabel("NAV")
    ax.set_xlabel("Date")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    tt, vv = col("index_value")
    ax.plot(tt, vv, color="C0", linewidth=0.8)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
