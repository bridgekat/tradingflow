#!/usr/bin/env python
"""Render a flow-engine demo CSV with matplotlib.

Standalone plotting companion to the Rust examples: the Rust binary computes and
writes a CSV, this script reads it and draws the figure. Keeping plotting in a
separate Python script means the Rust examples need no plotting dependency and
run without a GUI / the free-threaded interpreter.

Usage:
    python examples/plot.py target/flowops_demo.csv [--save out.png]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV with a timestamp_ns column + value columns")
    ap.add_argument("--save", default=None, help="save PNG here instead of showing")
    args = ap.parse_args()

    # Plain CSV read (no pandas dependency required).
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    cols = list(data.dtype.names)
    assert cols[0] == "timestamp_ns", f"expected timestamp_ns first, got {cols}"
    t = data["timestamp_ns"].astype("datetime64[ns]")
    value_cols = cols[1:]

    fig, ax = plt.subplots(figsize=(10, 4))
    for c in value_cols:
        ax.plot(t, data[c], label=c, linewidth=1.2)
    ax.set_xlabel("time")
    ax.set_ylabel(", ".join(value_cols))
    ax.set_title(f"flow demo — {args.csv}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
