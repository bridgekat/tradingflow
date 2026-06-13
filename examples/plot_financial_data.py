#!/usr/bin/env python
"""Render the `plot_financial_data` Rust example's CSV with matplotlib.

Three stacked panels: balance sheet & market cap, annualized income & cash
flow, valuation ratios.
Each column is masked for NaN/inf independently so the daily (market cap,
ratios) and quarterly (balance/income/cash-flow) series each render on their
own cadence.

Usage:
    python examples/plot_financial_data.py target/plot_financial_data.csv [--save out.png]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="wide CSV from the plot_financial_data Rust example")
    ap.add_argument("--save", default=None, help="save PNG here instead of showing")
    args = ap.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    t = data["timestamp_ns"].astype("datetime64[ns]")

    def col(name: str) -> np.ndarray:
        v = data[name].astype(float)
        return np.where(np.isfinite(v), v, np.nan)

    def plot(ax, name: str, *, scale: float = 1.0, **kw) -> None:
        v = col(name) * scale
        m = np.isfinite(v)
        ax.plot(t[m], v[m], **kw)

    plt.style.use(["fast"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: market cap, total assets, net assets (equity), parent equity.
    ax = axes[0]
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    plot(ax, "assets", scale=1e-8, label="Total assets", color="C0", linewidth=0.8)
    plot(ax, "equity", scale=1e-8, label="Net assets", color="C1", linewidth=0.8)
    plot(ax, "parent_equity", scale=1e-8, label="Parent equity", color="C1", linestyle="--", linewidth=0.8)
    plot(ax, "market_cap", scale=1e-8, label="Market cap", color="C2", linewidth=0.8)
    ax.set_ylabel("CNY (100M)")
    ax.set_title("Balance sheet & market cap")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: annualized income statement & cash flows.
    ax = axes[1]
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    plot(ax, "op_income", scale=1e-8, label="Operating income", color="C0", linewidth=0.8)
    plot(ax, "net_profit", scale=1e-8, label="Net profit", color="C1", linewidth=0.8)
    plot(ax, "cash_flow", scale=1e-8, label="Cash flow", color="C2", linewidth=0.8)
    ax.set_ylabel("CNY (100M, annualized)")
    ax.set_title("Income & cash flow (annualized)")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3: valuation ratios (%).
    ax = axes[2]
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    plot(ax, "ep_ratio", scale=100.0, label="E/P (TTM)", color="C0", linewidth=0.8)
    plot(ax, "bp_ratio", scale=100.0, label="B/P", color="C1", linewidth=0.8)
    plot(ax, "roe", scale=100.0, label="ROE (TTM)", color="C2", linewidth=0.8)
    ax.set_ylabel("%")
    ax.set_title("Valuation & profitability")
    ax.set_xlabel("Date")
    ax.set_ylim(-50, 50)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
