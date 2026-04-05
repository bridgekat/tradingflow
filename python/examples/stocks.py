"""Shared utilities for cross-sectional backtesting examples."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np


def load_symbols(data_dir: Path) -> list[str]:
    """Read stock symbols from the symbol list CSV."""
    path = data_dir / "symbol_list.csv"
    if not path.exists():
        raise SystemExit(f"Symbol list not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        i = header.index("symbol")
        return [row[i] for row in reader]


def calculate_index_weights(market_cap: np.ndarray, k: int) -> np.ndarray:
    """Market-cap-weighted index weights for the top *k* stocks.

    Returns a ``(num_stocks,)`` array where the top ``k`` stocks by
    market cap have positive weights (proportional to market cap,
    normalised to sum to 1) and the rest have zero weight.
    """
    w = np.zeros_like(market_cap)
    valid = np.isfinite(market_cap) & (market_cap > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return w
    k = min(k, n_valid)
    scores = np.where(valid, market_cap, -np.inf)
    top = np.argpartition(-scores, k)[:k]
    w[top] = market_cap[top]
    s = w.sum()
    return w / s if s > 0 else w
