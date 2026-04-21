"""Shared utilities for cross-sectional backtesting examples.

Provides:

- [`Market`](stocks.py) — enum of the five A-shares markets currently
  supported: `SZSE`, `CHINEXT`, `SSE`, `STAR`, `BSE`.
- [`parse_market`](stocks.py) — classify a `CODE.SUFFIX` symbol (e.g.
  `"000001.SZ"`) into its `Market`, based on the suffix and the code's
  leading digits.
- [`add_market_argument`](stocks.py) — attach a ``--markets`` option to
  an `argparse` parser, letting examples restrict the symbol universe
  to a subset of markets.
- [`load_symbols`](stocks.py) — read the crawler's `symbol_list.csv`,
  optionally filtering by market.
- [`resolve_data_start`](stocks.py),
  [`calculate_index_weights`](stocks.py) — scenario-building helpers
  unrelated to market classification.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
import argparse
import csv

import numpy as np


class Market(StrEnum):
    """A-shares market of a listed instrument.

    Classification rules (see [`parse_market`][stocks.parse_market]):

    - `SZSE` — Shenzhen Stock Exchange main board (``.SZ``, code begins
      with ``0``: historical 000/001 main board plus 002/003 expansion).
    - `CHINEXT` — ChiNext (``.SZ``, code begins with ``3``).
    - `SSE` — Shanghai Stock Exchange main board (``.SH``, code begins
      with ``6``: 600/601/603/605).
    - `STAR` — Shanghai STAR Market (``.SH``, code begins with ``68``).
    - `BSE` — Beijing Stock Exchange (any ``.BJ`` symbol).
    """

    SZSE = "SZSE"
    CHINEXT = "CHINEXT"
    SSE = "SSE"
    STAR = "STAR"
    BSE = "BSE"


def parse_market(symbol: str) -> Market:
    """Return the [`Market`][stocks.Market] of an A-shares symbol.

    Parameters
    ----------
    symbol
        A symbol in ``CODE.SUFFIX`` form, e.g. ``"000001.SZ"``,
        ``"600519.SH"``, ``"832735.BJ"``.

    Raises
    ------
    ValueError
        If the suffix is not one of ``SZ``, ``SH``, ``BJ``.
    """
    code, _, suffix = symbol.partition(".")
    match suffix:
        case "SZ":
            return Market.CHINEXT if code.startswith("3") else Market.SZSE
        case "SH":
            return Market.STAR if code.startswith(("68")) else Market.SSE
        case "BJ":
            return Market.BSE
        case _:
            raise ValueError(f"Unrecognized A-shares symbol: {symbol!r}")


def add_market_argument(parser: argparse.ArgumentParser) -> None:
    """Attach a ``--markets`` option to `parser`.

    The option accepts one or more [`Market`][stocks.Market] names (e.g.
    ``--markets SZSE CHINEXT``); when omitted, all markets are included.
    The parsed value is a list of [`Market`][stocks.Market] members or
    ``None``, suitable for passing straight to
    [`load_symbols`][stocks.load_symbols]'s ``markets`` parameter.
    """
    choices_str = ", ".join(m.name for m in Market)
    parser.add_argument(
        "--markets",
        type=Market,
        nargs="+",
        default=None,
        choices=list(Market),
        metavar="MARKET",
        help=f"restrict the symbol universe to these markets ({{{choices_str}}}; default: all)",
    )


def load_symbols(data_dir: Path, markets: Iterable[Market] | None = None) -> list[str]:
    """Read stock symbols from the symbol list CSV.

    Parameters
    ----------
    data_dir
        Crawler data directory containing ``symbol_list.csv``.
    markets
        If provided, only symbols whose
        [`parse_market`][stocks.parse_market] lies in this collection
        are returned.  Defaults to all markets.
    """
    path = data_dir / "symbol_list.csv"
    if not path.exists():
        raise SystemExit(f"Symbol list not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        i = header.index("symbol")
        symbols = [row[i] for row in reader]
    if markets is not None:
        allowed = set(markets)
        symbols = [s for s in symbols if parse_market(s) in allowed]
    return symbols


def resolve_data_start(
    sample_begin: np.datetime64 | None,
    trading_begin: np.datetime64,
    rebalance_days: int,
) -> np.datetime64:
    """Resolve the data start date.

    If ``sample_begin`` is ``None``, returns ``trading_begin`` minus
    ``rebalance_days`` calendar days.  Otherwise returns
    ``min(sample_begin, trading_begin)`` (clamping sample_begin so it
    never exceeds the trading begin).
    """
    if sample_begin is None:
        return trading_begin - np.timedelta64(rebalance_days, "D")
    return min(sample_begin, trading_begin)


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
