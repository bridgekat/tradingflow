"""Shared utilities and pipeline helpers for all A-shares examples.

A single source of truth for everything the example scripts need
beyond their strategy-specific logic.  Sections:

1. **A-shares market classification** -- [`Market`][common.Market],
   [`parse_market`][common.parse_market],
   [`add_market_argument`][common.add_market_argument],
   [`load_symbols`][common.load_symbols].
2. **CLI argument boilerplate** --
   [`add_common_arguments`][common.add_common_arguments],
   [`validate_data_dir`][common.validate_data_dir].
3. **Date helpers** --
   [`resolve_data_start`][common.resolve_data_start].
4. **Per-stock data pipeline** -- A-shares CSV schemas and
   [`build_stacked`][common.build_stacked], which loops over symbols,
   registers CSV / FinancialReport sources, derives the canonical base
   per-stock handles (close, volume, adjusts, adjusted_close,
   total_shares, circ_shares, parent_equity, net_profit), and
   stacks them into the cross-sectional ``stacked`` dict.
5. **Cross-sectional feature set** --
   [`build_features`][common.build_features], which derives the eight
   canonical factors (three percentile-ranked fundamentals plus
   ``momentum_ma``, ``volatility``, ``turnover_ma``, ``volume_ratio``)
   and the trading-day-aligned recorded panel.
6. **Scenario building blocks** --
   [`build_rebalance_clock`][common.build_rebalance_clock],
   [`build_cap_weighted_universe`][common.build_cap_weighted_universe],
   [`build_demeaned_log_return_target`][common.build_demeaned_log_return_target],
   [`build_price_limits`][common.build_price_limits],
   [`calculate_index_weights`][common.calculate_index_weights].
7. **Benchmark scaling for plots** --
   [`find_effective_trading_start`][common.find_effective_trading_start],
   [`scale_to_initial_cash`][common.scale_to_initial_cash].
8. **Progress reporting** --
   [`make_progress_tracker`][common.make_progress_tracker].

Routing every example through these helpers keeps mean_strategy.py,
mean_variance_strategy.py, covariance_gmv.py, factor_ic.py, and the
plot scripts aligned on data layout, factor definitions, and scenario
boilerplate.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Handle, Scenario, Schema
from tradingflow.operators import Apply, Clocked, Lag, Map, Record, Resample, Select, Stack, StackSync
from tradingflow.operators.num import Diff, Divide, Log, Max, Multiply, Percentile, Sqrt, Subtract, Winsorize
from tradingflow.operators.rolling import RollingMean, RollingVariance
from tradingflow.operators.stocks import Annualize, ForwardAdjust
from tradingflow.sources import Clock, CSVSource
from tradingflow.sources.stocks import FinancialReportSource

# =====================================================================
# A-shares market classification
# =====================================================================


class Market(StrEnum):
    """A-shares market of a listed instrument.

    Classification rules (see [`parse_market`][common.parse_market]):

    - `SZSE` - Shenzhen Stock Exchange main board (``.SZ``, code begins
      with ``0``: historical 000/001 main board plus 002/003 expansion).
    - `CHINEXT` - ChiNext (``.SZ``, code begins with ``3``).
    - `SSE` - Shanghai Stock Exchange main board (``.SH``, code begins
      with ``6``: 600/601/603/605).
    - `STAR` - Shanghai STAR Market (``.SH``, code begins with ``68``).
    - `BSE` - Beijing Stock Exchange (any ``.BJ`` symbol).
    """

    SZSE = "SZSE"
    CHINEXT = "CHINEXT"
    SSE = "SSE"
    STAR = "STAR"
    BSE = "BSE"


def parse_market(symbol: str) -> Market:
    """Return the [`Market`][common.Market] of an A-shares symbol.

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

    The option accepts one or more [`Market`][common.Market] names
    (e.g. ``--markets SZSE CHINEXT``); when omitted, all markets are
    included.  The parsed value is a list of [`Market`][common.Market]
    members or ``None``, suitable for passing straight to
    [`load_symbols`][common.load_symbols]'s ``markets`` parameter.
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
        [`parse_market`][common.parse_market] lies in this collection
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


# =====================================================================
# CLI argument boilerplate
# =====================================================================


def add_common_arguments(parser: argparse.ArgumentParser, *, include_initial_cash: bool = False) -> None:
    """Attach the CLI options shared by every example.

    Adds ``--data-dir``, ``--sample-begin``, ``-b/--begin``, ``-e/--end``,
    ``--rebalance-days``, ``--index-size``, ``--markets``.  Pass
    ``include_initial_cash=True`` to also add ``--initial-cash`` (used
    by the trading examples to set the
    [`RandomTrader`][tradingflow.operators.traders.simple.RandomTrader]
    starting capital and to rebase unit-cash :class:`Benchmark` curves
    to a comparable y-axis via
    [`scale_to_initial_cash`][common.scale_to_initial_cash]).
    """
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument("--sample-begin", type=np.datetime64, default=None, help="data sampling start date")
    parser.add_argument("-b", "--begin", type=np.datetime64, required=True, help="start date (e.g. 2020-01-01)")
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    parser.add_argument("--rebalance-days", type=int, default=90, help="rebalance every N calendar days")
    parser.add_argument("--index-size", type=int, default=100, help="number of stocks in the cap-weighted universe")
    if include_initial_cash:
        parser.add_argument("--initial-cash", type=float, default=1000000.0, help="starting capital (CNY)")
    add_market_argument(parser)


def validate_data_dir(args: argparse.Namespace) -> tuple[Path, list[str]]:
    """Resolve ``args.data_dir``, load filtered symbols, and print a count.

    Returns ``(data_dir, symbols)``.  Raises
    [`SystemExit`][SystemExit] (with a hint pointing at the crawler)
    if the directory doesn't exist.
    """
    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )
    symbols = load_symbols(data_dir, markets=args.markets)
    print(f"Discovered {len(symbols)} symbols.")
    return data_dir, symbols


# =====================================================================
# Date helpers
# =====================================================================


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


# =====================================================================
# Per-stock data pipeline
# =====================================================================


PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())
DIVIDEND_SCHEMA = Schema(CSVSchema.dividends().iter_field_ids())
BS_SCHEMA = Schema(CSVSchema.balance_sheet().iter_field_ids())
INC_SCHEMA = Schema(CSVSchema.income_statement().iter_field_ids())


def build_stacked(
    sc: Scenario,
    symbols: list[str],
    data_dir: Path,
    *,
    data_start: np.datetime64,
    end: np.datetime64,
    progress_desc: str = "Building scenario",
) -> dict[str, Handle]:
    """Load per-stock CSVs and stack into the cross-sectional panel.

    Iterates over `symbols`, registering one set of CSV /
    FinancialReport sources per stock and deriving the canonical base
    handles, then stacks all handles cross-sectionally into the
    returned dict.

    Two internal dicts carry different update semantics, surfaced via
    separate stackers:

    * ``per_stock_sync`` -- values that produce in lockstep across all
      stocks (daily prices and derived adjusted prices).  Stacked with
      :class:`StackSync` so slots of stocks that did not produce this
      cycle are filled with ``NaN``.
    * ``per_stock_irregular`` -- values updated on stock-specific
      dates (quarterly financial reports, equity-structure
      adjustments).  Stacked with :class:`Stack` so slots keep their
      last-known value across quiet periods.

    Parameters
    ----------
    sc
        Scenario under construction.
    symbols
        List of stock symbols to load.  Each symbol ``s`` corresponds
        to a set of files under
        ``data_dir / "a_shares_history" / s.daily_prices.csv``,
        ``...s.equity_structures.csv``, ``...s.dividends.csv``,
        ``...s.balance_sheets.csv``, ``...s.income_statements.csv``.
    data_dir
        Root data directory (typically passed in via ``--data-dir``).
    data_start, end
        Time window for the CSV sources.
    progress_desc
        Description string for the ``tqdm`` progress bar.

    Returns
    -------
    dict[str, Handle]
        Cross-sectional ``stacked`` dict.  Keys:

        * Lockstep (``StackSync``):
          ``close``, ``volume``, ``adjusted_close``
          (each shape ``(num_stocks,)``).
        * Irregular (``Stack``):
          ``adjusts``, ``total_shares``, ``circ_shares``,
          ``parent_equity``, ``net_profit``
          (each shape ``(num_stocks,)``).
    """
    history_dir = data_dir / "a_shares_history"
    per_stock_sync: dict[str, list[Handle]] = {k: [] for k in ("close", "volume", "adjusted_close")}
    per_stock_irregular: dict[str, list[Handle]] = {
        k: [] for k in ("adjusts", "total_shares", "circ_shares", "parent_equity", "net_profit")
    }

    for symbol in tqdm(symbols, desc=progress_desc):
        h = history_dir / symbol

        # Sources.
        prices = sc.add_source(
            CSVSource(f"{h}.daily_prices.csv", PRICE_SCHEMA, time_column="date", start=data_start, end=end)
        )
        equity = sc.add_source(
            CSVSource(f"{h}.equity_structures.csv", EQUITY_SCHEMA, time_column="date", start=data_start, end=end)
        )
        dividends = sc.add_source(
            CSVSource(f"{h}.dividends.csv", DIVIDEND_SCHEMA, time_column="date", start=data_start, end=end)
        )
        balance = sc.add_source(
            FinancialReportSource(
                f"{h}.balance_sheets.csv",
                BS_SCHEMA,
                report_date_column="date",
                notice_date_column="notice_date",
                use_effective_date=False,
                start=data_start,
                end=end,
            )
        )
        income_ytd = sc.add_source(
            FinancialReportSource(
                f"{h}.income_statements.csv",
                INC_SCHEMA,
                report_date_column="date",
                notice_date_column="notice_date",
                with_report_date=True,
                use_effective_date=False,
                start=data_start,
                end=end,
            )
        )

        # Derived per-stock handles.
        close = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.close")))
        volume = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.volume")))
        adjusts = sc.add_operator(ForwardAdjust(close, dividends, output_prices=False))
        adjusted_close = sc.add_operator(Multiply(close, adjusts))
        total_shares = sc.add_operator(Select(equity, EQUITY_SCHEMA.index("shares.total")))
        circ_shares = sc.add_operator(Select(equity, EQUITY_SCHEMA.index("shares.circulating")))
        income_ann = sc.add_operator(Annualize(income_ytd))
        net_profit = sc.add_operator(Select(income_ann, INC_SCHEMA.index("income_statement.profit")))
        neg_peq = sc.add_operator(
            Select(
                balance,
                BS_SCHEMA.indices(
                    [
                        "balance_sheet.equity.capital",
                        "balance_sheet.equity.reserves",
                        "balance_sheet.equity.parent_interests",
                    ]
                ),
            )
        )
        parent_equity = sc.add_operator(Map(neg_peq, lambda x: -x.sum(), shape=(), dtype=np.float64))

        per_stock_sync["close"].append(close)
        per_stock_sync["volume"].append(volume)
        per_stock_sync["adjusted_close"].append(adjusted_close)
        per_stock_irregular["adjusts"].append(adjusts)
        per_stock_irregular["total_shares"].append(total_shares)
        per_stock_irregular["circ_shares"].append(circ_shares)
        per_stock_irregular["parent_equity"].append(parent_equity)
        per_stock_irregular["net_profit"].append(net_profit)

    return {
        **{k: sc.add_operator(StackSync(v)) for k, v in per_stock_sync.items()},
        **{k: sc.add_operator(Stack(v)) for k, v in per_stock_irregular.items()},
    }


# =====================================================================
# Cross-sectional feature set
# =====================================================================


def build_features(
    sc: Scenario,
    stacked: dict[str, Handle],
    *,
    num_stocks: int,
    window: int,
) -> tuple[dict[str, Handle], Handle]:
    """Build the canonical factor panel from a stacked per-stock dict.

    The dict contains:

    - ``rank_market_cap``, ``rank_bp``, ``rank_ttm_roe`` -- three
      percentile-ranked fundamentals (NaN-preserving).
    - ``momentum_ma_{window}`` -- rolling mean of daily log-returns of
      adjusted close.
    - ``volatility_{window}`` -- rolling std dev of daily log-returns.
    - ``turnover_ma_{window}`` -- rolling mean of daily turnover
      (volume / circulating shares).
    - ``volume_ratio_{window}`` -- latest volume / rolling-mean volume.

    The companion ``features_series`` handle is the trading-day-
    aligned ``Record`` of the ``Stack`` over the dict, ready to feed
    into a predictor or metric.  The ``Resample`` on the per-stock
    daily ``adjusted_close`` pulse re-emits the features on the
    trading-day cadence (instead of also ticking on irregular
    corporate-event days), so the recorded panel and any same-cadence
    target record stay lock-step.

    Parameters
    ----------
    sc
        Scenario under construction.
    stacked
        The cross-sectional dict produced by
        [`build_stacked`][common.build_stacked].
    num_stocks
        Cross-section size.
    window
        Rolling window length (in samples) used by the momentum,
        volatility, turnover, and volume rolling means.

    Returns
    -------
    features
        Cross-sectional features dict, ready to pass to :class:`Stack`.
    features_series
        Trading-day-aligned ``Record`` of the ``Stack`` over the
        features dict, element shape ``(num_stocks, n_features)``.
    """
    # Market cap, book-to-price, TTM ROE.
    market_cap = sc.add_operator(Multiply(stacked["close"], stacked["total_shares"]))
    bp = sc.add_operator(Divide(stacked["parent_equity"], market_cap))
    net_profit_series = sc.add_operator(Record(stacked["net_profit"]))
    net_profit_ttm = sc.add_operator(RollingMean(net_profit_series, window=np.timedelta64(365, "D")))
    ttm_roe = sc.add_operator(Divide(net_profit_ttm, stacked["parent_equity"]))

    # Momentum MA (rolling mean of daily log-returns of adjusted close).
    log_adj = sc.add_operator(Log(stacked["adjusted_close"]))
    log_adj_series = sc.add_operator(Record(log_adj))
    log_adj_lag = sc.add_operator(Lag(log_adj_series, offset=window, fill=np.float64(np.nan)))
    momentum_ma = sc.add_operator(Subtract(log_adj, log_adj_lag))

    # Price volatility (rolling std dev of daily log-returns of adjusted close).
    volatility = sc.add_operator(Sqrt(sc.add_operator(RollingVariance(log_adj_series, window=window))))

    # Turnover MA (rolling mean of daily turnover).
    turnover = sc.add_operator(Divide(stacked["volume"], stacked["circ_shares"]))
    turnover_series = sc.add_operator(Record(turnover))
    turnover_ma = sc.add_operator(RollingMean(turnover_series, window=window))

    # Volume ratio (latest volume / rolling-mean of volume over window).
    volume_series = sc.add_operator(Record(stacked["volume"]))
    volume_ma = sc.add_operator(RollingMean(volume_series, window=window))
    volume_ratio = sc.add_operator(Divide(stacked["volume"], volume_ma))

    # # Volume rally (volume_ratio gated on positive momentum).
    # zero_const = sc.add_const(np.zeros(num_stocks, dtype=np.float64))
    # positive_momentum_magnitude = sc.add_operator(Max(momentum_ma, zero_const))
    # volume_rally = sc.add_operator(Multiply(volume_ratio, positive_momentum_magnitude))

    features: dict[str, Handle] = {
        "rank_market_cap": sc.add_operator(Percentile(market_cap)),
        "rank_bp": sc.add_operator(Percentile(bp)),
        "rank_ttm_roe": sc.add_operator(Percentile(ttm_roe)),
        f"momentum_ma_{window}": momentum_ma,
        f"volatility_{window}": volatility,
        f"turnover_ma_{window}": turnover_ma,
        f"volume_ratio_{window}": volume_ratio,
    }

    stacked_features = sc.add_operator(Stack(list(features.values()), axis=1))
    sampled_features = sc.add_operator(Resample(stacked["adjusted_close"], stacked_features))
    features_series = sc.add_operator(Record(sampled_features))

    return features, features_series


# =====================================================================
# Scenario building blocks
# =====================================================================


def build_rebalance_clock(
    sc: Scenario,
    start: np.datetime64,
    end: np.datetime64,
    rebalance_days: int,
) -> tuple[Handle, np.ndarray]:
    """Build the periodic rebalance clock.

    Fires every ``rebalance_days`` calendar days from ``start``
    (inclusive) through ``end`` (inclusive).  Returns
    ``(clock_handle, rebalance_dates)`` so callers can also feed the
    dates into per-rebalance plots.
    """
    rebalance_dates = np.arange(
        start,
        end + np.timedelta64(1, "D"),
        np.timedelta64(rebalance_days, "D"),
    )
    clock = sc.add_source(Clock(rebalance_dates))
    return clock, rebalance_dates


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


def build_cap_weighted_universe(
    sc: Scenario,
    market_cap: Handle,
    rebalance_clock: Handle,
    *,
    num_stocks: int,
    index_size: int,
) -> Handle:
    """Cap-weighted top-``index_size`` universe, clocked on rebalance.

    Equivalent to
    ``Clocked(rebalance_clock, Map(market_cap, calculate_index_weights,
    ...))``.  The same handle plays double duty across the examples:
    rebalance signal for `MeanPortfolio` subclasses (and the mask
    input for `RankBucket`), and soft-position input to a
    `Benchmark` to track the cap-weighted index.
    """
    return sc.add_operator(
        Clocked(
            rebalance_clock,
            Map(
                market_cap,
                lambda m: calculate_index_weights(m, index_size),
                shape=(num_stocks,),
                dtype=np.float64,
            ),
        )
    )


def build_demeaned_log_return_target(
    sc: Scenario,
    log_adj: Handle,
    *,
    num_stocks: int,
    p: float = 0.01,
) -> tuple[Handle, Handle]:
    """Winsorize-then-demean log returns to use as a regression target.

    Returns ``(target, target_series)``: the live cross-sectional
    target handle and its `Record`.

    Order of operations:

    1. ``Diff(log_adj)`` to get raw daily log returns.
    2. ``Winsorize(p=p)`` clips outliers at the ``[p, 1-p]``-quantile
       range of the input cross-section.
    3. ``Apply(... lambda r, m: r - m)`` subtracts the per-day
       cross-sectional mean of the winsorized values, so the predictor
       learns the idiosyncratic component instead of the time-varying
       market drift.

    The full-position optimization objective is invariant under
    per-day additive shifts (sum(w) = 1 absorbs the constant in
    ``E[r_p]``, variance / covariance are shift-invariant), and rank-
    based portfolio constructors only see the ordering, so the
    resulting portfolios are unchanged.
    """
    log_returns = sc.add_operator(Diff(log_adj))
    winsorized_log_returns = sc.add_operator(Winsorize(log_returns, p=p))
    target = sc.add_operator(
        Apply(
            (winsorized_log_returns,),
            lambda r: r - np.nanmean(r),
            shape=(num_stocks,),
            dtype=np.float64,
        )
    )
    target_series = sc.add_operator(Record(target))
    return target, target_series


def build_price_limits(
    sc: Scenario,
    close: Handle,
    *,
    num_stocks: int,
    limit_pct: float = 0.10,
) -> tuple[Handle, Handle]:
    """Constant-percentage daily price limits, ready for the trader operators.

    Returns ``(upper_limit, lower_limit)`` -- two ``(num_stocks,)``
    handles that tick whenever ``close`` ticks.  At each tick the
    values are::

        upper_limit = round(prev_close * (1 + limit_pct), 2)
        lower_limit = round(prev_close * (1 - limit_pct), 2)

    where ``prev_close`` is the value of ``close`` from the previous
    tick.  Rounding to 0.01 yuan matches the A-shares tick size (and
    the SSE / SZSE official rule: limit = ``prev_close * (1 ± pct)``
    rounded half-up to one fen).  The first tick yields NaN limits
    (no prior close); downstream trader operators treat NaN as "no
    constraint" and let those trades through.

    Parameters
    ----------
    sc
        Scenario under construction.
    close
        Stacked unadjusted close prices, shape ``(num_stocks,)``
        (e.g., ``stacked["close"]`` from
        [`build_stacked`][common.build_stacked]).
    num_stocks
        Cross-section size.
    limit_pct
        Daily limit fraction.  ``0.10`` for SSE / SZSE main boards,
        ``0.20`` for ChiNext / STAR, ``0.30`` for BSE, ``0.05`` for
        ST / *ST stocks.  Constant across stocks; per-stock limits
        require a per-stock ``limit_pct`` handle instead, which this
        helper does not currently expose.

    Notes
    -----
    Python's ``np.round`` uses banker's rounding (round half to even)
    while the official A-shares rule is round-half-up; the two differ
    only when the unrounded limit lands exactly on a half-fen
    (e.g. ``11.055``), so backtests differ from the official print by
    at most 0.01 yuan in those edge cases -- negligible at typical
    return-distribution scales.
    """
    close_series = sc.add_operator(Record(close))
    prev_close = sc.add_operator(Lag(close_series, offset=1, fill=np.float64(np.nan)))
    upper = sc.add_operator(
        Apply((prev_close,), lambda c: np.round(c * (1.0 + limit_pct), 2), shape=(num_stocks,), dtype=np.float64)
    )
    lower = sc.add_operator(
        Apply((prev_close,), lambda c: np.round(c * (1.0 - limit_pct), 2), shape=(num_stocks,), dtype=np.float64)
    )
    return upper, lower


# =====================================================================
# Benchmark scaling for plots
# =====================================================================


def find_effective_trading_start(value: pd.Series, initial_cash: float = 1.0) -> pd.Timestamp | None:
    """First date on which a recorded NAV first deviates from ``initial_cash``.

    Every example wires :class:`Benchmark` with ``initial_cash=1.0`` so
    its output stays at ``1.0`` until upstream soft positions trigger
    the first non-trivial trade.  This helper returns the date of that
    first trade, used as the anchor for
    [`scale_to_initial_cash`][common.scale_to_initial_cash] so a unit-
    cash baseline can be lined up with a strategy NAV that starts at a
    larger nominal capital.  Returns ``None`` if `value` is constant at
    ``initial_cash`` across the whole series.
    """
    mask = ~np.isclose(value.to_numpy(), initial_cash)
    if not mask.any():
        return None
    return value.index[int(mask.argmax())]


def scale_to_initial_cash(value: pd.Series, initial_cash: float, anchor: pd.Timestamp | None) -> pd.Series:
    """Rescale a unit-cash NAV so that ``value.asof(anchor) == initial_cash``.

    Pair with [`find_effective_trading_start`][common.find_effective_trading_start]
    to align a unit-cash :class:`Benchmark` curve with a strategy NAV
    that starts at a larger ``initial_cash``: both curves then equal
    ``initial_cash`` on the strategy's effective trading start date.
    Falls back to plain ``value * initial_cash`` when ``anchor`` is
    ``None`` or no value at or before ``anchor`` is available.
    """
    if anchor is None:
        return value * initial_cash
    v = value.asof(anchor)
    if not np.isfinite(v) or v == 0:
        return value * initial_cash
    return value * (initial_cash / float(v))


# =====================================================================
# Progress reporting
# =====================================================================


def make_progress_tracker(
    sc: Scenario,
    mid: np.datetime64,
    *,
    before_desc: str = "Loading data",
    after_desc: str = "Running scenario",
) -> tuple[Callable[[int, int, int | None], None], tqdm]:
    """Build a ``tqdm`` progress bar and matching ``on_flush`` callback.

    The bar advances by ``estimated_event_count()`` total events; the
    callback switches its description from ``before_desc`` to
    ``after_desc`` once flush timestamps cross ``mid`` (typically the
    trading / evaluation start date), so the bar visually distinguishes
    the warmup load from the productive run.

    Returns ``(on_flush, progress)``.  Wire as::

        on_flush, progress = make_progress_tracker(sc, args.begin)
        session = sc.run(on_flush=on_flush)
        progress.close()
    """
    progress = tqdm(total=sc.estimated_event_count(), unit=" events", desc=before_desc)

    def on_flush(ts_ns: int, events: int, total: int | None) -> None:
        if np.datetime64(ts_ns, "ns") > mid:
            progress.set_description(after_desc)
        if total != progress.total:
            progress.total = total
        progress.update(events - progress.n)

    return on_flush, progress
