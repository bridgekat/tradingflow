"""Multi-factor stock selection backtest on A-shares.

Loads EastMoney CSV data (daily prices, balance sheets, income statements,
equity structure), computes 5 cross-sectional factors, uses rolling
cross-sectional linear regression to predict 1-month returns, constructs
portfolios from top-predicted stocks, and simulates portfolio value with
transaction costs.  Results are plotted as a yield curve.

Usage::

    python examples/factor_model_backtest.py [--symbols SYM1,SYM2,...] [--data-dir PATH]
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so imports work when running the script directly
# ---------------------------------------------------------------------------

from tradingflow import Series, Scenario, ArrayBundleSource
from tradingflow.sources.eastmoney.history import (
    DailyMarketSnapshotCSVSource,
    EquityStructureCSVSource,
    FinancialReportCSVSource,
    INCOME_STATEMENT_SCHEMA,
    BALANCE_SHEET_SCHEMA,
)
from tradingflow.ops import Apply, divide
from tradingflow.ops.filters import MovingAverage, MovingVariance
from tradingflow.ops.predictors import CrossSectionalRegression
from tradingflow.ops.portfolios import RandomTopK
from tradingflow.ops.simulators import TradingSimulator
from tradingflow.sources.eastmoney.history.financial_reports.schema import FinancialReportKind

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("data") / "a_shares_history_raw"
BACKTEST_START = np.datetime64("2005-01-01", "ns")
BACKTEST_END = np.datetime64("2024-12-31", "ns")
COMMISSION_RATE = 0.001  # 0.1%
INITIAL_CASH = 100_000.0
LOT_SIZE = 100
RETRAIN_EVERY = 21  # ~1 month in trading days
RETURN_HORIZON = 21
TRAIN_SAMPLES = 100_000
TOP_FRAC = 0.1
SELECT_K = 20
SEED = 42

# Field indices in the default daily market snapshot schema
IDX_CLOSE = 1  # "close"
IDX_AMOUNT = 4  # "amount"

# Field indices in the default schemas
BS_EQUITY_IDX = BALANCE_SHEET_SCHEMA.field_index["balance_sheet.equity"]
IS_PROFIT_IDX = INCOME_STATEMENT_SCHEMA.field_index["income_statement.profit"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def discover_symbols(data_dir: Path, symbols: list[str] | None = None) -> list[str]:
    """Return a validated, sorted list of stock symbols available in *data_dir*.

    If *symbols* is given, only those with all required CSV files are kept.
    Otherwise all symbols found via daily price CSVs are returned, filtered
    for the existence of other required files.
    """
    required_suffixes = [
        "_daily_price_raw.csv",
        "_balance_sheet_raw.csv",
        "_income_statement_raw.csv",
        "_equity_structure_raw.csv",
    ]

    if symbols is not None:
        candidates = list(symbols)
    else:
        candidates = sorted({f.name.split("_")[0] for f in data_dir.glob("*_daily_price_raw.csv")})

    valid: list[str] = []
    for sym in candidates:
        if all((data_dir / f"{sym}{suffix}").exists() for suffix in required_suffixes):
            valid.append(sym)
    return valid


async def _drain_source(source) -> tuple[list[np.datetime64], list[np.ndarray]]:
    """Iterate a historical-only source and collect timestamps + values."""
    hist_iter, _ = source.subscribe()
    timestamps: list[np.datetime64] = []
    values: list[np.ndarray] = []
    async for ts, val in hist_iter:
        timestamps.append(ts)
        values.append(np.asarray(val))
    return timestamps, values


async def load_daily_prices(
    data_dir: Path,
    symbols: list[str],
) -> dict[str, tuple[list[np.datetime64], list[np.ndarray]]]:
    """Load daily price data for each symbol using DailyMarketSnapshotCSVSource."""
    result: dict[str, tuple[list[np.datetime64], list[np.ndarray]]] = {}
    for sym in symbols:
        path = data_dir / f"{sym}_daily_price_raw.csv"
        source = DailyMarketSnapshotCSVSource(path, strict_row_checks=False, name=sym)
        result[sym] = await _drain_source(source)
    return result


async def load_financial_data(
    data_dir: Path,
    symbols: list[str],
    kind: FinancialReportKind,
    quarterly: bool = False,
) -> dict[str, tuple[list[np.datetime64], list[np.ndarray]]]:
    """Load financial reports for each symbol."""
    suffix = "_balance_sheet_raw.csv" if kind == "balance_sheet" else "_income_statement_raw.csv"
    result: dict[str, tuple[list[np.datetime64], list[np.ndarray]]] = {}
    for sym in symbols:
        path = data_dir / f"{sym}{suffix}"
        source = FinancialReportCSVSource(
            path,
            kind=kind,
            quarterly=quarterly,
            strict_equation_check=False,
            name=sym,
        )
        result[sym] = await _drain_source(source)
    return result


async def load_equity_structure(
    data_dir: Path,
    symbols: list[str],
) -> dict[str, tuple[list[np.datetime64], list[np.ndarray]]]:
    """Load equity structure (total shares) for each symbol."""
    result: dict[str, tuple[list[np.datetime64], list[np.ndarray]]] = {}
    for sym in symbols:
        path = data_dir / f"{sym}_equity_structure_raw.csv"
        source = EquityStructureCSVSource(path, strict_row_checks=False, name=sym)
        result[sym] = await _drain_source(source)
    return result


# ---------------------------------------------------------------------------
# Panel alignment
# ---------------------------------------------------------------------------


def _find_common_dates(
    price_data: dict[str, tuple[list[np.datetime64], list[np.ndarray]]],
    start: np.datetime64,
    end: np.datetime64,
) -> np.ndarray:
    """Compute the union of all trading dates across stocks within [start, end]."""
    all_dates: set[np.datetime64] = set()
    for timestamps, _ in price_data.values():
        for ts in timestamps:
            if start <= ts <= end:
                all_dates.add(ts)
    return np.sort(np.array(list(all_dates), dtype="datetime64[ns]"))


def align_scalar_panel(
    stock_data: dict[str, tuple[list[np.datetime64], list[np.ndarray]]],
    common_dates: np.ndarray,
    symbols: list[str],
    field_index: int | None = None,
) -> np.ndarray:
    """Align per-stock scalar time series onto common dates with forward-fill.

    If *field_index* is not None, extract that index from vector-valued entries.
    Returns shape ``(T, N)`` with NaN for missing data.
    """
    T = len(common_dates)
    N = len(symbols)
    panel = np.full((T, N), np.nan, dtype=np.float64)

    for col, sym in enumerate(symbols):
        if sym not in stock_data:
            continue
        timestamps, values = stock_data[sym]
        if not timestamps:
            continue

        # Build lookup: for each common date, find latest value at or before it
        ts_arr = np.array(timestamps, dtype="datetime64[ns]")
        val_arr = np.array([(float(v[field_index]) if field_index is not None else float(v)) for v in values])

        # Forward-fill via searchsorted
        indices = np.searchsorted(ts_arr, common_dates, side="right") - 1
        valid = indices >= 0
        panel[valid, col] = val_arr[indices[valid]]

    return panel


def compute_ttm_panel(
    quarterly_data: dict[str, tuple[list[np.datetime64], list[np.ndarray]]],
    common_dates: np.ndarray,
    symbols: list[str],
    field_index: int,
) -> np.ndarray:
    """Compute trailing-twelve-month sum from quarterly data, aligned to daily dates.

    For each stock, computes the rolling sum of the last 4 quarterly observations
    and forward-fills onto common_dates.
    """
    T = len(common_dates)
    N = len(symbols)
    panel = np.full((T, N), np.nan, dtype=np.float64)

    for col, sym in enumerate(symbols):
        if sym not in quarterly_data:
            continue
        timestamps, values = quarterly_data[sym]
        if len(timestamps) < 4:
            continue

        ts_arr = np.array(timestamps, dtype="datetime64[ns]")
        val_arr = np.array([float(v[field_index]) for v in values])

        # Replace NaN with 0 for rolling sum, track NaN positions
        val_clean = np.where(np.isnan(val_arr), 0.0, val_arr)

        # Compute rolling sum of last 4 quarters
        ttm_ts: list[np.datetime64] = []
        ttm_vals: list[float] = []
        for i in range(3, len(val_clean)):
            ttm_ts.append(ts_arr[i])
            ttm_vals.append(float(np.sum(val_clean[i - 3 : i + 1])))

        if not ttm_ts:
            continue

        ttm_ts_arr = np.array(ttm_ts, dtype="datetime64[ns]")
        ttm_val_arr = np.array(ttm_vals)

        # Forward-fill onto common dates
        indices = np.searchsorted(ttm_ts_arr, common_dates, side="right") - 1
        valid = indices >= 0
        panel[valid, col] = ttm_val_arr[indices[valid]]

    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_backtest(
    data_dir: Path,
    symbols: list[str] | None = None,
    max_symbols: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full factor-model backtest and return (dates, portfolio_values)."""

    # 1. Discover symbols
    print("Discovering symbols...")
    all_symbols = discover_symbols(data_dir, symbols)
    if max_symbols is not None:
        all_symbols = all_symbols[:max_symbols]
    N = len(all_symbols)
    print(f"  Using {N} stocks")
    if N < SELECT_K:
        print(f"  Warning: fewer stocks ({N}) than SELECT_K ({SELECT_K})")

    # 2. Load data
    print("Loading daily prices...")
    price_data = await load_daily_prices(data_dir, all_symbols)

    print("Loading balance sheets...")
    bs_data = await load_financial_data(data_dir, all_symbols, "balance_sheet")

    print("Loading income statements (quarterly)...")
    is_data = await load_financial_data(
        data_dir,
        all_symbols,
        "income_statement",
        quarterly=True,
    )

    print("Loading equity structure...")
    eq_data = await load_equity_structure(data_dir, all_symbols)

    # 3. Build common date grid
    print("Aligning panel data...")
    common_dates = _find_common_dates(price_data, BACKTEST_START, BACKTEST_END)
    T = len(common_dates)
    print(f"  {T} trading days from {common_dates[0]} to {common_dates[-1]}")

    # 4. Build panels — shape (T, N)
    close_panel = align_scalar_panel(price_data, common_dates, all_symbols, field_index=IDX_CLOSE)
    amount_panel = align_scalar_panel(price_data, common_dates, all_symbols, field_index=IDX_AMOUNT)
    total_shares_panel = align_scalar_panel(eq_data, common_dates, all_symbols)
    equity_panel = align_scalar_panel(bs_data, common_dates, all_symbols, field_index=BS_EQUITY_IDX)
    profit_ttm_panel = compute_ttm_panel(is_data, common_dates, all_symbols, field_index=IS_PROFIT_IDX)

    # Derived panels
    market_cap_panel = close_panel * total_shares_panel

    print(f"  Panel shape: ({T}, {N})")
    print(
        f"  NaN rates: close={np.isnan(close_panel).mean():.1%}, "
        f"mcap={np.isnan(market_cap_panel).mean():.1%}, "
        f"equity={np.isnan(equity_panel).mean():.1%}, "
        f"profit_ttm={np.isnan(profit_ttm_panel).mean():.1%}"
    )

    # 5. Build Scenario pipeline
    print("Building scenario pipeline...")
    scenario = Scenario()

    close_s = scenario.add_source(ArrayBundleSource(common_dates, close_panel))
    mcap_s = scenario.add_source(ArrayBundleSource(common_dates, market_cap_panel))
    equity_s = scenario.add_source(ArrayBundleSource(common_dates, equity_panel))
    profit_s = scenario.add_source(ArrayBundleSource(common_dates, profit_ttm_panel))
    amount_s = scenario.add_source(ArrayBundleSource(common_dates, amount_panel))

    # Factor 1: log(market cap)
    log_mcap_s = scenario.add_operator(
        Apply(
            (mcap_s,),
            (N,),
            np.float64,
            lambda args: np.log(np.where(args[0] > 0, args[0], np.nan)),
        )
    )

    # Factor 2: log(book-to-price) = log(equity / market_cap)
    log_btp_s = scenario.add_operator(
        Apply(
            (equity_s, mcap_s),
            (N,),
            np.float64,
            lambda args: np.log(
                np.where(
                    (args[0] > 0) & (args[1] > 0),
                    args[0] / args[1],
                    np.nan,
                )
            ),
        )
    )

    # Factor 3: earnings-to-price = profit_ttm / market_cap
    ep_s = scenario.add_operator(
        Apply(
            (profit_s, mcap_s),
            (N,),
            np.float64,
            lambda args: np.where(args[1] > 0, args[0] / args[1], np.nan),
        )
    )

    # Factor 4: monthly turnover = MA(21, amount / market_cap)
    daily_turnover_s = scenario.add_operator(
        Apply(
            (amount_s, mcap_s),
            (N,),
            np.float64,
            lambda args: np.where(args[1] > 0, args[0] / args[1], np.nan),
        )
    )
    monthly_turnover_s = scenario.add_operator(MovingAverage(RETRAIN_EVERY, daily_turnover_s))

    # Factor 5: monthly volatility = sqrt(Var(21, close))
    monthly_var_s = scenario.add_operator(MovingVariance(RETRAIN_EVERY, close_s))
    monthly_vol_s = scenario.add_operator(
        Apply(
            (monthly_var_s,),
            (N,),
            np.float64,
            lambda args: np.sqrt(np.maximum(args[0], 0.0)),
        )
    )

    # Stack factors into (N, 5) feature matrix
    factors_s = scenario.add_operator(
        Apply(
            (log_mcap_s, log_btp_s, ep_s, monthly_turnover_s, monthly_vol_s),
            (N, 5),
            np.float64,
            lambda args: np.column_stack(args),
        )
    )

    # Cross-sectional regression
    train_window = max(1, TRAIN_SAMPLES // N) + RETURN_HORIZON
    predictions_s = scenario.add_operator(
        CrossSectionalRegression(
            factors_s,
            close_s,
            train_window=train_window,
            retrain_every=RETRAIN_EVERY,
            return_horizon=RETURN_HORIZON,
        )
    )

    # Portfolio construction (random 20 from top 10%)
    select_k = min(SELECT_K, N)
    weights_s = scenario.add_operator(
        RandomTopK(
            predictions_s,
            top_frac=TOP_FRAC,
            select_k=select_k,
            seed=SEED,
        )
    )

    # Portfolio simulation
    portfolio_value_s = scenario.add_operator(
        TradingSimulator(
            close_s,
            weights_s,
            commission_rate=COMMISSION_RATE,
            initial_cash=INITIAL_CASH,
            weight_mode=True,
            lot_size=LOT_SIZE,
        )
    )

    # 6. Run
    print("Running scenario...")

    # Find the series that holds portfolio values
    portfolio_series = portfolio_value_s

    await scenario.run()

    # Extract results from the last registered operator's series
    # The scenario.operators list contains (operator, series) pairs
    op_series_list = list(scenario.operators)
    portfolio_series = op_series_list[-1][1]  # last operator = TradingSimulator

    result_dates = portfolio_series.index[: len(portfolio_series)]
    result_values = portfolio_series.values[: len(portfolio_series)]

    print(f"  Portfolio series length: {len(portfolio_series)}")
    if len(portfolio_series) > 0:
        print(f"  Final portfolio value: {float(result_values[-1]):,.2f}")

    return result_dates, result_values


def plot_results(dates: np.ndarray, values: np.ndarray, save_path: Path | None = None) -> None:
    """Plot portfolio value over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    fig, ax = plt.subplots(figsize=(14, 6))

    # Convert datetime64[ns] to Python datetime for matplotlib
    py_dates = [ts.astype("datetime64[ms]").astype(datetime) for ts in dates]

    ax.plot(py_dates, values, linewidth=0.8, color="#2563eb")
    ax.set_title("Factor Model Backtest \u2014 A-Shares", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (CNY)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = save_path or Path("factor_model_backtest.png")
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Factor model backtest on A-shares")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to raw CSV data directory",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of stock symbols (e.g. 000001,000002)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to use",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot image",
    )
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None

    dates, values = asyncio.run(
        run_backtest(
            data_dir=args.data_dir,
            symbols=symbols,
            max_symbols=args.max_symbols,
        )
    )

    if len(dates) == 0:
        print("No portfolio data produced. Check data availability.")
        return

    plot_results(dates, values, save_path=args.output)


if __name__ == "__main__":
    main()
