"""Multi-factor stock selection backtest on A-shares.

Loads EastMoney CSV data (daily prices, balance sheets, income statements,
equity structure), computes 5 cross-sectional factors, uses rolling
cross-sectional linear regression to predict 1-month returns, constructs
portfolios from top-predicted stocks, and simulates portfolio value with
transaction costs.  Results are plotted as a yield curve.

Per-stock sources are registered directly in the Scenario.  The POCQ handles
heterogeneous per-stock timestamps naturally; :class:`Stack` assembles
cross-sectional vectors with forward-fill via ``Series.at()``.

Usage::

    python factor_model_backtest.py [--symbols SYM1,SYM2,...] [--data-dir PATH]
"""

from __future__ import annotations

import asyncio
import argparse
from pathlib import Path
from typing import Any

import numpy as np

from tradingflow import Scenario, Series
from tradingflow.sources.eastmoney.history import (
    DailyMarketSnapshotCSVSource,
    EquityStructureCSVSource,
    FinancialReportCSVSource,
    INCOME_STATEMENT_SCHEMA,
    BALANCE_SHEET_SCHEMA,
)
from tradingflow.operators import Select, Stack, divide, map, multiply
from tradingflow.operators.indicators import MovingAverage, MovingVariance
from tradingflow.operators.predictors import CrossSectionalRegression
from tradingflow.operators.portfolios import RandomTopK
from tradingflow.operators.simulators import TradingSimulator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("data") / "a_shares_history_raw"
COMMISSION_RATE = 0.001  # 0.1%
INITIAL_CASH = 100000.0
LOT_SIZE = 100
RETRAIN_EVERY = 21  # ~1 month in trading days
RETURN_HORIZON = 21
TRAIN_SAMPLES = 100000
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
# Helpers
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_backtest(
    data_dir: Path,
    symbols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full factor-model backtest and return (dates, portfolio_values)."""

    # 1. Discover symbols
    print("Discovering symbols...")
    all_symbols = discover_symbols(data_dir, symbols)
    N = len(all_symbols)
    print(f"  Using {N} stocks")
    if N < SELECT_K:
        print(f"  Warning: fewer stocks ({N}) than SELECT_K ({SELECT_K})")

    # 2. Register per-stock sources
    print("Registering per-stock sources...")
    scenario = Scenario()

    price_series: list[Series[Any, Any]] = []
    eq_series: list[Series[Any, Any]] = []
    bs_series: list[Series[Any, Any]] = []
    is_series: list[Series[Any, Any]] = []

    for sym in all_symbols:
        price_series.append(
            scenario.add_source(
                DailyMarketSnapshotCSVSource(
                    data_dir / f"{sym}_daily_price_raw.csv",
                    strict_row_checks=False,
                    name=f"{sym}/price",
                )
            )
        )
        eq_series.append(
            scenario.add_source(
                EquityStructureCSVSource(
                    data_dir / f"{sym}_equity_structure_raw.csv",
                    strict_row_checks=False,
                    name=f"{sym}/eq",
                )
            )
        )
        bs_series.append(
            scenario.add_source(
                FinancialReportCSVSource(
                    data_dir / f"{sym}_balance_sheet_raw.csv",
                    kind="balance_sheet",
                    strict_equation_check=False,
                    name=f"{sym}/bs",
                )
            )
        )
        is_series.append(
            scenario.add_source(
                FinancialReportCSVSource(
                    data_dir / f"{sym}_income_statement_raw.csv",
                    kind="income_statement",
                    annualize=True,
                    strict_equation_check=False,
                    name=f"{sym}/is",
                )
            )
        )

    print(f"  {4 * N} sources registered")

    # 3. Cross-sectional assembly via Stack
    #    Stack assembles per-stock series into (N, ...) vectors/matrices.

    # Daily prices: Stack (6,) vectors into (N, 6), then extract columns
    price_matrix_s = scenario.add_operator(Stack(price_series))  # (N, 6)
    close_s = scenario.add_operator(Select(price_matrix_s, IDX_CLOSE))  # (N,)
    amount_s = scenario.add_operator(Select(price_matrix_s, IDX_AMOUNT))  # (N,)

    # Equity structure: scalar sources -> Stack directly into (N,)
    total_shares_s = scenario.add_operator(Stack(eq_series))  # (N,)

    # Balance sheets: Stack (K,) vectors into (N, K), then extract equity column
    bs_matrix_s = scenario.add_operator(Stack(bs_series))  # (N, K_bs)
    equity_s = scenario.add_operator(Select(bs_matrix_s, BS_EQUITY_IDX))  # (N,)

    # Income statements (annualized): per-stock field extraction + MA(4) for TTM, then Stack
    profit_ttm_series: list[Series[Any, Any]] = []
    for s in is_series:
        profit_field_s = scenario.add_operator(Select(s, IS_PROFIT_IDX))  # ()
        ttm_s = scenario.add_operator(MovingAverage(4, profit_field_s))
        profit_ttm_series.append(ttm_s)
    profit_s = scenario.add_operator(Stack(profit_ttm_series))  # (N,)

    # 4. Derived cross-sectional series
    mcap_s = scenario.add_operator(multiply(close_s, total_shares_s))

    # 5. Factor pipeline

    # Factor 1: log(market cap)
    log_mcap_s = scenario.add_operator(map(mcap_s, np.log))

    # Factor 2: log(book-to-price) = log(equity / market_cap)
    btp_s = scenario.add_operator(divide(equity_s, mcap_s))
    log_btp_s = scenario.add_operator(map(btp_s, np.log))

    # Factor 3: earnings-to-price = profit_ttm / market_cap
    ep_s = scenario.add_operator(divide(profit_s, mcap_s))

    # Factor 4: monthly turnover = MA(21, amount / market_cap)
    daily_turnover_s = scenario.add_operator(divide(amount_s, mcap_s))
    monthly_turnover_s = scenario.add_operator(MovingAverage(RETRAIN_EVERY, daily_turnover_s))

    # Factor 5: monthly volatility = sqrt(Var(21, close))
    monthly_var_s = scenario.add_operator(MovingVariance(RETRAIN_EVERY, close_s))
    monthly_vol_s = scenario.add_operator(map(monthly_var_s, lambda x: np.sqrt(np.maximum(x, 0.0))))

    # Stack factors into (N, 5) feature matrix
    factors_s = scenario.add_operator(Stack([log_mcap_s, log_btp_s, ep_s, monthly_turnover_s, monthly_vol_s], axis=1))

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
    await scenario.run()

    result_dates = portfolio_value_s.index
    result_values = portfolio_value_s.values

    print(f"  Portfolio series length: {len(portfolio_value_s)}")
    if len(portfolio_value_s) > 0:
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
        )
    )

    if len(dates) == 0:
        print("No portfolio data produced. Check data availability.")
        return

    plot_results(dates, values, save_path=args.output)


if __name__ == "__main__":
    main()
