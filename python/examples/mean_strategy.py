"""Backtest a linear regression strategy on all A-shares stocks.

For each stock the following features are computed every trading day:

- ``log(market_cap)`` — log of circulating market capitalisation
- ``log(B/P)`` — log of book-to-price ratio (parent equity / market cap)
- ``turnover_ma`` — simple moving average of daily turnover over the rebalance
  period (volume / circulating shares)

The pipeline consists of four independent, composable operators:

1. **MeanPredictor** — periodically fits a model and predicts future returns.
   Subclass: ``LinearRegression`` (pooled OLS regression).
2. **MeanPortfolio** — converts predicted returns into soft positions.
   Subclass: ``RankLinear`` (rank-linear top-fraction selection).
3. **RandomTrader** — converts soft positions into actual trades
   (lots of 100 shares), deducts transaction fees, and tracks the portfolio.
4. **Metric computation** — post-hoc analysis of the portfolio value series.

Requires ``pip install -e ".[examples]"`` and A-shares market data downloaded
via the crawler.  See ``python -m a_shares_crawler --help`` for configuration
and download instructions.
"""

from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow.types import Handle
from tradingflow.sources import CSVSource
from tradingflow.sources.stocks import FinancialReportSource
from tradingflow.operators import Map, Record, Select, Stack, Last, Lag
from tradingflow.operators.num import Subtract, Divide, Log, Multiply
from tradingflow.operators.predictors.mean import LinearRegression
from tradingflow.operators.portfolios.mean import RankLinear
from tradingflow.operators.traders.simple import RandomTrader
from tradingflow.operators.metrics import CompoundReturn, SharpeRatio, Drawdown
from tradingflow.operators.num import ForwardFill
from tradingflow.operators.rolling import RollingMean
from tradingflow.operators.stocks import Annualize, ForwardAdjust
from tradingflow.sources import MonthlyClock

DAY_NS = 86_400_000_000_000
PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())
DIVIDEND_SCHEMA = Schema(CSVSchema.dividends().iter_field_ids())
BS_SCHEMA = Schema(CSVSchema.balance_sheet().iter_field_ids())
INC_SCHEMA = Schema(CSVSchema.income_statement().iter_field_ids())

OHLCV_INDICES = PRICE_SCHEMA.indices(
    [
        "prices.open",
        "prices.high",
        "prices.low",
        "prices.close",
        "prices.volume",
    ]
)


def load_symbols(data_dir: Path) -> list[str]:
    """Read stock symbols from the symbol list CSV."""

    symbol_list_path = data_dir / "symbol_list.csv"
    if not symbol_list_path.exists():
        raise SystemExit(f"Symbol list not found: {symbol_list_path}")

    with symbol_list_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        i = header.index("symbol")
        symbols = [row[i] for row in reader]

    return symbols


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    rebalance_days: int,
    initial_cash: float,
    start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict]:
    """Build the full backtesting scenario."""

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    # Per-stock raw handles collected for cross-sectional stacking.
    per_stock: dict[str, list[Handle]] = {
        k: []
        for k in (
            "ohlcv",
            "adjusts",
            "adjusted_close",
            "circ_shares",
            "parent_equity",
            "net_profit",
        )
    }

    for symbol in tqdm(symbols, desc="Building scenario"):
        h = history_dir / symbol

        # ------------------------------------------------------------------
        # Sources
        # ------------------------------------------------------------------

        prices = sc.add_source(
            CSVSource(f"{h}.daily_prices.csv", PRICE_SCHEMA, time_column="date", start=start, end=end)
        )
        equity = sc.add_source(
            CSVSource(f"{h}.equity_structures.csv", EQUITY_SCHEMA, time_column="date", start=start, end=end)
        )
        dividends = sc.add_source(
            CSVSource(f"{h}.dividends.csv", DIVIDEND_SCHEMA, time_column="date", start=start, end=end)
        )
        balance = sc.add_source(
            FinancialReportSource(
                f"{h}.balance_sheets.csv",
                BS_SCHEMA,
                report_date_column="date",
                notice_date_column="notice_date",
                use_effective_date=False,
                start=start,
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
                start=start,
                end=end,
            )
        )

        # ------------------------------------------------------------------
        # Operators
        # ------------------------------------------------------------------

        ohlcv = sc.add_operator(Select(prices, OHLCV_INDICES))
        close = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.close")))
        adjusts = sc.add_operator(ForwardAdjust(close, dividends, output_prices=False))
        adjusted_close = sc.add_operator(Multiply(close, adjusts))
        circ_shares = sc.add_operator(Select(equity, EQUITY_SCHEMA.index("shares.circulating")))
        income_ann = sc.add_operator(Annualize(income_ytd))
        net_profit = sc.add_operator(Select(income_ann, INC_SCHEMA.index("income_statement.profit")))
        neg_parent_equity_components = sc.add_operator(
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
        parent_equity = sc.add_operator(
            Map(neg_parent_equity_components, lambda x: -x.sum(), shape=(), dtype=np.float64)
        )

        per_stock["ohlcv"].append(ohlcv)
        per_stock["adjusts"].append(adjusts)
        per_stock["adjusted_close"].append(adjusted_close)
        per_stock["circ_shares"].append(circ_shares)
        per_stock["parent_equity"].append(parent_equity)
        per_stock["net_profit"].append(net_profit)

    # ------------------------------------------------------------------
    # Cross-sectional operators
    # ------------------------------------------------------------------

    # Stack per-stock handles into (num_stocks, ...) arrays, with forward-fill to handle missing data.
    stacked = {k: sc.add_operator(ForwardFill(sc.add_operator(Stack(v)))) for k, v in per_stock.items()}

    # Extract close and volume from stacked OHLCV for feature computation.
    close = sc.add_operator(Select(stacked["ohlcv"], 3, axis=1))
    volume = sc.add_operator(Select(stacked["ohlcv"], 4, axis=1))

    # Market cap and log(market cap).
    market_cap = sc.add_operator(Multiply(close, stacked["circ_shares"]))
    log_mcap = sc.add_operator(Log(market_cap))

    # log(B/P) = log(parent_equity / market_cap).
    bp = sc.add_operator(Divide(stacked["parent_equity"], market_cap))
    log_bp = sc.add_operator(Log(bp))

    # # TTM net profit via 365-day rolling mean of annualized net profit.
    # net_profit_series = sc.add_operator(Record(stacked["net_profit"]))
    # net_profit_ttm = sc.add_operator(RollingMean(net_profit_series, window=np.timedelta64(365, "D")))

    # # TTM E/P and TTM ROE.
    # ttm_ep = sc.add_operator(Divide(net_profit_ttm, market_cap))
    # ttm_roe = sc.add_operator(Divide(net_profit_ttm, stacked["parent_equity"]))

    # # Momentum MA (-day MA of daily log-returns of adjusted_close).
    # log_adj = sc.add_operator(Log(stacked["adjusted_close"]))
    # log_adj_series = sc.add_operator(Record(log_adj))
    # log_adj_lag = sc.add_operator(Last(sc.add_operator(Lag(log_adj_series, 1, fill=np.float64(np.nan)))))
    # daily_ret = sc.add_operator(Subtract(log_adj, log_adj_lag))
    # daily_ret_series = sc.add_operator(Record(daily_ret))
    # momentum_ma = sc.add_operator(RollingMean(daily_ret_series, window=rebalance_days))

    # Turnover MA.
    turnover = sc.add_operator(Divide(volume, stacked["circ_shares"]))
    turnover_series = sc.add_operator(Record(turnover))
    turnover_ma = sc.add_operator(RollingMean(turnover_series, window=rebalance_days))

    # Stack features into (num_stocks, num_features).
    stacked_features = sc.add_operator(Stack([log_mcap, log_bp, turnover_ma], axis=1))

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    predicted = sc.add_operator(
        LinearRegression(
            stacked_features,
            stacked["adjusted_close"],
            rebalance_period=rebalance_days,
            max_samples=100000,
        )
    )
    positions = sc.add_operator(RankLinear(predicted, top_fraction=0.1))
    portfolio_value = sc.add_operator(
        RandomTrader(
            positions,
            stacked["ohlcv"],
            stacked["adjusts"],
            portfolio_size=20,
            initial_cash=initial_cash,
            lot_size=100.0,
            fee_base=5.0,
            fee_rate=0.001,
        )
    )

    # Total portfolio value = holdings + cash.
    total_value = sc.add_operator(Map(portfolio_value, np.sum, shape=(), dtype=np.float64))

    # ------------------------------------------------------------------
    # Metrics (clock-driven, since inception)
    # ------------------------------------------------------------------

    monthly_clock = sc.add_source(MonthlyClock(start, end, tz="Asia/Shanghai"))
    sharpe = sc.add_operator(SharpeRatio(total_value), clock=monthly_clock)
    compound_ret = sc.add_operator(CompoundReturn(total_value), clock=monthly_clock)
    drawdown = sc.add_operator(Drawdown(total_value))  # Triggers on every update

    return sc, {
        "trader": sc.add_operator(Record(portfolio_value)),
        "total_value": sc.add_operator(Record(total_value)),
        "sharpe": sc.add_operator(Record(sharpe)),
        "compound_return": sc.add_operator(Record(compound_ret)),
        "drawdown": sc.add_operator(Record(drawdown)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument("-b", "--begin", type=np.datetime64, required=True, help="start date (e.g. 2020-01-01)")
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    parser.add_argument("--rebalance-days", type=int, default=20, help="rebalance every N trading days")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="starting capital (CNY)")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    symbols = load_symbols(data_dir)
    print(f"Discovered {len(symbols)} symbols.")

    sc, handles = build_scenario(
        symbols,
        data_dir,
        rebalance_days=args.rebalance_days,
        initial_cash=args.initial_cash,
        start=args.begin,
        end=args.end,
    )

    first_ns_opt, last_ns_opt = sc.time_range()
    assert first_ns_opt is not None and last_ns_opt is not None, "all sources must provide a time range"
    first_ns, last_ns = first_ns_opt, last_ns_opt

    total_days = (last_ns - first_ns) // DAY_NS
    progress = tqdm(total=total_days, unit="d", desc="Running scenario")
    sc.run(on_flush=lambda ts: progress.update((min(max(ts, first_ns), last_ns) - first_ns) // DAY_NS - progress.n))
    progress.close()

    # Extract results.
    trader_df = sc.series_view(handles["trader"]).to_dataframe(["holdings_value", "cash"])
    total_df = sc.series_view(handles["total_value"]).to_dataframe(["total_value"])
    sharpe_df = sc.series_view(handles["sharpe"]).to_dataframe(["sharpe"])
    compound_ret_df = sc.series_view(handles["compound_return"]).to_dataframe(["compound_return"])
    drawdown_df = sc.series_view(handles["drawdown"]).to_dataframe(["drawdown"])

    n = len(total_df)
    if n == 0:
        raise SystemExit("No data produced.")

    total_value = total_df["total_value"]

    print(f"{n} trading days, {total_df.index[0].date()} to {total_df.index[-1].date()}")
    print(f"Initial value: {total_value.iloc[0]:,.2f} CNY")
    print(f"Final value:   {total_value.iloc[-1]:,.2f} CNY")
    print()

    if len(compound_ret_df) > 0:
        compound_return_annualized = (compound_ret_df["compound_return"].iloc[-1] + 1) ** 12 - 1
        print(f"Compound return annualized: {compound_return_annualized:.2%}")
    if len(sharpe_df) > 0:
        monthly_sharpe_annualized = sharpe_df["sharpe"].iloc[-1] * np.sqrt(12)
        print(f"Monthly Sharpe ratio annualized: {monthly_sharpe_annualized:.4f}")
    if len(drawdown_df) > 0:
        max_drawdown = drawdown_df["drawdown"].min()
        print(f"Max drawdown: {max_drawdown:.2%}")
    print()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    ax = axes[0]
    ax.set_title(f"Portfolio value ({len(symbols)} stocks, rebalance every {args.rebalance_days}d)")
    ax.set_ylabel("CNY (10k)")
    ax.axhline(args.initial_cash / 1e4, color="grey", linewidth=0.5, linestyle="--", label="Initial")
    ax.plot(total_df.index, total_value / 1e4, linewidth=0.8, label="Total value")
    ax.plot(trader_df.index, trader_df["cash"] / 1e4, linewidth=0.8, color="C2", alpha=0.7, label="Excess liquidity")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.set_title("Monthly Sharpe ratio annualized (since inception)")
    ax.set_ylabel("Sharpe ratio")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.plot(sharpe_df.index, sharpe_df["sharpe"] * np.sqrt(12), linewidth=0.8, color="C1")

    ax = axes[2]
    ax.set_title("Drawdown (since previous high)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.fill_between(drawdown_df.index, drawdown_df["drawdown"] * 100, 0, alpha=0.4, color="C3", linewidth=0)

    fig.tight_layout()
    plt.show()
