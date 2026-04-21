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

from pathlib import Path
from tqdm import tqdm
import argparse

import numpy as np
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow import Handle
from tradingflow.sources import CSVSource
from tradingflow.sources.stocks import FinancialReportSource
from tradingflow.operators import Clocked, Map, Record, Select, Stack, StackSync
from tradingflow.operators.num import Divide, Log, Multiply
from tradingflow.operators.predictors.mean import LinearRegression
from tradingflow.operators.portfolios.mean import RankLinear
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.traders.simple import RandomTrader
from tradingflow.operators.metrics import CompoundReturn, SharpeRatio, Drawdown
from tradingflow.operators.rolling import RollingMean
from tradingflow.operators.stocks import Annualize, ForwardAdjust
from tradingflow.sources import Clock

from stocks import load_symbols, calculate_index_weights, resolve_data_start, add_market_argument


DAY_NS = 86_400_000_000_000
PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())
DIVIDEND_SCHEMA = Schema(CSVSchema.dividends().iter_field_ids())
BS_SCHEMA = Schema(CSVSchema.balance_sheet().iter_field_ids())
INC_SCHEMA = Schema(CSVSchema.income_statement().iter_field_ids())
OHLCV_INDICES = PRICE_SCHEMA.indices(["prices.open", "prices.high", "prices.low", "prices.close", "prices.volume"])


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    rebalance_days: int,
    initial_cash: float,
    index_size: int,
    data_start: np.datetime64,
    trading_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, np.ndarray]:
    """Build the full backtesting scenario."""

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    # Per-stock raw handles grouped by cadence.
    #
    # * `per_stock_sync` — values produced in lockstep across all stocks
    #   (e.g., daily prices/equity on trading days).  Stacked with
    #   `StackSync` to give message-passing semantics: slots of stocks
    #   that did not produce this cycle are filled with `NaN`.
    # * `per_stock_irregular` — values updated on stock-specific dates
    #   (e.g., quarterly financial reports filed on different dates).
    #   Stacked with `Stack` to give time-series semantics: slots keep
    #   their last-known value across quiet periods.
    per_stock_sync: dict[str, list[Handle]] = {
        k: []
        for k in (
            "ohlcv",
            "adjusted_close",
        )
    }
    per_stock_irregular: dict[str, list[Handle]] = {
        k: []
        for k in (
            "adjusts",
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

        per_stock_sync["ohlcv"].append(ohlcv)
        per_stock_sync["adjusted_close"].append(adjusted_close)
        per_stock_irregular["adjusts"].append(adjusts)
        per_stock_irregular["circ_shares"].append(circ_shares)
        per_stock_irregular["parent_equity"].append(parent_equity)
        per_stock_irregular["net_profit"].append(net_profit)

    # ------------------------------------------------------------------
    # Cross-sectional operators
    # ------------------------------------------------------------------

    num_stocks = len(symbols)

    # Stack per-stock handles into (num_stocks, ...) arrays.
    stacked = {
        **{k: sc.add_operator(StackSync(v)) for k, v in per_stock_sync.items()},
        **{k: sc.add_operator(Stack(v)) for k, v in per_stock_irregular.items()},
    }

    # Extract close and volume from stacked OHLCV for feature computation.
    close = sc.add_operator(Select(stacked["ohlcv"], 3, axis=1))
    volume = sc.add_operator(Select(stacked["ohlcv"], 4, axis=1))

    # Market cap and log(market cap).
    market_cap = sc.add_operator(Multiply(close, stacked["circ_shares"]))
    log_mcap = sc.add_operator(Log(market_cap))

    # log(B/P) = log(parent_equity / market_cap).
    bp = sc.add_operator(Divide(stacked["parent_equity"], market_cap))
    log_bp = sc.add_operator(Log(bp))

    # Turnover MA.
    turnover = sc.add_operator(Divide(volume, stacked["circ_shares"]))
    turnover_series = sc.add_operator(Record(turnover))
    turnover_ma = sc.add_operator(RollingMean(turnover_series, window=rebalance_days))

    # Stack features into (num_stocks, num_features).
    stacked_features = sc.add_operator(Stack([log_mcap, log_bp, turnover_ma], axis=1))

    # Record feature and price history for predictors.
    features_series = sc.add_operator(Record(stacked_features))
    adjusted_prices_series = sc.add_operator(Record(stacked["adjusted_close"]))

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    # Rebalance clock: the single periodic signal in this scenario.
    rebalance_dates = np.arange(
        trading_start,
        end + np.timedelta64(1, "D"),
        np.timedelta64(rebalance_days, "D"),
    )
    rebalance_clock = sc.add_source(Clock(rebalance_dates))

    universe = sc.add_operator(
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

    predicted_returns = sc.add_operator(
        LinearRegression(
            universe,
            features_series,
            adjusted_prices_series,
            universe_size=index_size,
            min_periods=100,
            verbose=True,
        ),
    )

    soft_positions = sc.add_operator(
        RankLinear(
            universe,
            predicted_returns,
            top_fraction=0.1,
        )
    )

    index = sc.add_operator(
        Benchmark(
            universe,
            stacked["ohlcv"],
            stacked["adjusts"],
            initial_cash=initial_cash,
            use_adjusts=True,
        )
    )

    strategy_frictionless = sc.add_operator(
        Benchmark(
            soft_positions,
            stacked["ohlcv"],
            stacked["adjusts"],
            initial_cash=initial_cash,
            use_adjusts=True,
        )
    )

    strategy_actual = sc.add_operator(
        RandomTrader(
            soft_positions,
            stacked["ohlcv"],
            stacked["adjusts"],
            portfolio_size=20,
            initial_cash=initial_cash,
            lot_size=100.0,
            fee_base=5.0,
            fee_rate=0.001,
        )
    )

    # ------------------------------------------------------------------
    # Metrics (clock-driven, since inception)
    # ------------------------------------------------------------------

    actual_value = sc.add_operator(Map(strategy_actual, np.sum, shape=(), dtype=np.float64))
    sharpe = sc.add_operator(SharpeRatio(actual_value, rebalance_clock))
    compound_ret = sc.add_operator(CompoundReturn(actual_value, rebalance_clock))
    drawdown = sc.add_operator(Drawdown(actual_value))  # Triggers on every update

    return (
        sc,
        {
            "index": sc.add_operator(Record(index)),
            "strategy_frictionless": sc.add_operator(Record(strategy_frictionless)),
            "strategy_actual": sc.add_operator(Record(strategy_actual)),
            "sharpe": sc.add_operator(Record(sharpe)),
            "compound_return": sc.add_operator(Record(compound_ret)),
            "drawdown": sc.add_operator(Record(drawdown)),
        },
        rebalance_dates,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument("--sample-begin", type=np.datetime64, default=None, help="data sampling start date")
    parser.add_argument("-b", "--begin", type=np.datetime64, required=True, help="start date (e.g. 2020-01-01)")
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    parser.add_argument("--rebalance-days", type=int, default=90, help="rebalance every N calendar days")
    parser.add_argument("--initial-cash", type=float, default=1000000.0, help="starting capital (CNY)")
    parser.add_argument("--index-size", type=int, default=100, help="number of stocks in the market-cap index")
    add_market_argument(parser)
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    symbols = load_symbols(data_dir, markets=args.markets)
    print(f"Discovered {len(symbols)} symbols.")

    sc, handles, rebalance_dates = build_scenario(
        symbols,
        data_dir,
        rebalance_days=args.rebalance_days,
        initial_cash=args.initial_cash,
        index_size=args.index_size,
        data_start=resolve_data_start(args.sample_begin, args.begin, args.rebalance_days),
        trading_start=args.begin,
        end=args.end,
    )

    mid = args.begin
    progress = tqdm(total=sc.estimated_event_count(), unit=" events", desc="Loading samples")

    def on_flush(ts_ns: int, events: int, total: int | None) -> None:
        if np.datetime64(ts_ns, "ns") > mid:
            progress.set_description("Running strategy")
        if total != progress.total:
            progress.total = total
        progress.update(events - progress.n)

    sc.run(on_flush=on_flush)
    progress.close()

    # Extract results.
    index = sc.series_view(handles["index"]).to_dataframe(["holdings", "cash"])
    strategy_frictionless = sc.series_view(handles["strategy_frictionless"]).to_dataframe(["holdings", "cash"])
    strategy_actual = sc.series_view(handles["strategy_actual"]).to_dataframe(["holdings", "cash"])
    sharpe = sc.series_view(handles["sharpe"]).to_series()
    compound_return = sc.series_view(handles["compound_return"]).to_series()
    drawdown = sc.series_view(handles["drawdown"]).to_series()

    n = len(index)
    if n == 0:
        raise SystemExit("No data produced.")

    total_value = strategy_actual.sum(axis=1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(f"{n} calendar days, {index.index[0].date()} to {index.index[-1].date()}")
    print(f"Initial value: {total_value.iloc[0]:,.2f} CNY")
    print(f"Final value:   {total_value.iloc[-1]:,.2f} CNY")
    print()

    periods_per_year = 365.0 / args.rebalance_days
    if len(compound_return) > 0:
        compound_return_annualized = (compound_return.iloc[-1] + 1) ** periods_per_year - 1
        print(f"Compound return annualized: {compound_return_annualized:.2%}")
    if len(sharpe) > 0:
        sharpe_annualized = sharpe.iloc[-1] * np.sqrt(periods_per_year)
        print(f"Sharpe ratio annualized: {sharpe_annualized:.4f}")
    if len(drawdown) > 0:
        max_drawdown = drawdown.min()
        print(f"Max drawdown: {max_drawdown:.2%}")
    print()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    def draw_rebalances(ax):
        for d in rebalance_dates:
            ax.axvline(d, color="lightgray", linestyle="--", linewidth=0.4, zorder=0)

    ax = axes[0]
    ax.set_title(f"Portfolio value")
    ax.set_ylabel("CNY (10k)")
    draw_rebalances(ax)
    ax.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    ax.plot(
        index.index,
        index.sum(axis=1) / 1e4,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"Index (top {args.index_size})",
    )
    ax.plot(
        strategy_frictionless.index,
        strategy_frictionless.sum(axis=1) / 1e4,
        color="C0",
        linestyle="--",
        linewidth=0.8,
        label="Strategy (frictionless)",
    )
    ax.plot(
        strategy_actual.index,
        strategy_actual.sum(axis=1) / 1e4,
        color="C0",
        linewidth=0.8,
        label="Strategy (actual)",
    )
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.set_title("Sharpe ratio annualized (since inception)")
    ax.set_ylabel("Sharpe ratio")
    draw_rebalances(ax)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.plot(sharpe.index, sharpe * np.sqrt(periods_per_year), color="C1")

    ax = axes[2]
    ax.set_title("Drawdown (since previous high)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    draw_rebalances(ax)
    ax.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.4, color="C3")

    fig.tight_layout()
    plt.show()
