"""Backtest Markowitz mean-variance strategies with different risk aversions.

Compares multiple portfolios optimized with the following formulation:

    ```
    maximize:       mu' x - δ * sqrt(x' Σ x)
    subject to:     1' x = 1
                    x >= 0
    ```

where `δ` (risk aversion) is varied across runs.  All variants share
the same data sources, features, return predictor, and covariance estimator
within a single computation graph; they diverge only at the portfolio
construction and trading stages.

Requires ``pip install -e ".[examples]"`` and A-shares market data downloaded
via the crawler.  See ``python -m a_shares_crawler --help`` for configuration
and download instructions.
"""

from pathlib import Path
from statistics import LinearRegression
from tqdm import tqdm
import argparse

import numpy as np
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow.types import Handle
from tradingflow.sources import Clock, CSVSource, MonthlyClock
from tradingflow.sources.stocks import FinancialReportSource
from tradingflow.operators import Apply, Const, Map, Record, Select, Stack
from tradingflow.operators.num import Divide, ForwardFill, Log, Multiply
from tradingflow.operators.predictors.mean import LinearRegression
from tradingflow.operators.predictors.variance import Sample, Shrinkage
from tradingflow.operators.portfolios.mean_variance import Markowitz
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.metrics import CompoundReturn, SharpeRatio, Drawdown
from tradingflow.operators.rolling import RollingMean
from tradingflow.operators.stocks import Annualize, ForwardAdjust

from stocks import load_symbols, calculate_index_weights, resolve_data_start


PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())
DIVIDEND_SCHEMA = Schema(CSVSchema.dividends().iter_field_ids())
BS_SCHEMA = Schema(CSVSchema.balance_sheet().iter_field_ids())
INC_SCHEMA = Schema(CSVSchema.income_statement().iter_field_ids())
OHLCV_INDICES = PRICE_SCHEMA.indices(["prices.open", "prices.high", "prices.low", "prices.close", "prices.volume"])

RISK_AVERSIONS = np.linspace(0.0, 5.0, 11).round(2).tolist()


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    risk_aversions: list[float],
    rebalance_days: int,
    initial_cash: float,
    index_size: int,
    data_start: np.datetime64,
    trading_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, dict]:
    """Build a scenario with shared data/features and multiple Markowitz variants."""

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    per_stock: dict[str, list[Handle]] = {
        k: [] for k in ("ohlcv", "adjusts", "adjusted_close", "circ_shares", "parent_equity", "net_profit")
    }

    for symbol in tqdm(symbols, desc="Building scenario"):
        h = history_dir / symbol

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

        ohlcv = sc.add_operator(Select(prices, OHLCV_INDICES))
        close = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.close")))
        adjusts = sc.add_operator(ForwardAdjust(close, dividends, output_prices=False))
        adjusted_close = sc.add_operator(Multiply(close, adjusts))
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

        per_stock["ohlcv"].append(ohlcv)
        per_stock["adjusts"].append(adjusts)
        per_stock["adjusted_close"].append(adjusted_close)
        per_stock["circ_shares"].append(circ_shares)
        per_stock["parent_equity"].append(parent_equity)
        per_stock["net_profit"].append(net_profit)

    # ------------------------------------------------------------------
    # Cross-sectional operators (shared)
    # ------------------------------------------------------------------

    stacked = {k: sc.add_operator(ForwardFill(sc.add_operator(Stack(v)))) for k, v in per_stock.items()}

    num_stocks = len(symbols)
    close = sc.add_operator(Select(stacked["ohlcv"], 3, axis=1))
    volume = sc.add_operator(Select(stacked["ohlcv"], 4, axis=1))
    market_cap = sc.add_operator(Multiply(close, stacked["circ_shares"]))
    log_mcap = sc.add_operator(Log(market_cap))

    # Stock index: top index_size companies by market cap, cap-weighted.
    # Rebalanced monthly to avoid per-tick overhead.
    monthly_clock = sc.add_source(MonthlyClock(data_start, end, tz="Asia/Shanghai"))
    universe = sc.add_operator(
        Map(
            market_cap,
            lambda m: calculate_index_weights(m, index_size),
            shape=(num_stocks,),
            dtype=np.float64,
        ),
        clock=monthly_clock,
    )
    log_bp = sc.add_operator(Log(sc.add_operator(Divide(stacked["parent_equity"], market_cap))))
    turnover = sc.add_operator(Divide(volume, stacked["circ_shares"]))
    turnover_series = sc.add_operator(Record(turnover))
    turnover_ma = sc.add_operator(RollingMean(turnover_series, window=rebalance_days))
    stacked_features = sc.add_operator(Stack([log_mcap, log_bp, turnover_ma], axis=1))

    # Record feature and price history for predictors.
    features_series = sc.add_operator(Record(stacked_features))
    adjusted_prices_series = sc.add_operator(Record(stacked["adjusted_close"]))

    # ------------------------------------------------------------------
    # Shared predictors
    # ------------------------------------------------------------------

    # Rebalance clock fires every `rebalance_days` from trading_start.
    # Const clocked by it produces an Array[float64] rebalance signal
    # which the predictors consume as a regular input.
    rebalance_dates = np.arange(
        trading_start,
        end + np.timedelta64(1, "D"),
        np.timedelta64(rebalance_days, "D"),
    )
    rebalance_clock = sc.add_source(Clock(rebalance_dates))
    rebalance = sc.add_operator(Const(np.array(np.nan, dtype=np.float64)), clock=rebalance_clock)

    predicted_returns = sc.add_operator(
        LinearRegression(
            universe,
            features_series,
            adjusted_prices_series,
            rebalance=rebalance,
            universe_size=index_size,
            min_periods=100,
            verbose=True,
        ),
    )

    predicted_covariances = sc.add_operator(
        Shrinkage(
            universe,
            features_series,
            adjusted_prices_series,
            rebalance=rebalance,
            universe_size=index_size,
            max_periods=100,
            min_periods=50,
        ),
    )

    # ------------------------------------------------------------------
    # Multiple Markowitz variants (one per risk_aversion)
    # ------------------------------------------------------------------

    index = sc.add_operator(
        Benchmark(
            universe,
            stacked["ohlcv"],
            stacked["adjusts"],
            initial_cash=initial_cash,
            use_adjusts=True,
        )
    )

    index_value = sc.add_operator(Map(index, np.sum, shape=(), dtype=np.float64))

    variants: dict[float, dict[str, Handle]] = {}
    for delta in risk_aversions:
        soft_positions = sc.add_operator(
            Markowitz(
                universe,
                predicted_returns,
                predicted_covariances,
                risk_aversion=delta,
                long_only=True,
                verbose=True,
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

        frictionless_value = sc.add_operator(Map(strategy_frictionless, np.sum, shape=(), dtype=np.float64))

        def _expected_return_risk(x, mu, sigma):
            mask = np.isfinite(x) & (x > 0)
            x = x[mask]
            mu = mu[mask]
            sigma = sigma[np.ix_(mask, mask)]
            exp_ret = mu @ x
            exp_risk = np.sqrt(np.max(x @ sigma @ x, 0))
            return np.array([exp_ret, exp_risk])

        frontier_point = sc.add_operator(
            Apply(
                (soft_positions, predicted_returns, predicted_covariances),
                _expected_return_risk,
                shape=(2,),
                dtype=np.float64,
            )
        )

        variants[delta] = {
            "value": sc.add_operator(Record(frictionless_value)),
            "sharpe": sc.add_operator(Record(sc.add_operator(SharpeRatio(frictionless_value), clock=monthly_clock))),
            "drawdown": sc.add_operator(Record(sc.add_operator(Drawdown(frictionless_value)))),
            "compound": sc.add_operator(
                Record(sc.add_operator(CompoundReturn(frictionless_value), clock=monthly_clock))
            ),
            "frontier": sc.add_operator(Record(frontier_point)),
        }

    return sc, variants, {"index_value": sc.add_operator(Record(index_value))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument(
        "--data-begin",
        type=np.datetime64,
        default=None,
        help="data start date (default: trading begin minus --rebalance-days calendar days)",
    )
    parser.add_argument("-b", "--begin", type=np.datetime64, required=True, help="trading start date (e.g. 2024-01-01)")
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    parser.add_argument("--rebalance-days", type=int, default=120, help="rebalance every N trading days")
    parser.add_argument("--initial-cash", type=float, default=1000000.0, help="starting capital (CNY)")
    parser.add_argument("--index-size", type=int, default=300, help="number of stocks in the market-cap index")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    symbols = load_symbols(data_dir)
    print(f"Discovered {len(symbols)} symbols.")

    sc, variants, handles = build_scenario(
        symbols,
        data_dir,
        risk_aversions=RISK_AVERSIONS,
        rebalance_days=args.rebalance_days,
        initial_cash=args.initial_cash,
        index_size=args.index_size,
        data_start=resolve_data_start(args.data_begin, args.begin, args.rebalance_days),
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
    results: dict[float, dict] = {}
    index_value = sc.series_view(handles["index_value"]).to_series()
    for delta, variant in variants.items():
        value = sc.series_view(variant["value"]).to_series()
        sharpe = sc.series_view(variant["sharpe"]).to_series()
        drawdown = sc.series_view(variant["drawdown"]).to_series()
        compound = sc.series_view(variant["compound"]).to_series()
        frontier = sc.series_view(variant["frontier"]).to_dataframe(["exp_return", "exp_risk"])
        results[delta] = {
            "value": value,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "compound": compound,
            "frontier": frontier,
        }
        car = ((compound.iloc[-1] + 1) ** 12 - 1) if len(compound) > 0 else 0.0
        sr = sharpe.iloc[-1] * np.sqrt(12) if len(sharpe) > 0 else 0.0
        mdd = drawdown.min() if len(drawdown) > 0 else 0.0
        print(f"delta={delta:.1f}: final={value.iloc[-1]:,.0f} CNY, annual={car:.2%}, sharpe={sr:.3f}, mdd={mdd:.2%}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    cm = plt.colormaps.get("viridis")
    assert cm is not None
    colors = cm(np.linspace(0.0, 1.0, len(results)))

    # Panel 1 (top-left): Portfolio value
    ax = axes[0, 0]
    ax.set_title("Portfolio value")
    ax.set_ylabel("CNY (10k)")
    ax.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    ax.plot(
        index_value.index,
        index_value / 1e4,
        color="gray",
        linewidth=0.8,
        label=f"Index (top {args.index_size})",
    )
    for (delta, result), color in zip(results.items(), colors):
        ax.plot(
            result["value"].index,
            result["value"] / 1e4,
            color=color,
            linewidth=0.8,
            label=f"delta={delta}",
        )
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("Date")

    # Panel 2 (top-right): Efficient frontier over time
    ax = axes[0, 1]
    ax.set_title("Efficient frontier (at each rebalance)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Build per-rebalance curves connecting deltas from lowest to highest.
    sorted_deltas = sorted(results.keys())
    frontier_by_delta = {d: results[d]["frontier"].dropna() for d in sorted_deltas}
    common_ts = frontier_by_delta[sorted_deltas[0]].index
    for d in sorted_deltas[1:]:
        common_ts = common_ts.intersection(frontier_by_delta[d].index)

    delta_to_color = dict(zip(results.keys(), colors))
    n_rebalances = len(common_ts)
    for i, ts in enumerate(common_ts):
        alpha = 0.15 + 0.85 * i / max(n_rebalances - 1, 1)
        risks = [frontier_by_delta[d].loc[ts, "exp_risk"] * 100 for d in sorted_deltas]
        rets = [frontier_by_delta[d].loc[ts, "exp_return"] * 100 for d in sorted_deltas]
        ax.plot(risks, rets, color="gray", linewidth=0.4, alpha=alpha * 0.5)
        for d, r, ret in zip(sorted_deltas, risks, rets):
            ax.scatter(r, ret, s=10, color=delta_to_color[d], alpha=alpha, zorder=3)

    for (delta, _), color in zip(results.items(), colors):
        ax.scatter([], [], s=20, color=color, label=f"δ={delta}")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("Expected risk (%)")
    ax.set_ylabel("Expected return (%)")

    # Panel 3 (bottom-left): Sharpe ratio
    ax = axes[1, 0]
    ax.set_title("Monthly Sharpe ratio annualized (since inception)")
    ax.set_ylabel("Sharpe")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    for (delta, result), color in zip(results.items(), colors):
        ax.plot(
            result["sharpe"].index,
            result["sharpe"] * np.sqrt(12),
            linewidth=0.8,
            color=color,
            label=f"delta={delta}",
        )
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("Date")

    # Panel 4 (bottom-right): Drawdown
    ax = axes[1, 1]
    ax.set_title("Drawdown (since previous high)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    for (delta, result), color in zip(results.items(), colors):
        ax.plot(
            result["drawdown"].index,
            result["drawdown"] * 100,
            linewidth=0.8,
            color=color,
            alpha=0.7,
            label=f"delta={delta}",
        )
    ax.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    plt.show()
