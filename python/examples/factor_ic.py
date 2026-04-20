"""Evaluate factor RankIC on all A-shares stocks.

For each stock the following features are computed every trading day:

- ``log(market_cap)`` -- log of circulating market capitalisation
- ``log(B/P)`` -- log of book-to-price ratio (parent equity / market cap)
- ``turnover_ma`` -- simple moving average of daily turnover over the
  rebalance window (volume / circulating shares)
- ``ttm_ep`` -- TTM earnings-to-price ratio (annualized net profit / market cap)
- ``ttm_roe`` -- TTM return on equity (annualized net profit / parent equity)
- ``momentum_ma`` -- rolling mean of daily log-returns of adjusted close

Each factor is wrapped in a ``SingleFeature`` pass-through predictor so
its output emits at the rebalance cadence (parallel to the variance
predictor pattern used in ``covariance_gmv.py``).  A ``Sample`` mean
predictor (historical mean returns, no features) is included as a
baseline.  An ``InformationCoefficient`` evaluator (ranking mode)
accumulates daily cross-sectional Spearman rank correlations between
each prediction and realised 1-period forward returns, and reports the
mean RankIC at each rebalance.  Cumulative RankIC curves are plotted.

Requires ``pip install -e ".[examples]"`` and A-shares market data downloaded
via the crawler.  See ``python -m a_shares_crawler --help`` for configuration
and download instructions.
"""

from pathlib import Path
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow import Handle
from tradingflow.sources import Clock, CSVSource, MonthlyClock
from tradingflow.sources.stocks import FinancialReportSource
from tradingflow.operators import Clocked, Lag, Map, NotifyStack, Record, Select, Stack
from tradingflow.operators.num import Divide, Log, Multiply, Subtract
from tradingflow.operators.rolling import RollingMean
from tradingflow.operators.stocks import Annualize, ForwardAdjust
from tradingflow.operators.predictors.mean import Sample, SingleFeature
from tradingflow.operators.metrics.mean import InformationCoefficient

from stocks import load_symbols, calculate_index_weights, resolve_data_start


PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())
DIVIDEND_SCHEMA = Schema(CSVSchema.dividends().iter_field_ids())
BS_SCHEMA = Schema(CSVSchema.balance_sheet().iter_field_ids())
INC_SCHEMA = Schema(CSVSchema.income_statement().iter_field_ids())


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    rebalance_days: int,
    index_size: int,
    data_start: np.datetime64,
    eval_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, np.ndarray]:
    """Build the factor IC evaluation scenario."""

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()
    num_stocks = len(symbols)

    # Per-stock handles grouped by cadence.
    #
    # * `per_stock_sync` — values produced in lockstep across all stocks
    #   (e.g., daily prices/equity on trading days).  Stacked with
    #   `NotifyStack` to give message-passing semantics: slots of stocks
    #   that did not produce this cycle are filled with `NaN`.
    # * `per_stock_irregular` — values updated on stock-specific dates
    #   (e.g., quarterly financial reports filed on different dates).
    #   Stacked with `Stack` to give time-series semantics: slots keep
    #   their last-known value across quiet periods.
    per_stock_sync: dict[str, list[Handle]] = {
        k: []
        for k in (
            "adjusted_close",
            "close",
            "volume",
        )
    }
    per_stock_irregular: dict[str, list[Handle]] = {
        k: []
        for k in (
            "circ_shares",
            "parent_equity",
            "net_profit",
        )
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

        close = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.close")))
        adjusts = sc.add_operator(ForwardAdjust(close, dividends, output_prices=False))
        adjusted_close = sc.add_operator(Multiply(close, adjusts))
        circ_shares = sc.add_operator(Select(equity, EQUITY_SCHEMA.index("shares.circulating")))
        volume = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.volume")))
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

        per_stock_sync["adjusted_close"].append(adjusted_close)
        per_stock_sync["close"].append(close)
        per_stock_sync["volume"].append(volume)
        per_stock_irregular["circ_shares"].append(circ_shares)
        per_stock_irregular["parent_equity"].append(parent_equity)
        per_stock_irregular["net_profit"].append(net_profit)

    # ------------------------------------------------------------------
    # Cross-sectional stacking
    # ------------------------------------------------------------------

    stacked = {
        **{k: sc.add_operator(NotifyStack(v)) for k, v in per_stock_sync.items()},
        **{k: sc.add_operator(Stack(v)) for k, v in per_stock_irregular.items()},
    }

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    market_cap = sc.add_operator(Multiply(stacked["close"], stacked["circ_shares"]))
    log_mcap = sc.add_operator(Log(market_cap))
    log_bp = sc.add_operator(Log(sc.add_operator(Divide(stacked["parent_equity"], market_cap))))
    turnover = sc.add_operator(Divide(stacked["volume"], stacked["circ_shares"]))
    turnover_series = sc.add_operator(Record(turnover))
    turnover_ma = sc.add_operator(RollingMean(turnover_series, window=rebalance_days))

    # TTM net profit via 365-day rolling mean of annualized net profit.
    net_profit_series = sc.add_operator(Record(stacked["net_profit"]))
    net_profit_ttm = sc.add_operator(RollingMean(net_profit_series, window=np.timedelta64(365, "D")))

    # TTM E/P and TTM ROE.
    ttm_ep = sc.add_operator(Divide(net_profit_ttm, market_cap))
    ttm_roe = sc.add_operator(Divide(net_profit_ttm, stacked["parent_equity"]))

    # Momentum MA (rolling mean of daily log-returns of adjusted close).
    log_adj = sc.add_operator(Log(stacked["adjusted_close"]))
    log_adj_series = sc.add_operator(Record(log_adj))
    log_adj_lag = sc.add_operator(Lag(log_adj_series, 1, fill=np.float64(np.nan)))
    daily_ret = sc.add_operator(Subtract(log_adj, log_adj_lag))
    daily_ret_series = sc.add_operator(Record(daily_ret))
    momentum_ma = sc.add_operator(RollingMean(daily_ret_series, window=rebalance_days))

    factor_names = ["log_mcap", "log_bp", "turnover_ma", "ttm_ep", "ttm_roe", "momentum_ma"]
    stacked_features = sc.add_operator(Stack([log_mcap, log_bp, turnover_ma, ttm_ep, ttm_roe, momentum_ma], axis=1))
    features_series = sc.add_operator(Record(stacked_features))
    adjusted_prices_series = sc.add_operator(Record(stacked["adjusted_close"]))

    # Universe: top stocks by market cap.
    monthly_clock = sc.add_source(MonthlyClock(data_start, end, tz="Asia/Shanghai"))
    universe = sc.add_operator(
        Clocked(
            monthly_clock,
            Map(
                market_cap,
                lambda m: calculate_index_weights(m, index_size),
                shape=(num_stocks,),
                dtype=np.float64,
            ),
        )
    )

    # ------------------------------------------------------------------
    # IC / RankIC evaluation
    # ------------------------------------------------------------------

    # Rebalance clock: fires every `rebalance_days` calendar days from
    # `eval_start` to `end` (inclusive).  A Const operator clocked by
    # the clock source produces an Array[float64] rebalance signal —
    # routed as a regular input into the predictors so they observe
    # every data tick (for future incremental accumulators) and emit
    # only when the rebalance signal produces.
    rebalance_dates = np.arange(
        eval_start,
        end + np.timedelta64(1, "D"),
        np.timedelta64(rebalance_days, "D"),
    )
    rebalance_clock = sc.add_source(Clock(rebalance_dates))
    rebalance = rebalance_clock

    predictor_kwargs = dict(universe_size=index_size, rebalance=rebalance)

    estimators = {
        "sample": sc.add_operator(
            Sample(
                universe,
                features_series,
                adjusted_prices_series,
                max_periods=rebalance_days,
                **predictor_kwargs,
            ),
        )
    }
    for i, name in enumerate(factor_names):
        estimators[name] = sc.add_operator(
            SingleFeature(
                universe,
                features_series,
                adjusted_prices_series,
                feature_index=i,
                **predictor_kwargs,
            ),
        )

    eval_handles = {}
    for name, predicted in estimators.items():
        # `predicted` and `stacked["adjusted_close"]` are fed directly
        # as `Array` inputs — IC only reads the latest cross-section
        # of each and caches one previous price tick in its state.
        metric = sc.add_operator(
            InformationCoefficient(
                predicted,
                stacked["adjusted_close"],
                ranking=True,
            )
        )
        eval_handles[name] = sc.add_operator(Record(metric))

    return sc, eval_handles, rebalance_dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument("--data-begin", type=np.datetime64, default=None, help="data sampling start date")
    parser.add_argument("-b", "--begin", type=np.datetime64, required=True, help="start date (e.g. 2020-01-01)")
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    parser.add_argument("--rebalance-days", type=int, default=90, help="rebalance every N calendar days")
    parser.add_argument("--index-size", type=int, default=100, help="number of stocks in the universe")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    symbols = load_symbols(data_dir)
    print(f"Discovered {len(symbols)} symbols.")

    sc, eval_handles, rebalance_dates = build_scenario(
        symbols,
        data_dir,
        rebalance_days=args.rebalance_days,
        index_size=args.index_size,
        data_start=resolve_data_start(args.data_begin, args.begin, args.rebalance_days),
        eval_start=args.begin,
        end=args.end,
    )

    mid = args.begin
    progress = tqdm(total=sc.estimated_event_count(), unit=" events", desc="Loading data")

    def on_flush(ts_ns: int, events: int, total: int | None) -> None:
        if np.datetime64(ts_ns, "ns") > mid:
            progress.set_description("Evaluating IC")
        if total != progress.total:
            progress.total = total
        progress.update(events - progress.n)

    sc.run(on_flush=on_flush)
    progress.close()

    # ------------------------------------------------------------------
    # Extract results with prediction timestamps
    # ------------------------------------------------------------------

    eval_data = {}
    for name, handle in eval_handles.items():
        values = sc.series_view(handle).values()
        n = len(values)
        # `values[i]` covers the window starting at rebalance_dates[i].
        series = pd.Series(values, index=pd.DatetimeIndex(rebalance_dates[:n]), name=name)
        eval_data[name] = series

        finite = series[np.isfinite(series)]
        if len(finite) > 0:
            ic_mean = finite.mean()
            ic_std = finite.std()
            icir = ic_mean / ic_std if ic_std > 0 else np.nan
            print(f"{name}: mean={ic_mean:.4f}, std={ic_std:.4f}, IR={icir:.4f} ({n} periods)")
        else:
            print(f"{name}: no valid values")

    # ------------------------------------------------------------------
    # Plot cumulative RankIC
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("Cumulative RankIC by factor")
    ax.set_ylabel("Cumulative RankIC")
    ax.set_xlabel("Date")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    for i, (name, series) in enumerate(eval_data.items()):
        ax.plot(series.index, series.fillna(0.0).cumsum(), label=name, color=f"C{i}")

    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
