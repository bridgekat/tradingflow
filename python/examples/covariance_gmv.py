"""Compare covariance estimators inside a Markowitz mean-variance strategy.

Selects representative covariance estimators from the survey of
Pantaleo, Tumminello, Lillo, and Mantegna (arXiv:1004.4272) — one from
each filtering family — and compares them on a configurable A-shares
universe.  Each estimator is plugged into a Markowitz mean-variance
portfolio that shares a single
[`LinearRegression`][tradingflow.operators.predictors.mean.LinearRegression]
mean predictor (pooled OLS on `log_mcap`, `log_bp`, `turnover_ma` — the
same features as the
[`mean_variance_strategy`](mean_variance_strategy.py) example), so any
difference in portfolio performance is attributable to the covariance
estimator alone.

Three complementary outputs are reported for every estimator:

1. ``MinimumVariance`` metric — realized variance of the global minimum
   variance portfolio built from each prediction.  Plotted as annualized
   realized volatility; lower is better.
2. Frictionless long-only Markowitz portfolio total value — the
   mean-variance portfolio from the
   [`Markowitz`][tradingflow.operators.portfolios.mean_variance.Markowitz]
   operator (``Mode.MIN_MEAN_VARIANCE``, risk-aversion ``δ``, ``long_only=True``),
   traded frictionlessly via
   [`Benchmark`][tradingflow.operators.traders.Benchmark] with dividend
   reinvestment.
3. Frictionless long-short Markowitz portfolio total value — same
   Markowitz operator with ``long_only=False``.

The ``MinimumVariance`` metric is fed the predicted covariance directly
(without a mean predictor) so it remains a pure diagnostic of covariance
quality and stays comparable to the two strategy curves.  The three
outputs are stacked in a single figure with a shared time axis so their
per-period movements can be compared side by side.

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
from tradingflow.sources import Clock, CSVSource
from tradingflow.sources.stocks import FinancialReportSource
from tradingflow.operators import Clocked, Map, Record, Resample, Select, Stack, StackSync
from tradingflow.operators.num import Diff, Divide, Log, Multiply
from tradingflow.operators.rolling import RollingMean
from tradingflow.operators.stocks import Annualize, ForwardAdjust
from tradingflow.operators.predictors.mean import LinearRegression
from tradingflow.operators.predictors.variance import RMT0, RMTM, Sample, Shrinkage, Target, SingleIndex
from tradingflow.operators.portfolios.mean_variance import Markowitz, Mode
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.metrics.variance import MinimumVariance

from stocks import load_symbols, calculate_index_weights, resolve_data_start, add_market_argument


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
    index_size: int,
    initial_cash: float,
    risk_aversion: float,
    data_start: np.datetime64,
    trading_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, dict, dict, np.ndarray, dict]:
    """Build the covariance comparison scenario.

    Returns
    -------
    sc
        The scenario.
    mv_handles
        ``{estimator_name: record handle}`` for the
        [`MinimumVariance`][tradingflow.operators.metrics.variance.MinimumVariance]
        metric (realized GMV variance per period) — fed the predicted
        covariance directly, independent of the mean predictor.
    l_value_handles
        ``{estimator_name: record handle}`` for the total value
        (``holdings_value + cash``) of a long-only Markowitz portfolio,
        sharing one
        [`LinearRegression`][tradingflow.operators.predictors.mean.LinearRegression]
        mean predictor across estimators and traded frictionlessly via
        [`Benchmark`][tradingflow.operators.traders.Benchmark] with
        dividend reinvestment.
    ls_value_handles
        Same as ``l_value_handles`` but with ``long_only=False`` (long-short).
    rebalance_dates
        The rebalance clock dates, used for timestamp alignment when
        extracting metric values.  Predicted return vectors and covariance
        matrices are *not* recorded — each is fed to the metrics and
        portfolio operators directly, so only one matrix per estimator
        is ever resident in memory.
    handles
        ``{"index_value": record handle}`` for the market-cap-weighted
        index portfolio (universe weights traded frictionlessly via
        [`Benchmark`][tradingflow.operators.traders.Benchmark]), plotted
        as a baseline on both Markowitz portfolio panels.
    """

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    # Per-stock handles grouped by cadence.
    #
    # * `per_stock_sync` — values produced in lockstep across all stocks
    #   (e.g., daily prices/equity on trading days).  Stacked with
    #   `StackSync` to give message-passing semantics: slots of stocks
    #   that did not produce this cycle are filled with `NaN`.
    # * `per_stock_irregular` — values updated on stock-specific dates
    #   (e.g., quarterly financial reports filed on different dates).
    #   Stacked with `Stack` to give time-series semantics: slots keep
    #   their last-known value across quiet periods.
    per_stock_sync: dict[str, list[Handle]] = {k: [] for k in ("ohlcv", "adjusted_close")}
    per_stock_irregular: dict[str, list[Handle]] = {
        k: [] for k in ("adjusts", "circ_shares", "parent_equity", "net_profit")
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

    # Record feature and target history for predictors.  Covariance
    # estimators train on log returns (more symmetric, closer to
    # Gaussian); `VariancePortfolio` / `MeanVariancePortfolio` and the
    # `MinimumVariance` metric perform the lognormal conversion to
    # linear-return moments internally.
    #
    # `stacked_features` ticks on trading days *and* on irregular
    # corporate-event days (balance-sheet notices, equity structure
    # changes), while the returns target only ticks on trading days.
    # Gate feature recording on the trading-day pulse so the features
    # and target records advance lock-step, satisfying the predictor's
    # positional alignment contract.
    sampled_features = sc.add_operator(Resample(stacked["adjusted_close"], stacked_features))
    features_series = sc.add_operator(Record(sampled_features))
    log_adj = sc.add_operator(Log(stacked["adjusted_close"]))
    log_returns = sc.add_operator(Diff(log_adj))
    target_series = sc.add_operator(Record(log_returns))

    # ------------------------------------------------------------------
    # Shared predictors
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
            target_series,
            universe_size=index_size,
            target_offset=1,
            min_periods=100,
            verbose=True,
        ),
    )

    common_args = (universe, features_series, target_series)
    common_kwargs = dict(universe_size=index_size, target_offset=1, max_periods=rebalance_days)

    predicted_covariances = {
        "sample": sc.add_operator(Sample(*common_args, **common_kwargs)),
        "shrinkage_comm_cov": sc.add_operator(
            Shrinkage(*common_args, target=Target.COMMON_COVARIANCE, verbose=False, **common_kwargs)
        ),
        "shrinkage_const_corr": sc.add_operator(
            Shrinkage(*common_args, target=Target.CONSTANT_CORRELATION, verbose=False, **common_kwargs)
        ),
        "shrinkage_single_index": sc.add_operator(
            Shrinkage(*common_args, target=Target.SINGLE_INDEX, verbose=False, **common_kwargs)
        ),
        "rmt_0": sc.add_operator(RMT0(*common_args, **common_kwargs)),
        "rmt_m": sc.add_operator(RMTM(*common_args, **common_kwargs)),
        "single_index": sc.add_operator(SingleIndex(*common_args, **common_kwargs)),
    }

    # ------------------------------------------------------------------
    # Multiple Markowitz portfolios (one per covariance estimator)
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

    mv_handles = {}
    l_value_handles = {}
    ls_value_handles = {}
    for name, predicted_covariance in predicted_covariances.items():
        # GMV realized-variance metric (top plot).  NOTE: `predicted_covariance`
        # is fed directly as an `Array` input — the metric only reads the
        # latest covariance, so recording the full (N, N) history per
        # rebalance would waste O(periods · N²) memory for no benefit.
        mv_metric = sc.add_operator(MinimumVariance(predicted_covariance, log_returns))
        mv_handles[name] = sc.add_operator(Record(mv_metric))

        for long_only, value_handles in ((True, l_value_handles), (False, ls_value_handles)):
            soft_positions = sc.add_operator(
                Markowitz(
                    universe,
                    predicted_returns,
                    predicted_covariance,
                    mode=Mode.MIN_MEAN_VARIANCE,
                    bound=risk_aversion,
                    long_only=long_only,
                    verbose=False,
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
            value_handles[name] = sc.add_operator(Record(frictionless_value))

    return (
        sc,
        mv_handles,
        l_value_handles,
        ls_value_handles,
        rebalance_dates,
        {"index_value": sc.add_operator(Record(index_value))},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument("--sample-begin", type=np.datetime64, default=None, help="data sampling start date")
    parser.add_argument("-b", "--begin", type=np.datetime64, required=True, help="start date (e.g. 2020-01-01)")
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    parser.add_argument("--rebalance-days", type=int, default=90, help="rebalance every N calendar days")
    parser.add_argument("--initial-cash", type=float, default=1000000.0, help="starting capital (CNY)")
    parser.add_argument("--index-size", type=int, default=100, help="number of stocks in the universe")
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
        help="Markowitz variance penalty coefficient δ (Mode.MIN_MEAN_VARIANCE)",
    )
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

    sc, mv_handles, l_value_handles, ls_value_handles, rebalance_dates, handles = build_scenario(
        symbols,
        data_dir,
        rebalance_days=args.rebalance_days,
        index_size=args.index_size,
        initial_cash=args.initial_cash,
        risk_aversion=args.risk_aversion,
        data_start=resolve_data_start(args.sample_begin, args.begin, args.rebalance_days),
        trading_start=args.begin,
        end=args.end,
    )

    mid = args.begin
    progress = tqdm(total=sc.estimated_event_count(), unit=" events", desc="Loading data")

    def on_flush(ts_ns: int, events: int, total: int | None) -> None:
        if np.datetime64(ts_ns, "ns") > mid:
            progress.set_description("Running strategy")
        if total != progress.total:
            progress.total = total
        progress.update(events - progress.n)

    sc.run(on_flush=on_flush)
    progress.close()

    # ------------------------------------------------------------------
    # Extract metric + portfolio-value time series
    # ------------------------------------------------------------------

    # The MV metric emits once per rebalance: `record[i]` corresponds to
    # the window starting at `rebalance_dates[i]`.  Portfolio values are
    # recorded every tick and keep their own daily timestamps.
    TRADING_DAYS = 252

    def extract_per_rebalance(handles: dict) -> dict[str, pd.Series]:
        out = {}
        for name, handle in handles.items():
            values = sc.series_view(handle).values()
            n = len(values)
            out[name] = pd.Series(values, index=pd.DatetimeIndex(rebalance_dates[:n]), name=name)
        return out

    def extract_daily(handles: dict) -> dict[str, pd.Series]:
        return {name: sc.series_view(handle).to_series(name=name) for name, handle in handles.items()}

    mv_data = extract_per_rebalance(mv_handles)
    l_value_data = extract_daily(l_value_handles)
    ls_value_data = extract_daily(ls_value_handles)
    index_value = sc.series_view(handles["index_value"]).to_series()

    for name in mv_handles:
        mv = mv_data[name]
        l_v = l_value_data[name]
        ls_v = ls_value_data[name]
        mv_finite = mv[np.isfinite(mv)]
        mv_str = f"ann vol={np.sqrt(mv_finite.mean() * TRADING_DAYS):.4f}" if len(mv_finite) > 0 else "no valid MV"
        l_final = f"{l_v.iloc[-1]:,.0f}" if len(l_v) > 0 else "—"
        ls_final = f"{ls_v.iloc[-1]:,.0f}" if len(ls_v) > 0 else "—"
        print(f"{name}: {mv_str}, long-only final={l_final} CNY, long-short final={ls_final} CNY ({len(mv)} periods)")

    # ------------------------------------------------------------------
    # Plots — three panels stacked in one figure with shared time axis
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, (ax_mv, ax_l_val, ax_ls_val) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    def draw_rebalances(ax):
        for d in rebalance_dates:
            ax.axvline(d, color="lightgray", linestyle="--", linewidth=0.4, zorder=0)

    # Top panel: GMV annualized realized volatility (lower is better).
    ax_mv.set_title("GMV annualized realized volatility  (lower is better)")
    ax_mv.set_ylabel(f"Annualized (× √{TRADING_DAYS}) realized volatility")
    draw_rebalances(ax_mv)
    for i, (name, series) in enumerate(mv_data.items()):
        ann_vol = np.sqrt(series.clip(lower=0.0) * TRADING_DAYS)
        ax_mv.plot(series.index, ann_vol, label=name, color=f"C{i}", marker="o", markersize=3)
    ax_mv.legend(fontsize=9)

    # Initial view y-range for the two Markowitz panels: 0 to 2× initial
    # capital, so runaway long-short paths don't dominate the axis.  Users
    # can interactively pan/zoom past these limits in the matplotlib GUI.
    value_ylim = (0.0, 2.0 * args.initial_cash / 1e4)

    # Middle panel: Frictionless long-only Markowitz portfolio value (higher is better).
    ax_l_val.set_title(f"Frictionless long-only Markowitz portfolio value  (δ={args.risk_aversion}, higher is better)")
    ax_l_val.set_ylabel("Total value (CNY, 10k)")
    draw_rebalances(ax_l_val)
    ax_l_val.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    ax_l_val.plot(
        index_value.index,
        index_value / 1e4,
        color="gray",
        linewidth=0.8,
        label=f"Index (top {args.index_size})",
    )
    for i, (name, series) in enumerate(l_value_data.items()):
        ax_l_val.plot(series.index, series / 1e4, label=name, color=f"C{i}", linewidth=0.8)
    ax_l_val.set_ylim(value_ylim)
    ax_l_val.legend(fontsize=9)

    # Bottom panel: Frictionless long-short Markowitz portfolio value (higher is better).
    ax_ls_val.set_title(
        f"Frictionless long-short Markowitz portfolio value  (δ={args.risk_aversion}, higher is better)"
    )
    ax_ls_val.set_ylabel("Total value (CNY, 10k)")
    ax_ls_val.set_xlabel("Date")
    draw_rebalances(ax_ls_val)
    ax_ls_val.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    ax_ls_val.plot(
        index_value.index,
        index_value / 1e4,
        color="gray",
        linewidth=0.8,
        label=f"Index (top {args.index_size})",
    )
    for i, (name, series) in enumerate(ls_value_data.items()):
        ax_ls_val.plot(series.index, series / 1e4, label=name, color=f"C{i}", linewidth=0.8)
    ax_ls_val.set_ylim(value_ylim)
    ax_ls_val.legend(fontsize=9)

    fig.tight_layout()
    plt.show()
