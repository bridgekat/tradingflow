"""Compare covariance estimators via two evaluators.

Selects three representative covariance estimators from the survey of
Pantaleo, Tumminello, Lillo, and Mantegna (arXiv:1004.4272) — one from
each filtering family — and compares them on a configurable A-shares
universe:

- ``sample`` — sample covariance (Markowitz baseline).
- ``rmt_m`` — RMT eigenvalue mean-replacement (Potters et al.).
- ``shrinkage_cc`` — Ledoit-Wolf shrinkage toward constant correlation.

Three complementary outputs (all computed per rebalance period) are
reported for every estimator:

1. ``MinimumVariance`` metric — realized variance of the global
   minimum variance portfolio built from each prediction.  Plotted
   as annualized realized volatility; lower is better.
2. ``LogLikelihood`` metric — period-averaged Gaussian negative
   log-likelihood of the realized daily returns under the predicted
   covariance, ``log |Σ| + (1/T) Σₜ rₜᵀ Σ⁻¹ rₜ``.  Lower is better.
3. ``Benchmark``-traced total portfolio value — the long-only GMV
   portfolio from the
   [`MinimumVariance`][tradingflow.operators.portfolios.variance.MinimumVariance]
   operator, frictionlessly traded via
   [`Benchmark`][tradingflow.operators.traders.Benchmark] with
   dividend reinvestment.  Higher is better.

All estimators ignore features, so the features series is fed an empty
``(num_stocks, 0)`` array per tick.  The three outputs are stacked in a
single figure with a shared time axis so their per-period movements
can be compared side by side.

Requires ``pip install -e ".[examples]"`` and A-shares market data downloaded
via the crawler.  See ``python -m a_shares_crawler --help`` for configuration
and download instructions.
"""

from pathlib import Path
from re import L
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow.types import Handle
from tradingflow.sources import Clock, CSVSource, MonthlyClock
from tradingflow.operators import Clocked, Map, NotifyStack, Record, Select, Stack
from tradingflow.operators.num import Multiply
from tradingflow.operators.stocks import ForwardAdjust
from tradingflow.operators.predictors.variance import RMT0, RMTM, Sample, Shrinkage, Target, SingleIndex
from tradingflow.operators.portfolios.variance import MinimumVariance as MinimumVariancePortfolio
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.metrics.variance import LogLikelihood, MinimumVariance

from stocks import load_symbols, calculate_index_weights, resolve_data_start


PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())
DIVIDEND_SCHEMA = Schema(CSVSchema.dividends().iter_field_ids())
OHLCV_INDICES = PRICE_SCHEMA.indices(["prices.open", "prices.high", "prices.low", "prices.close", "prices.volume"])


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    rebalance_days: int,
    index_size: int,
    initial_cash: float,
    data_start: np.datetime64,
    eval_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, dict, dict, dict, np.ndarray]:
    """Build the covariance evaluation scenario.

    Returns
    -------
    sc
        The scenario.
    mv_handles
        ``{estimator_name: record handle}`` for the
        [`MinimumVariance`][tradingflow.operators.metrics.variance.MinimumVariance]
        metric (realized GMV variance per period).
    ll_handles
        ``{estimator_name: record handle}`` for the
        [`LogLikelihood`][tradingflow.operators.metrics.variance.LogLikelihood]
        metric.
    value_handles
        ``{estimator_name: record handle}`` for the total value
        (``holdings_value + cash``) of a long-only GMV portfolio
        traded via [`Benchmark`][tradingflow.operators.traders.Benchmark]
        with dividend reinvestment.
    rebalance_dates
        The rebalance clock dates, used for timestamp alignment when
        extracting metric values.  Predicted covariance matrices are
        *not* recorded — each is fed to the metrics directly as an
        ``Array`` input, so only one covariance matrix per estimator
        is ever resident in memory.
    """

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
            "ohlcv",
            "adjusted_close",
            "close",
        )
    }
    per_stock_irregular: dict[str, list[Handle]] = {
        k: []
        for k in (
            "adjusts",
            "circ_shares",
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

        ohlcv = sc.add_operator(Select(prices, OHLCV_INDICES))
        close = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.close")))
        adjusts = sc.add_operator(ForwardAdjust(close, dividends, output_prices=False))
        adjusted_close = sc.add_operator(Multiply(close, adjusts))
        circ_shares = sc.add_operator(Select(equity, EQUITY_SCHEMA.index("shares.circulating")))

        per_stock_sync["ohlcv"].append(ohlcv)
        per_stock_sync["adjusted_close"].append(adjusted_close)
        per_stock_sync["close"].append(close)
        per_stock_irregular["adjusts"].append(adjusts)
        per_stock_irregular["circ_shares"].append(circ_shares)

    # ------------------------------------------------------------------
    # Cross-sectional stacking
    # ------------------------------------------------------------------

    stacked = {
        **{k: sc.add_operator(NotifyStack(v)) for k, v in per_stock_sync.items()},
        **{k: sc.add_operator(Stack(v)) for k, v in per_stock_irregular.items()},
    }

    # ------------------------------------------------------------------
    # Features (empty — none of the covariance estimators use features)
    # ------------------------------------------------------------------

    market_cap = sc.add_operator(Multiply(stacked["close"], stacked["circ_shares"]))

    # Empty features array (N, 0) ticking with adjusted_close.
    empty_features = sc.add_operator(
        Map(
            stacked["adjusted_close"],
            lambda _: np.zeros((num_stocks, 0), dtype=np.float64),
            shape=(num_stocks, 0),
            dtype=np.float64,
        )
    )
    features_series = sc.add_operator(Record(empty_features))
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
    # GMV evaluation
    # ------------------------------------------------------------------

    # Rebalance clock fires every `rebalance_days` calendar days from
    # `eval_start` to `end`.  A Const operator clocked by it produces
    # an Array[float64] rebalance signal that the predictors consume
    # as a regular input (observing every data tick but emitting only
    # on rebalance).
    rebalance_dates = np.arange(
        eval_start,
        end + np.timedelta64(1, "D"),
        np.timedelta64(rebalance_days, "D"),
    )
    rebalance_clock = sc.add_source(Clock(rebalance_dates))
    rebalance = rebalance_clock

    predictor_kwargs = dict(universe_size=index_size, rebalance=rebalance)

    common_args = (universe, features_series, adjusted_prices_series)
    common_kwargs = dict(max_periods=rebalance_days, **predictor_kwargs)

    estimators = {
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

    mv_handles = {}
    ll_handles = {}
    l_value_handles = {}
    ls_value_handles = {}
    for name, predicted in estimators.items():
        # Two evaluation metrics.  NOTE: `predicted` is fed directly
        # as an `Array` input — the metrics only read the latest
        # covariance, so recording the full (N, N) history per
        # rebalance would waste O(periods · N²) memory for no benefit.
        mv_metric = sc.add_operator(MinimumVariance(predicted, stacked["adjusted_close"]))
        ll_metric = sc.add_operator(LogLikelihood(predicted, stacked["adjusted_close"]))
        mv_handles[name] = sc.add_operator(Record(mv_metric))
        ll_handles[name] = sc.add_operator(Record(ll_metric))

        # Traded long-only GMV portfolio.
        soft_positions = sc.add_operator(MinimumVariancePortfolio(universe, predicted, long_only=True, verbose=False))
        traded = sc.add_operator(
            Benchmark(
                soft_positions,
                stacked["ohlcv"],
                stacked["adjusts"],
                initial_cash=initial_cash,
                use_adjusts=True,
            )
        )
        total_value = sc.add_operator(Map(traded, np.sum, shape=(), dtype=np.float64))
        l_value_handles[name] = sc.add_operator(Record(total_value))

        # Traded long-short GMV portfolio.
        soft_positions = sc.add_operator(MinimumVariancePortfolio(universe, predicted, long_only=False, verbose=False))
        traded = sc.add_operator(
            Benchmark(
                soft_positions,
                stacked["ohlcv"],
                stacked["adjusts"],
                initial_cash=initial_cash,
                use_adjusts=True,
            )
        )
        total_value = sc.add_operator(Map(traded, np.sum, shape=(), dtype=np.float64))
        ls_value_handles[name] = sc.add_operator(Record(total_value))

    return sc, mv_handles, ll_handles, l_value_handles, ls_value_handles, rebalance_dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument(
        "--data-begin",
        type=np.datetime64,
        default=None,
        help="data start date (default: trading begin minus --rebalance-days calendar days)",
    )
    parser.add_argument(
        "-b", "--begin", type=np.datetime64, required=True, help="evaluation start date (e.g. 2020-01-01)"
    )
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    parser.add_argument("--rebalance-days", type=int, default=120, help="rebalance every N trading days")
    parser.add_argument("--index-size", type=int, default=300, help="number of stocks in the universe")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000.0,
        help="starting capital for the Benchmark-traced GMV portfolio (CNY)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    symbols = load_symbols(data_dir)
    print(f"Discovered {len(symbols)} symbols.")

    sc, mv_handles, ll_handles, l_value_handles, ls_value_handles, rebalance_dates = build_scenario(
        symbols,
        data_dir,
        rebalance_days=args.rebalance_days,
        index_size=args.index_size,
        initial_cash=args.initial_cash,
        data_start=resolve_data_start(args.data_begin, args.begin, args.rebalance_days),
        eval_start=args.begin,
        end=args.end,
    )

    mid = args.begin
    progress = tqdm(total=sc.estimated_event_count(), unit=" events", desc="Loading data")

    def on_flush(ts_ns: int, events: int, total: int | None) -> None:
        if np.datetime64(ts_ns, "ns") > mid:
            progress.set_description("Evaluating GMV")
        if total != progress.total:
            progress.total = total
        progress.update(events - progress.n)

    sc.run(on_flush=on_flush)
    progress.close()

    # ------------------------------------------------------------------
    # Extract metric + portfolio-value time series
    # ------------------------------------------------------------------

    # MV and LL metrics emit once per rebalance: `record[i]` corresponds
    # to the window starting at rebalance_dates[i].  The portfolio value
    # is recorded every tick and keeps its own daily timestamps.
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
    ll_data = extract_per_rebalance(ll_handles)
    l_value_data = extract_daily(l_value_handles)
    ls_value_data = extract_daily(ls_value_handles)

    for name in mv_handles:
        mv = mv_data[name]
        ll = ll_data[name]
        l_v = l_value_data[name]
        ls_v = ls_value_data[name]
        mv_finite = mv[np.isfinite(mv)]
        ll_finite = ll[np.isfinite(ll)]
        mv_str = f"ann vol={np.sqrt(mv_finite.mean() * TRADING_DAYS):.4f}" if len(mv_finite) > 0 else "no valid MV"
        ll_str = f"NLL={ll_finite.mean():.4f}" if len(ll_finite) > 0 else "no valid LL"
        print(f"{name}: {mv_str}, {ll_str} ({len(mv)} periods)")

    # ------------------------------------------------------------------
    # Plots — three panels stacked in one figure with shared time axis
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, (ax_mv, ax_l_val, ax_ls_val) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Covariance estimator comparison (shared time axis)")

    # Top panel: GMV annualized realized volatility (lower is better).
    ax_mv.set_title("GMV annualized realized volatility  (lower is better)")
    ax_mv.set_ylabel(f"Annualized (× √{TRADING_DAYS}) realized volatility")
    for i, (name, series) in enumerate(mv_data.items()):
        ann_vol = np.sqrt(series.clip(lower=0.0) * TRADING_DAYS)
        ax_mv.plot(series.index, ann_vol, label=name, color=f"C{i}", marker="o", markersize=3)
    ax_mv.legend(fontsize=9)

    # # Middle panel: Gaussian negative log-likelihood (lower is better).
    # ax_ll.set_title("Gaussian negative log-likelihood  (lower is better)")
    # ax_ll.set_ylabel(r"log|$\Sigma$| + mean $r^T \Sigma^{-1} r$")
    # for i, (name, series) in enumerate(ll_data.items()):
    #     ax_ll.plot(series.index, series, label=name, color=f"C{i}", marker="o", markersize=3)
    # ax_ll.legend(fontsize=9)

    # Middle panel: Frictionless long-only GMV portfolio value (less volatile is better).
    ax_l_val.set_title("Frictionless long-only GMV portfolio value  (less volatile is better)")
    ax_l_val.set_ylabel("Total value (CNY, 10k)")
    ax_l_val.set_xlabel("Date")
    ax_l_val.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    for i, (name, series) in enumerate(l_value_data.items()):
        ax_l_val.plot(series.index, series / 1e4, label=name, color=f"C{i}", linewidth=0.8)
    ax_l_val.legend(fontsize=9)

    # Bottom panel: Frictionless long-short GMV portfolio value (less volatile is better).
    ax_ls_val.set_title("Frictionless long-short GMV portfolio value  (less volatile is better)")
    ax_ls_val.set_ylabel("Total value (CNY, 10k)")
    ax_ls_val.set_xlabel("Date")
    ax_ls_val.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    for i, (name, series) in enumerate(ls_value_data.items()):
        ax_ls_val.plot(series.index, series / 1e4, label=name, color=f"C{i}", linewidth=0.8)
    ax_ls_val.legend(fontsize=9)

    fig.tight_layout()
    plt.show()
