"""Compare covariance estimators inside a Markowitz mean-variance strategy.

Selects representative covariance estimators from the survey of
Pantaleo, Tumminello, Lillo, and Mantegna (arXiv:1004.4272) - one from
each filtering family - and compares them on a configurable A-shares
universe.  Each estimator is plugged into a Markowitz mean-variance
portfolio that shares a single
[`LinearRegression`][tradingflow.operators.predictors.mean.LinearRegression]
mean predictor (pooled OLS on `log_mcap`, `log_bp`, `turnover_ma` - the
same features as the
[`mean_variance_strategy`](mean_variance_strategy.py) example), so any
difference in portfolio performance is attributable to the covariance
estimator alone.

Three complementary outputs are reported for every estimator:

1. ``MinimumVariance`` metric - realized variance of the global minimum
   variance portfolio built from each prediction.  Plotted as annualized
   realized volatility; lower is better.
2. Frictionless long-only Markowitz portfolio total value - the
   mean-variance portfolio from the
   [`Markowitz`][tradingflow.operators.portfolios.mean_variance.Markowitz]
   operator (``Mode.MIN_MEAN_VARIANCE``, risk-aversion ``δ``, ``long_only=True``),
   traded frictionlessly via
   [`Benchmark`][tradingflow.operators.traders.Benchmark] with dividend
   reinvestment.
3. Frictionless long-short Markowitz portfolio total value - same
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
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tradingflow import Scenario
from tradingflow.operators import Map, Record
from tradingflow.operators.num import Diff, Log, Multiply
from tradingflow.operators.predictors.mean import LinearRegression
from tradingflow.operators.predictors.variance import RMT0, RMTM, Sample, Shrinkage, Target, SingleIndex
from tradingflow.operators.portfolios.mean_variance import Markowitz, Mode
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.metrics.variance import MinimumVariance

from common import (
    add_common_arguments,
    build_cap_weighted_universe,
    build_demeaned_log_return_target,
    build_features,
    build_price_limits,
    build_rebalance_clock,
    build_stacked,
    make_progress_tracker,
    resolve_data_start,
    validate_data_dir,
)


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
        metric (realized GMV variance per period) - fed the predicted
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
        matrices are *not* recorded - each is fed to the metrics and
        portfolio operators directly, so only one matrix per estimator
        is ever resident in memory.
    handles
        ``{"index_value": record handle}`` for the market-cap-weighted
        index portfolio (universe weights traded frictionlessly via
        [`Benchmark`][tradingflow.operators.traders.Benchmark]), plotted
        as a baseline on both Markowitz portfolio panels.
    """

    sc = Scenario()

    # Load per-stock CSVs and stack into the cross-sectional panel.
    # See `per_stock.py` for the canonical data pipeline.
    stacked = build_stacked(sc, symbols, data_dir, data_start=data_start, end=end)

    num_stocks = len(symbols)
    window = 20

    # Cross-sectional features (canonical factor set; see `features.py`).
    features, features_series = build_features(sc, stacked, num_stocks=num_stocks, window=window)

    # Re-compute the handles examples need downstream: `market_cap` for
    # `market_cap` for the cap-weighted universe, `log_adj` for the
    # daily log-return target.  Both are also built inside
    # `build_features`; the duplication is intentional to keep the
    # helper's return signature minimal.
    market_cap = sc.add_operator(Multiply(stacked["close"], stacked["total_shares"]))
    log_adj = sc.add_operator(Log(stacked["adjusted_close"]))

    # Regression target: cross-sectionally de-meaned, winsorized daily
    # log returns.  Both the mean predictor and the covariance
    # estimators consume `target_series`; both are shift-invariant
    # under per-day additive shifts, so the demean step changes no
    # downstream portfolio under full-position constraints.
    target, target_series = build_demeaned_log_return_target(sc, log_adj, num_stocks=num_stocks)

    # Raw (non-winsorized, non-demeaned) daily log returns, used by the
    # `MinimumVariance` metric to score each covariance estimator on
    # realized portfolio variance.
    log_returns = sc.add_operator(Diff(log_adj))

    # Daily price-limit handles (constant ±10% for now).
    upper_limit, lower_limit = build_price_limits(sc, stacked["close"], num_stocks=num_stocks)

    # ------------------------------------------------------------------
    # Shared predictors
    # ------------------------------------------------------------------

    rebalance_clock, rebalance_dates = build_rebalance_clock(sc, trading_start, end, rebalance_days)
    universe = build_cap_weighted_universe(
        sc,
        market_cap,
        rebalance_clock,
        num_stocks=num_stocks,
        index_size=index_size,
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
            stacked["close"],
            stacked["adjusts"],
            upper_limit,
            lower_limit,
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
        # is fed directly as an `Array` input - the metric only reads the
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
                    stacked["close"],
                    stacked["adjusts"],
                    upper_limit,
                    lower_limit,
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
    add_common_arguments(parser, include_initial_cash=True)
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1.0,
        help="Markowitz variance penalty coefficient δ (Mode.MIN_MEAN_VARIANCE)",
    )
    args = parser.parse_args()

    data_dir, symbols = validate_data_dir(args)

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

    on_flush, progress = make_progress_tracker(sc, args.begin, after_desc="Running strategy")
    session = sc.run(on_flush=on_flush)
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
            values = session.series_view(handle).values()
            n = len(values)
            out[name] = pd.Series(values, index=pd.DatetimeIndex(rebalance_dates[:n]), name=name)
        return out

    def extract_daily(handles: dict) -> dict[str, pd.Series]:
        return {name: session.series_view(handle).to_series(name=name) for name, handle in handles.items()}

    mv_data = extract_per_rebalance(mv_handles)
    l_value_data = extract_daily(l_value_handles)
    ls_value_data = extract_daily(ls_value_handles)
    index_value = session.series_view(handles["index_value"]).to_series()

    for name in mv_handles:
        mv = mv_data[name]
        l_v = l_value_data[name]
        ls_v = ls_value_data[name]
        mv_finite = mv[np.isfinite(mv)]
        mv_str = f"ann vol={np.sqrt(mv_finite.mean() * TRADING_DAYS):.4f}" if len(mv_finite) > 0 else "no valid MV"
        l_final = f"{l_v.iloc[-1]:,.0f}" if len(l_v) > 0 else "-"
        ls_final = f"{ls_v.iloc[-1]:,.0f}" if len(ls_v) > 0 else "-"
        print(f"{name}: {mv_str}, long-only final={l_final} CNY, long-short final={ls_final} CNY ({len(mv)} periods)")

    # ------------------------------------------------------------------
    # Plots - three panels stacked in one figure with shared time axis
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
