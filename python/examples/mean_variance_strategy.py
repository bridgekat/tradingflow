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
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tradingflow import Scenario, Handle
from tradingflow.operators import Apply, Map, Record
from tradingflow.operators.num import Log, Multiply
from tradingflow.operators.predictors.mean import LinearRegression
from tradingflow.operators.predictors.variance import Shrinkage
from tradingflow.operators.portfolios.mean_variance import Markowitz, Mode
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.metrics import CompoundReturn, SharpeRatio, Drawdown

from common import (
    add_common_arguments,
    build_cap_weighted_universe,
    build_demeaned_log_return_target,
    build_features,
    build_price_limits,
    build_rebalance_clock,
    build_stacked,
    find_effective_trading_start,
    make_progress_tracker,
    resolve_data_start,
    scale_to_initial_cash,
    validate_data_dir,
)

RISK_AVERSIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    risk_aversions: list[float],
    rebalance_days: int,
    index_size: int,
    data_start: np.datetime64,
    trading_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, dict, np.ndarray]:
    """Build the full backtesting scenario."""

    sc = Scenario()

    # Load per-stock CSVs and stack into the cross-sectional panel.
    # See `per_stock.py` for the canonical data pipeline.
    stacked = build_stacked(sc, symbols, data_dir, data_start=data_start, end=end)

    num_stocks = len(symbols)
    window = 20

    # Cross-sectional features (canonical factor set; see `features.py`).
    features, features_series = build_features(sc, stacked, num_stocks=num_stocks, window=window)

    # `market_cap` for the cap-weighted universe, `log_adj` for the
    # daily log-return target.  Both are also built inside
    # `build_features`; the duplication is intentional to keep the
    # helper's return signature minimal.
    market_cap = sc.add_operator(Multiply(stacked["close"], stacked["total_shares"]))
    log_adj = sc.add_operator(Log(stacked["adjusted_close"]))

    # Regression target: cross-sectionally de-meaned, winsorized daily
    # log returns.  The predictor pairs feature[i] with
    # target[i + target_offset] - here target_offset=1, so feature at
    # day t predicts the return from t to t+1.
    target, target_series = build_demeaned_log_return_target(sc, log_adj, num_stocks=num_stocks)

    # Daily price-limit handles (constant ±10% for now).
    upper_limit, lower_limit = build_price_limits(sc, stacked["close"], num_stocks=num_stocks)

    # ------------------------------------------------------------------
    # Strategy pipeline
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

    predicted_covariances = sc.add_operator(
        Shrinkage(
            universe,
            features_series,
            target_series,
            universe_size=index_size,
            target_offset=1,
            max_periods=200,
            min_periods=100,
        ),
    )

    # ------------------------------------------------------------------
    # Multiple Markowitz variants (one per risk_aversion)
    # ------------------------------------------------------------------

    index = sc.add_operator(
        Benchmark(
            universe,
            stacked["close"],
            stacked["adjusts"],
            upper_limit,
            lower_limit,
            initial_cash=1.0,
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
                mode=Mode.MIN_MEAN_STD_DEV,
                bound=delta,
                long_only=True,
                verbose=True,
            )
        )

        strategy_frictionless = sc.add_operator(
            Benchmark(
                soft_positions,
                stacked["close"],
                stacked["adjusts"],
                upper_limit,
                lower_limit,
                initial_cash=1.0,
                use_adjusts=True,
            )
        )

        frictionless_value = sc.add_operator(Map(strategy_frictionless, np.sum, shape=(), dtype=np.float64))

        def _expected_return_risk(x, mu_log, sigma_log):
            # Markowitz internally converts log-return predictions to
            # linear-return moments via the lognormal moment map before
            # optimising; evaluate the resulting weights against the
            # same linear-return moments so the plotted frontier matches
            # the objective that was actually maximised.  Evaluating in
            # log space here would systematically penalise concentrated
            # high-variance portfolios (via the −½σ² drag) and produce
            # a non-monotonic frontier at the high-risk end.
            mask = np.isfinite(x) & (x > 0)
            x = x[mask]
            mu_log = mu_log[mask]
            sigma_log = sigma_log[np.ix_(mask, mask)]
            mu = np.expm1(mu_log + 0.5 * np.diag(sigma_log))
            factor = 1.0 + mu
            sigma = np.outer(factor, factor) * np.expm1(sigma_log)
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
            "sharpe": sc.add_operator(Record(sc.add_operator(SharpeRatio(frictionless_value, rebalance_clock)))),
            "drawdown": sc.add_operator(Record(sc.add_operator(Drawdown(frictionless_value)))),
            "compound": sc.add_operator(Record(sc.add_operator(CompoundReturn(frictionless_value, rebalance_clock)))),
            "frontier": sc.add_operator(Record(frontier_point)),
        }

    return sc, variants, {"index_value": sc.add_operator(Record(index_value))}, rebalance_dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser, include_initial_cash=True)
    args = parser.parse_args()

    data_dir, symbols = validate_data_dir(args)

    sc, variants, handles, rebalance_dates = build_scenario(
        symbols,
        data_dir,
        risk_aversions=RISK_AVERSIONS,
        rebalance_days=args.rebalance_days,
        index_size=args.index_size,
        data_start=resolve_data_start(args.sample_begin, args.begin, args.rebalance_days),
        trading_start=args.begin,
        end=args.end,
    )

    on_flush, progress = make_progress_tracker(
        sc, args.begin, before_desc="Loading samples", after_desc="Running strategy"
    )
    session = sc.run(on_flush=on_flush)
    progress.close()

    # Extract results.
    periods_per_year = 365.0 / args.rebalance_days
    results: dict[float, dict] = {}
    index_value = session.series_view(handles["index_value"]).to_series()
    for delta, variant in variants.items():
        value = session.series_view(variant["value"]).to_series()
        sharpe = session.series_view(variant["sharpe"]).to_series()
        drawdown = session.series_view(variant["drawdown"]).to_series()
        compound = session.series_view(variant["compound"]).to_series()
        frontier = session.series_view(variant["frontier"]).to_dataframe(["exp_return", "exp_risk"])
        results[delta] = {
            "value": value,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "compound": compound,
            "frontier": frontier,
        }

    # Effective trading start: the earliest date on which any Markowitz
    # variant first deviates from `1.0`.  The index baseline runs
    # frictionlessly with `initial_cash=1.0` from the first universe
    # tick, so by the time the strategies start trading it has already
    # drifted away from 1.0; rebasing the index to equal
    # `args.initial_cash` on this anchor keeps the visual comparison
    # fair.  The Markowitz variants all start trading at this anchor
    # (shared predictor and clock), so anchor-based scaling reduces to
    # a plain `* initial_cash` for them.
    strategy_starts = [
        s
        for s in (find_effective_trading_start(r["value"], initial_cash=1.0) for r in results.values())
        if s is not None
    ]
    trading_start = min(strategy_starts) if strategy_starts else None
    for delta, result in results.items():
        result["value_scaled"] = result["value"] * args.initial_cash
    index_value_scaled = scale_to_initial_cash(index_value, args.initial_cash, trading_start)

    for delta, result in results.items():
        compound = result["compound"]
        sharpe = result["sharpe"]
        drawdown = result["drawdown"]
        value_scaled = result["value_scaled"]
        car = ((compound.iloc[-1] + 1) ** periods_per_year - 1) if len(compound) > 0 else 0.0
        sr = sharpe.iloc[-1] * np.sqrt(periods_per_year) if len(sharpe) > 0 else 0.0
        mdd = drawdown.min() if len(drawdown) > 0 else 0.0
        print(
            f"delta={delta:.1f}: final={value_scaled.iloc[-1]:,.0f} CNY, "
            f"annual={car:.2%}, sharpe={sr:.3f}, mdd={mdd:.2%}"
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    cm = plt.colormaps.get("viridis")
    assert cm is not None
    colors = cm(np.linspace(0.0, 1.0, len(results)))

    def draw_rebalances(ax):
        for d in rebalance_dates:
            ax.axvline(d, color="lightgray", linestyle="--", linewidth=0.4, zorder=0)

    # Panel 1 (top-left): Portfolio value
    ax = axes[0, 0]
    ax.set_title("Portfolio value")
    ax.set_ylabel("CNY (10k)")
    draw_rebalances(ax)
    ax.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    ax.plot(
        index_value_scaled.index,
        index_value_scaled / 1e4,
        color="gray",
        linewidth=0.8,
        label=f"Index (top {args.index_size})",
    )
    for (delta, result), color in zip(results.items(), colors):
        ax.plot(
            result["value_scaled"].index,
            result["value_scaled"] / 1e4,
            color=color,
            linewidth=0.8,
            label=f"delta={delta}",
        )
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("Date")

    # Panel 2 (top-right): Efficient frontier over time (annualized).
    # `exp_return` and `exp_risk` are per trading day (predictors train
    # on 1-day forward returns), so scale mean by 252 and std by √252.
    ax = axes[0, 1]
    ax.set_title("Efficient frontier, annualized (at each rebalance)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    trading_days = 252
    ret_scale = trading_days * 100
    risk_scale = np.sqrt(trading_days) * 100

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
        risks = [frontier_by_delta[d].loc[ts, "exp_risk"] * risk_scale for d in sorted_deltas]
        rets = [frontier_by_delta[d].loc[ts, "exp_return"] * ret_scale for d in sorted_deltas]
        ax.plot(risks, rets, color="gray", linewidth=0.4, alpha=alpha * 0.5)
        for d, r, ret in zip(sorted_deltas, risks, rets):
            ax.scatter(r, ret, s=10, color=delta_to_color[d], alpha=alpha, zorder=3)

    for (delta, _), color in zip(results.items(), colors):
        ax.scatter([], [], s=20, color=color, label=f"δ={delta}")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("Expected risk, annualized (%)")
    ax.set_ylabel("Expected return, annualized (%)")

    # Panel 3 (bottom-left): Sharpe ratio
    ax = axes[1, 0]
    ax.set_title("Sharpe ratio annualized (since inception)")
    ax.set_ylabel("Sharpe")
    draw_rebalances(ax)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    for (delta, result), color in zip(results.items(), colors):
        ax.plot(
            result["sharpe"].index,
            result["sharpe"] * np.sqrt(periods_per_year),
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
    draw_rebalances(ax)
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
