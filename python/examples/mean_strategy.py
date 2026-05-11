"""Backtest a linear regression strategy on all A-shares stocks.

The cross-sectional factor set fed to the linear regression is the
canonical one defined in
[`features.py`][python.examples.features.build_features] - shared with
``mean_variance_strategy.py``, ``covariance_gmv.py``, and
``factor_ic.py``.  Three fundamentals are percentile-ranked at every
rebalance; the remaining rolling price/volume factors are passed
through as magnitudes.  NaN factor values stay NaN and are ignored by
the regression.

The training target is the **cross-sectionally de-meaned, winsorized
1-step-forward log return** (winsorize first at the 1st / 99th
percentile, then subtract the per-day cross-sectional mean).  Keeping
magnitudes (instead of ranking the target too) preserves the scale
information pooled OLS needs for well-conditioned coefficients; the
per-day winsorization caps tail leverage; the demean step removes the
common market-drift component so the predictor focuses on the
idiosyncratic spread.

The pipeline consists of four independent, composable operators:

1. **MeanPredictor** - periodically fits a model and predicts future returns.
   Subclass: ``LinearRegression`` (pooled OLS regression).
2. **MeanPortfolio** - converts predicted returns into soft positions.
   Subclass: ``RankLinear`` (rank-linear top-fraction selection).
3. **RandomTrader** - converts soft positions into actual trades
   (lots of 100 shares), deducts transaction fees, and tracks the portfolio.
4. **Metric computation** - post-hoc analysis of the portfolio value series.

Requires ``pip install -e ".[examples]"`` and A-shares market data downloaded
via the crawler.  See ``python -m a_shares_crawler --help`` for configuration
and download instructions.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tradingflow import Scenario
from tradingflow.operators import Map, Record, Stack
from tradingflow.operators.num import Diff, Log, Multiply
from tradingflow.operators.predictors.mean import LinearRegression
from tradingflow.operators.portfolios.mean import RankLinear
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.traders.simple import RandomTrader
from tradingflow.operators.metrics import CompoundReturn, SharpeRatio, Drawdown
from tradingflow.operators.metrics.mean import RegressionCoefficients

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
    initial_cash: float,
    index_size: int,
    data_start: np.datetime64,
    trading_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, np.ndarray]:
    """Build the full backtesting scenario."""

    sc = Scenario()

    # Load per-stock CSVs and stack into the cross-sectional panel.
    # See `per_stock.py` for the canonical data pipeline (same set of
    # sources, fields, and stacking semantics across all four
    # examples).
    stacked = build_stacked(sc, symbols, data_dir, data_start=data_start, end=end)

    num_stocks = len(symbols)
    window = 20

    # Cross-sectional features (canonical factor set; see `common.py`).
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

    # Daily price-limit handles (constant ±10% for now; per-board
    # limits and ST / IPO exceptions are not yet modelled).
    upper_limit, lower_limit = build_price_limits(sc, stacked["close"], num_stocks=num_stocks)

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    rebalance_clock, rebalance_dates = build_rebalance_clock(sc, trading_start, end, rebalance_days)
    universe = build_cap_weighted_universe(
        sc, market_cap, rebalance_clock, num_stocks=num_stocks, index_size=index_size,
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

    soft_positions = sc.add_operator(
        RankLinear(
            universe,
            predicted_returns,
            top_fraction=1.0,
        )
    )

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

    strategy_actual = sc.add_operator(
        RandomTrader(
            soft_positions,
            stacked["close"],
            stacked["adjusts"],
            upper_limit,
            lower_limit,
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

    # Market beta / alpha vs. the cap-weighted index, computed on
    # trading-day log returns of total portfolio value.  `actual_value`
    # and `index_value` both tick on the trading-day pulse driven by
    # `stacked["close"]`, so `Diff(Log(...))` produces aligned daily
    # log returns; `Stack([..., axis=0)` lifts the scalar index log
    # return into a 1-element baseline vector for the regressor.  The
    # regressor adds the intercept column itself, so the emitted
    # coefficient vector has shape (2,): [beta, alpha].
    index_value = sc.add_operator(Map(index, np.sum, shape=(), dtype=np.float64))
    strategy_log_return = sc.add_operator(Diff(sc.add_operator(Log(actual_value))))
    index_log_return = sc.add_operator(Diff(sc.add_operator(Log(index_value))))
    strategy_log_return_series = sc.add_operator(Record(strategy_log_return))
    index_log_return_series = sc.add_operator(Record(sc.add_operator(Stack([index_log_return], axis=0))))
    beta_alpha = sc.add_operator(
        RegressionCoefficients(
            rebalance_clock,
            strategy_log_return_series,
            index_log_return_series,
            max_periods=252,
            min_periods=20,
        )
    )

    return (
        sc,
        {
            "index": sc.add_operator(Record(index)),
            "strategy_frictionless": sc.add_operator(Record(strategy_frictionless)),
            "strategy_actual": sc.add_operator(Record(strategy_actual)),
            "sharpe": sc.add_operator(Record(sharpe)),
            "compound_return": sc.add_operator(Record(compound_ret)),
            "drawdown": sc.add_operator(Record(drawdown)),
            "beta_alpha": sc.add_operator(Record(beta_alpha)),
        },
        rebalance_dates,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser, include_initial_cash=True)
    args = parser.parse_args()

    data_dir, symbols = validate_data_dir(args)

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

    on_flush, progress = make_progress_tracker(sc, args.begin, before_desc="Loading samples", after_desc="Running strategy")
    session = sc.run(on_flush=on_flush)
    progress.close()

    # Extract results.
    index = session.series_view(handles["index"]).to_dataframe(["holdings", "cash"])
    strategy_frictionless = session.series_view(handles["strategy_frictionless"]).to_dataframe(["holdings", "cash"])
    strategy_actual = session.series_view(handles["strategy_actual"]).to_dataframe(["holdings", "cash"])
    sharpe = session.series_view(handles["sharpe"]).to_series()
    compound_return = session.series_view(handles["compound_return"]).to_series()
    drawdown = session.series_view(handles["drawdown"]).to_series()

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

    # Rolling 1-year market beta / alpha vs. the cap-weighted index.
    beta_alpha_view = session.series_view(handles["beta_alpha"])
    beta_alpha_ts = beta_alpha_view.timestamps()
    beta_alpha_vals = beta_alpha_view.values()  # (n_rebalances, 2): [beta, alpha_daily]
    rolling_beta = beta_alpha_vals[:, 0] if len(beta_alpha_vals) else np.empty(0)
    # Log returns are additive, so daily-log alpha annualizes by simple
    # multiplication by trading days per year.
    rolling_alpha_annualized = beta_alpha_vals[:, 1] * 252 if len(beta_alpha_vals) else np.empty(0)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})

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
    draw_rebalances(ax)
    ax.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.4, color="C3")

    ax = axes[3]
    ax.set_title("Rolling 1-year market beta / alpha (vs. cap-weighted index, 252-day window)")
    ax.set_ylabel("Beta")
    ax.set_xlabel("Date")
    draw_rebalances(ax)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    (line_beta,) = ax.plot(beta_alpha_ts, rolling_beta, color="C2", label="Beta")
    ax_alpha = ax.twinx()
    ax_alpha.set_ylabel("Alpha annualized (%)")
    ax_alpha.axhline(0, color="lightgray", linewidth=0.5, linestyle=":")
    (line_alpha,) = ax_alpha.plot(
        beta_alpha_ts, rolling_alpha_annualized * 100, color="C4", linestyle="--", label="Alpha (annualized)"
    )
    ax.legend(handles=[line_beta, line_alpha], loc="upper left", fontsize=8)

    fig.tight_layout()
    plt.show()
