"""Backtest a linear regression strategy on all A-shares stocks.

The cross-sectional factor set fed to the linear regression is the
canonical one defined in
[`features.py`][python.examples.features.build_features] - shared with
``mean_variance_strategy.py``, ``covariance_gmv.py``, and
``factor_ic.py``.  Three fundamentals are percentile-ranked at every
rebalance; the remaining rolling price/volume factors are passed
through as magnitudes.  NaN factor values stay NaN and are ignored by
the regression.

The training target is the **winsorized 1-step-forward log return**
(winsorize at the 1st / 99th percentile of the cross-section).  Keeping
magnitudes (instead of ranking the target too) preserves the scale
information pooled regression needs for well-conditioned coefficients;
the per-day winsorization caps tail leverage.

The pipeline consists of four independent, composable operators:

1. **MeanPredictor** - periodically fits a model and predicts future returns.
   Subclass: ``Ridge`` (pooled L2-penalised regression on
   pool-standardised features; ``alpha=1.0`` under our
   ``(1/m) ||y - Xβ||² + α ||β||²`` convention gives sample-size-invariant
   moderate shrinkage - roughly ~50% damping of each coefficient -
   throughout the backtest, regardless of how the pooled sample count
   grows from rebalance to rebalance).
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
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tradingflow import Scenario
from tradingflow.operators import Apply, Map, Record, Stack
from tradingflow.operators.num import Diff, Log, Multiply
from tradingflow.operators.predictors.mean import Ridge
from tradingflow.operators.portfolios.mean import RankLinear
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.traders.simple import RandomTrader
from tradingflow.operators.metrics import CompoundReturn, SharpeRatio, Drawdown
from tradingflow.operators.metrics.mean import RegressionCoefficients

from common import (
    add_common_arguments,
    build_cap_weighted_universe,
    build_log_return_target,
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


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    rebalance_days: int,
    initial_cash: float,
    index_size: int,
    data_start: np.datetime64,
    trading_start: np.datetime64,
    end: np.datetime64,
    noise_alpha: float = 0.0,
    noise_seed: int = 0,
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

    # `circ_market_cap` for the cap-weighted universe, `log_adj` for the
    # daily log-return target.  Both are also built inside
    # `build_features`; the duplication is intentional to keep the
    # helper's return signature minimal.
    circ_market_cap = sc.add_operator(Multiply(stacked["close"], stacked["circ_shares"]))
    log_adj = sc.add_operator(Log(stacked["adjusted_close"]))

    # Regression target: winsorized daily
    # log returns.  The predictor pairs feature[i] with
    # target[i + target_offset] - here target_offset=1, so feature at
    # day t predicts the return from t to t+1.
    target, target_series, demeaned_series = build_log_return_target(sc, log_adj, num_stocks=num_stocks)

    # Daily price-limit handles (constant ±10% for now; per-board
    # limits and ST / IPO exceptions are not yet modelled).
    upper_limit, lower_limit = build_price_limits(sc, stacked["close"], num_stocks=num_stocks)

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    rebalance_clock, rebalance_dates = build_rebalance_clock(sc, trading_start, end, rebalance_days)
    universe = build_cap_weighted_universe(
        sc,
        circ_market_cap,
        rebalance_clock,
        num_stocks=num_stocks,
        index_size=index_size,
    )

    predicted_returns = sc.add_operator(
        Ridge(
            universe,
            features_series,
            demeaned_series,
            alpha=1.0,
            universe_size=index_size,
            target_offset=1,
            min_periods=100,
            verbose=True,
        ),
    )

    # Optional noise injection on the mean predictor:
    # mu' = noise_alpha * rho + (1 - noise_alpha) * mu,
    # rho_i iid ~ N(0, sigma) with sigma = cross-sectional std of the
    # finite entries of mu at the current rebalance.  NaN entries are
    # preserved (out-of-universe / insufficient-data stocks stay NaN).
    if noise_alpha > 0.0:
        rng = np.random.default_rng(noise_seed)
        a = float(noise_alpha)

        def _add_noise(mu: np.ndarray) -> np.ndarray:
            out = mu.copy()
            finite = np.isfinite(mu)
            if not finite.any():
                return out
            sigma = float(np.std(mu[finite]))
            if sigma <= 0.0:
                return out
            rho = rng.normal(0.0, sigma, size=mu.shape)
            out[finite] = a * rho[finite] + (1.0 - a) * mu[finite]
            return out

        predicted_returns = sc.add_operator(
            Apply((predicted_returns,), _add_noise, shape=(num_stocks,), dtype=np.float64)
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
            initial_cash=1.0,
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
            initial_cash=1.0,
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


def _summary_stats(value: pd.Series, index: pd.Series, trading_days: int = 252) -> dict | None:
    """Annualised stats over the effective trading window vs an index baseline.

    Both `value` and `index` are NAV series at any common cadence (the
    Benchmark/RandomTrader records tick daily).  We restrict to the
    window from the first NAV deviation onwards, align on common dates,
    drop non-positive NAVs (post-blowup zeros from the bankruptcy guard),
    and compute CAGR, annualised volatility, Sharpe (rf=0), max drawdown,
    and full-sample beta / annualised Jensen alpha on daily log returns
    versus the index.
    """

    mask = ~np.isclose(value.to_numpy(), value.iloc[0])
    if not mask.any():
        return None
    start = int(mask.argmax())
    df = pd.concat([value.iloc[start:].rename("s"), index.rename("i")], axis=1).dropna()
    df = df[(df["s"] > 0) & (df["i"] > 0)]
    if len(df) < 10:
        return None
    s = df["s"].to_numpy()
    years = len(df) / trading_days
    cagr = (s[-1] / s[0]) ** (1.0 / years) - 1.0
    rs = np.diff(np.log(s))
    ann_vol = float(rs.std(ddof=0) * np.sqrt(trading_days))
    sharpe = (rs.mean() * trading_days) / ann_vol if ann_vol > 0 else float("nan")
    peak = np.maximum.accumulate(s)
    mdd = float((s / peak - 1.0).min())
    ri = np.diff(np.log(df["i"].to_numpy()))
    if ri.std(ddof=0) > 0:
        beta = float(np.cov(rs, ri, ddof=0)[0, 1] / np.var(ri, ddof=0))
        alpha_daily = float(rs.mean() - beta * ri.mean())
        alpha_ann = alpha_daily * trading_days
    else:
        beta = alpha_ann = float("nan")
    return {
        "cagr": float(cagr), "ann_vol": ann_vol, "sharpe": float(sharpe),
        "mdd": mdd, "beta": beta, "alpha_ann": float(alpha_ann),
        "days": int(len(df)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser, include_initial_cash=True)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="if set, save the 4-panel figure and frictionless/actual stats JSON to this directory",
    )
    parser.add_argument(
        "--noise-alpha",
        type=float,
        default=0.0,
        help="if > 0, inject Gaussian noise into the mean predictor before the portfolio: "
        "mu' = noise_alpha * rho + (1 - noise_alpha) * mu, rho ~ N(0, std(mu)) i.i.d. per stock",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="seed for the noise RNG (deterministic per run)",
    )
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
        noise_alpha=args.noise_alpha,
        noise_seed=args.noise_seed,
    )

    on_flush, progress = make_progress_tracker(
        sc, args.begin, before_desc="Loading samples", after_desc="Running strategy"
    )
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

    # Effective trading start: the first date on which the strategy NAV
    # first deviates from `initial_cash`.  The index baseline runs
    # frictionlessly with `initial_cash=1.0` from the very first
    # universe tick, so by the time the strategy starts trading it has
    # already moved away from 1.0; rebasing the index to equal
    # `args.initial_cash` on this date keeps the visual comparison
    # fair.  The frictionless strategy starts trading at the same date
    # as the actual one (shared soft positions), so anchor-based
    # scaling reduces to a plain `* initial_cash` for it.
    trading_start = find_effective_trading_start(total_value, initial_cash=args.initial_cash)
    index_total_scaled = scale_to_initial_cash(index.sum(axis=1), args.initial_cash, trading_start)
    strategy_frictionless_total_scaled = strategy_frictionless.sum(axis=1) * args.initial_cash

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
    # Plot: portfolio value vs benchmark only, restricted to the trading
    # period.  The recorded NAV series begin at the sample-begin date
    # (2005-01-01) and stay flat at `initial_cash` through the warm-up
    # until the first rebalance trades (~`begin`); we trim everything
    # before the effective trading start so only the live segment shows.
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 7))

    def _trim(s):
        return s[s.index >= trading_start] if trading_start is not None else s

    ax.set_title(f"Portfolio value vs benchmark (top {args.index_size})")
    ax.set_ylabel("CNY (10k)")
    ax.set_xlabel("Date")
    for d in rebalance_dates:
        ax.axvline(d, color="lightgray", linestyle="--", linewidth=0.4, zorder=0)
    ax.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    idx_trim = _trim(index_total_scaled)
    ax.plot(idx_trim.index, idx_trim / 1e4, color="gray", linewidth=1.2, label=f"Index (top {args.index_size})")
    fric_trim = _trim(strategy_frictionless_total_scaled)
    ax.plot(fric_trim.index, fric_trim / 1e4, color="C0", linestyle="--", linewidth=1.0, label="Strategy (frictionless)")
    act_trim = _trim(strategy_actual.sum(axis=1))
    ax.plot(act_trim.index, act_trim / 1e4, color="C0", linewidth=1.0, label="Strategy (actual)")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Optional: save figure and frictionless/actual stats JSON
    # ------------------------------------------------------------------

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        tag = f"_noise{args.noise_alpha:.1f}" if args.noise_alpha > 0.0 else ""
        fig.savefig(args.save_dir / f"mean_strategy_{args.index_size}{tag}.png", dpi=150, bbox_inches="tight")
        # Build pandas value series indexed by tick timestamp.
        index_value_series = index.sum(axis=1)
        frictionless_series = strategy_frictionless.sum(axis=1)
        actual_series = strategy_actual.sum(axis=1)
        stats = {
            "index_size": args.index_size,
            "rebalance_days": args.rebalance_days,
            "noise_alpha": float(args.noise_alpha),
            "noise_seed": int(args.noise_seed),
            "begin": str(args.begin),
            "end": str(args.end),
            "index": _summary_stats(index_value_series, index_value_series),
            "mean_only_frictionless": _summary_stats(frictionless_series, index_value_series),
            "mean_only_actual": _summary_stats(actual_series, index_value_series),
        }
        stats_path = args.save_dir / f"stats_mean_strategy_k{args.index_size}{tag}.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"Saved figure and stats to {args.save_dir}")

    plt.show()
