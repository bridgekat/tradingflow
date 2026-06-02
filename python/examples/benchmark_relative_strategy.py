"""Backtest benchmark-relative (index-enhancement) strategies with different tracking-error budgets.

Compares multiple portfolios optimized with the following formulation:

    ```
    maximize:       mu' x
    subject to:     (x - x_bm)' Σ (x - x_bm) <= γ_TE²
                    1' x = 1
                    x >= 0
    ```

where `γ_TE` (tracking-error budget) is varied across runs and `x_bm`
is the cap-weighted universe taken as the benchmark.  All variants
share the same data sources, features, return predictor, and
covariance estimator within a single computation graph; they diverge
only at the portfolio construction and trading stages.  Structurally
parallel to ``mean_variance_strategy.py`` (which sweeps the absolute
variance-penalty `δ` instead of the active-risk budget `γ_TE`).

Tracking-error budgets are specified in *annualised* fraction (e.g.
``0.05`` for a 5\\% tracking-error budget) and converted to *daily*
units (γ_daily = γ_ann / √252) before being passed to the operator,
which works on the daily moments produced by Ridge and Shrinkage.

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

from tradingflow import Scenario, Handle
from tradingflow.operators import Apply, Map, Record
from tradingflow.operators.num import Log, Multiply
from tradingflow.operators.predictors.mean import Ridge
from tradingflow.operators.predictors.variance import Shrinkage
from tradingflow.operators.portfolios.mean_variance import BenchmarkRelative
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.metrics import CompoundReturn, SharpeRatio, Drawdown

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

# Annualised tracking-error budgets.  Tight (2%) / standard (5%) /
# loose (10%) tiers for an enhanced-index mandate.  Same length and
# role as ``RISK_AVERSIONS`` in mean_variance_strategy.py.
TRACKING_ERRORS_ANN = [0.02, 0.05, 0.10]
TRADING_DAYS = 252


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    tracking_errors_ann: list[float],
    rebalance_days: int,
    index_size: int,
    data_start: np.datetime64,
    trading_start: np.datetime64,
    end: np.datetime64,
    noise_alpha: float = 0.0,
    noise_seed: int = 0,
) -> tuple[Scenario, dict, dict, np.ndarray]:
    """Build the full backtesting scenario."""

    sc = Scenario()

    # Load per-stock CSVs and stack into the cross-sectional panel.
    # See `common.py` for the canonical data pipeline.
    stacked = build_stacked(sc, symbols, data_dir, data_start=data_start, end=end)

    num_stocks = len(symbols)
    window = 20

    # Cross-sectional features (canonical factor set; see `common.py`).
    features, features_series = build_features(sc, stacked, num_stocks=num_stocks, window=window)

    # `circ_market_cap` for the cap-weighted universe / benchmark,
    # `log_adj` for the daily log-return target.
    circ_market_cap = sc.add_operator(Multiply(stacked["close"], stacked["circ_shares"]))
    log_adj = sc.add_operator(Log(stacked["adjusted_close"]))

    # Regression target: winsorized daily
    # log returns.  target_offset=1 → feature at day t predicts the
    # return from t to t+1.
    target, target_series, demeaned_series = build_log_return_target(sc, log_adj, num_stocks=num_stocks)

    # Daily price-limit handles (constant ±10% for now).
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

    # Optional noise injection on the mean predictor: mu' = noise_alpha
    # * eps + (1 - noise_alpha) * mu, with eps_i iid N(0, std(mu_finite))
    # per rebalance.  Sigma is NOT perturbed.  Matches the convention in
    # mean_variance_strategy.py.
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
            eps = rng.normal(0.0, sigma, size=mu.shape)
            out[finite] = a * eps[finite] + (1.0 - a) * mu[finite]
            return out

        predicted_returns = sc.add_operator(
            Apply((predicted_returns,), _add_noise, shape=(num_stocks,), dtype=np.float64)
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
    # Multiple BenchmarkRelative variants (one per tracking-error budget)
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
    for gamma_ann in tracking_errors_ann:
        # Convert annualised TE budget to the daily scale used by the
        # operator (matches the 1-day moments produced by Ridge / Shrinkage).
        gamma_daily = float(gamma_ann) / np.sqrt(TRADING_DAYS)

        soft_positions = sc.add_operator(
            BenchmarkRelative(
                universe,
                predicted_returns,
                predicted_covariances,
                bound=gamma_daily,
                long_only=True,
                full_position=True,
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

        def _active_return_risk(x, mu_log, sigma_log, x_bm):
            # Compute predicted active return and predicted tracking
            # error against the cap-weighted benchmark, both in the
            # *linear-return* space that the operator actually optimises
            # over (after the lognormal moment map).  Mirrors the
            # `_expected_return_risk` helper in mean_variance_strategy.py
            # but expressed as benchmark-relative quantities.
            mask = np.isfinite(x) & np.isfinite(x_bm) & np.isfinite(mu_log)
            x = x[mask]
            mu_log = mu_log[mask]
            sigma_log = sigma_log[np.ix_(mask, mask)]
            x_bm_sub = x_bm[mask]
            s = x_bm_sub.sum()
            x_bm_sub = x_bm_sub / s if s > 0 else np.full_like(x_bm_sub, 1.0 / x_bm_sub.size)
            mu_lin = np.expm1(mu_log + 0.5 * np.diag(sigma_log))
            factor = 1.0 + mu_lin
            sigma_lin = np.outer(factor, factor) * np.expm1(sigma_log)
            active = x - x_bm_sub
            exp_active_ret = mu_lin @ active
            exp_te = np.sqrt(max(active @ sigma_lin @ active, 0.0))
            return np.array([exp_active_ret, exp_te])

        frontier_point = sc.add_operator(
            Apply(
                (soft_positions, predicted_returns, predicted_covariances, universe),
                _active_return_risk,
                shape=(2,),
                dtype=np.float64,
            )
        )

        variants[gamma_ann] = {
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
    parser.add_argument(
        "--tracking-errors",
        type=float,
        nargs="+",
        default=None,
        help=f"override the hardcoded annualised TE budget sweep (default {TRACKING_ERRORS_ANN})",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="if set, save the 4-panel figure and per-γ stats JSON to this directory",
    )
    parser.add_argument(
        "--noise-alpha",
        type=float,
        default=0.0,
        help="if > 0, inject Gaussian noise into the mean predictor before optimisation: "
        "mu' = noise_alpha * eps + (1 - noise_alpha) * mu, eps ~ N(0, std(mu)) i.i.d. per stock. "
        "Sigma is not perturbed.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="seed for the noise RNG (deterministic per run)",
    )
    args = parser.parse_args()

    data_dir, symbols = validate_data_dir(args)
    tracking_errors_ann = args.tracking_errors if args.tracking_errors is not None else TRACKING_ERRORS_ANN

    sc, variants, handles, rebalance_dates = build_scenario(
        symbols,
        data_dir,
        tracking_errors_ann=tracking_errors_ann,
        rebalance_days=args.rebalance_days,
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
    periods_per_year = 365.0 / args.rebalance_days
    results: dict[float, dict] = {}
    index_value = session.series_view(handles["index_value"]).to_series()
    for gamma_ann, variant in variants.items():
        value = session.series_view(variant["value"]).to_series()
        sharpe = session.series_view(variant["sharpe"]).to_series()
        drawdown = session.series_view(variant["drawdown"]).to_series()
        compound = session.series_view(variant["compound"]).to_series()
        frontier = session.series_view(variant["frontier"]).to_dataframe(["exp_active_return", "exp_tracking_error"])
        results[gamma_ann] = {
            "value": value,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "compound": compound,
            "frontier": frontier,
        }

    # Effective trading start: the earliest date on which any
    # benchmark-relative variant first deviates from `1.0`.  The index
    # baseline runs frictionlessly with `initial_cash=1.0` from the
    # first universe tick, so by the time the strategies start trading
    # it has already drifted away from 1.0; rebasing the index to equal
    # `args.initial_cash` on this anchor keeps the visual comparison
    # fair.  All BR variants start trading at this anchor (shared
    # predictor and clock), so anchor-based scaling reduces to a plain
    # `* initial_cash` for them.
    strategy_starts = [
        s
        for s in (find_effective_trading_start(r["value"], initial_cash=1.0) for r in results.values())
        if s is not None
    ]
    trading_start = min(strategy_starts) if strategy_starts else None
    for gamma_ann, result in results.items():
        result["value_scaled"] = result["value"] * args.initial_cash
    index_value_scaled = scale_to_initial_cash(index_value, args.initial_cash, trading_start)

    for gamma_ann, result in results.items():
        compound = result["compound"]
        sharpe = result["sharpe"]
        drawdown = result["drawdown"]
        value_scaled = result["value_scaled"]
        car = ((compound.iloc[-1] + 1) ** periods_per_year - 1) if len(compound) > 0 else 0.0
        sr = sharpe.iloc[-1] * np.sqrt(periods_per_year) if len(sharpe) > 0 else 0.0
        mdd = drawdown.min() if len(drawdown) > 0 else 0.0
        print(
            f"gamma_ann={gamma_ann:.3f}: final={value_scaled.iloc[-1]:,.0f} CNY, "
            f"annual={car:.2%}, sharpe={sr:.3f}, mdd={mdd:.2%}"
        )

    # ------------------------------------------------------------------
    # Plot: portfolio value vs benchmark only, restricted to the trading
    # period.  The recorded NAV series begin at the sample-begin date
    # (2005-01-01) and stay flat at `initial_cash` through the warm-up
    # until the first rebalance trades (~`begin`); we trim everything
    # before the effective trading start so only the live segment shows.
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 7))

    cm = plt.colormaps.get("viridis")
    assert cm is not None
    colors = cm(np.linspace(0.0, 1.0, len(results)))

    def _trim(s):
        return s[s.index >= trading_start] if trading_start is not None else s

    ax.set_title(f"Portfolio value vs benchmark (top {args.index_size})")
    ax.set_ylabel("CNY (10k)")
    ax.set_xlabel("Date")
    for d in rebalance_dates:
        ax.axvline(d, color="lightgray", linestyle="--", linewidth=0.4, zorder=0)
    ax.axhline(args.initial_cash / 1e4, color="gray", linewidth=0.5, linestyle="--", label="Initial")
    idx_trim = _trim(index_value_scaled)
    ax.plot(idx_trim.index, idx_trim / 1e4, color="gray", linewidth=1.2, label=f"Index (top {args.index_size})")
    for (gamma_ann, result), color in zip(results.items(), colors):
        v = _trim(result["value_scaled"])
        ax.plot(v.index, v / 1e4, color=color, linewidth=1.0, label=f"BR γ_ann={gamma_ann:.0%}")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Optional: save figure and per-γ stats JSON
    # ------------------------------------------------------------------

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        tag = f"_noise{args.noise_alpha:.1f}" if args.noise_alpha > 0.0 else ""
        fig.savefig(
            args.save_dir / f"benchmark_relative_strategy_{args.index_size}{tag}.png",
            dpi=150,
            bbox_inches="tight",
        )

        def _stats(value: pd.Series, index: pd.Series, td: int = TRADING_DAYS) -> dict | None:
            mask = ~np.isclose(value.to_numpy(), value.iloc[0])
            if not mask.any():
                return None
            start = int(mask.argmax())
            df = pd.concat([value.iloc[start:].rename("s"), index.rename("i")], axis=1).dropna()
            df = df[(df["s"] > 0) & (df["i"] > 0)]
            if len(df) < 10:
                return None
            s = df["s"].to_numpy()
            years = len(df) / td
            cagr = (s[-1] / s[0]) ** (1.0 / years) - 1.0
            rs = np.diff(np.log(s))
            ann_vol = float(rs.std(ddof=0) * np.sqrt(td))
            sharpe = (rs.mean() * td) / ann_vol if ann_vol > 0 else float("nan")
            peak = np.maximum.accumulate(s)
            mdd = float((s / peak - 1.0).min())
            ri = np.diff(np.log(df["i"].to_numpy()))
            if ri.std(ddof=0) > 0:
                beta = float(np.cov(rs, ri, ddof=0)[0, 1] / np.var(ri, ddof=0))
                alpha_daily = float(rs.mean() - beta * ri.mean())
                alpha_ann = alpha_daily * td
                # Realised tracking error = std of (strategy log return -
                # index log return).  IR = annualised mean active return
                # over realised annualised TE.
                active = rs - ri
                te_ann = float(active.std(ddof=0) * np.sqrt(td))
                ir = (active.mean() * td) / te_ann if te_ann > 0 else float("nan")
            else:
                beta = alpha_ann = te_ann = ir = float("nan")
            return {
                "cagr": float(cagr), "ann_vol": ann_vol, "sharpe": float(sharpe),
                "mdd": mdd, "beta": beta, "alpha_ann": float(alpha_ann),
                "tracking_error_ann": float(te_ann), "information_ratio": float(ir),
                "days": int(len(df)),
            }

        stats = {
            "index_size": args.index_size,
            "rebalance_days": args.rebalance_days,
            "tracking_errors_ann": list(tracking_errors_ann),
            "noise_alpha": float(args.noise_alpha),
            "noise_seed": int(args.noise_seed),
            "begin": str(args.begin),
            "end": str(args.end),
            "index": _stats(index_value, index_value),
            "br": {f"{g:.3f}": _stats(result["value"], index_value) for g, result in results.items()},
        }
        stats_path = args.save_dir / f"stats_benchmark_relative_strategy_k{args.index_size}{tag}.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"Saved figure and stats to {args.save_dir}")

    plt.show()
