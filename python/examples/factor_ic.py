"""Evaluate factor IC on all A-shares stocks.

The cross-sectional factor set is the canonical one defined in
[`features.py`][python.examples.features.build_features] - shared with
``mean_strategy.py``, ``mean_variance_strategy.py``, and
``covariance_gmv.py``.  Three fundamentals are percentile-ranked at
every rebalance; the remaining rolling price/volume factors are passed
through as magnitudes (NaN-preserving, so missing stocks stay NaN
rather than clustering at the top rank).

The realised target fed to the evaluator is the cross-sectionally
de-meaned daily log return, with winsorization applied first to clip
outliers at the 1st / 99th percentile of the raw cross-section.
Combined with the percentile-ranked factors, the reported IC is the
per-period cross-sectional correlation between ranked factors and
realised log-return magnitudes: a rank-vs.-magnitude measure that
penalises factors whose top-rank stocks consistently fail to deliver
large positive returns, rather than the pure Spearman rank correlation
a rank-on-rank evaluator would report.  IC is invariant under per-row
additive shifts, so the demean step leaves the metric unchanged - it
is included here to keep the target construction identical across the
mean-forecast examples (`mean_strategy.py`, `mean_variance_strategy.py`,
`covariance_gmv.py`), where it does help the predictor learn the
idiosyncratic component instead of the time-varying market drift.

The IC evaluator is restricted to the cap-weighted top-``index_size``
universe (same construction as ``mean_strategy.py``): out-of-universe
stocks are NaN-masked at each rebalance so they don't dilute the
cross-sectional rank correlation.

Two figures are produced: cumulative IC by factor over time, and the
pooled Pearson correlation matrix across (date, stock) pairs of the
feature panel.  The correlation heatmap exposes rank-vs-raw block
structure and residual cross-factor overlap that would drive
multicollinearity in any downstream linear model.

Requires ``pip install -e ".[examples]"`` and A-shares market data downloaded
via the crawler.  See ``python -m a_shares_crawler --help`` for configuration
and download instructions.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tradingflow import Scenario, Handle
from tradingflow.operators import Apply, Map, Record, Resample
from tradingflow.operators.num import Log, Multiply, Subtract
from tradingflow.operators.portfolios.mean import RankBucket
from tradingflow.operators.traders import Benchmark
from tradingflow.operators.metrics.mean import InformationCoefficient

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
    window: int,
    index_size: int,
    initial_cash: float,
    data_start: np.datetime64,
    eval_start: np.datetime64,
    end: np.datetime64,
    *,
    decile_analysis: bool = False,
) -> tuple[Scenario, dict, Handle, Handle, list[str], np.ndarray, dict | None]:
    """Build the factor IC evaluation scenario.

    Returns
    -------
    sc
        The constructed [`Scenario`][tradingflow.Scenario].
    eval_handles
        Per-feature `Record(InformationCoefficient(...))` handles, keyed
        by feature name.
    features_handle
        `Record` of the trading-day-aligned ``(num_stocks, n_features)``
        feature panel.  Used downstream to compute the cross-feature
        correlation matrix and decile portfolio returns.
    target_handle
        `Record` of the per-stock realized log returns (winsorized at
        the 1st / 99th cross-sectional percentile).  Aligned to
        ``features_handle`` row-by-row.
    feature_names
        Column names matching the last axis of ``features_handle``.
    rebalance_dates
        Dates at which the rebalance clock fires.
    decile_handles
        ``None`` when ``decile_analysis=False``.  Otherwise a dict
        with a ``"features"`` key holding a per-feature dict with
        ``"decile_values"`` (list of 10 recorded scalar handles,
        bottom -> top), ``"long_short_value"`` (recorded scalar
        handle for the long-short portfolio -- top-minus-bottom by
        default, bottom-minus-top for features whose IC is empirically
        negative), and ``"inverted"`` (boolean indicating which
        direction was used).
    """

    sc = Scenario()

    # Load per-stock CSVs and stack into the cross-sectional panel.
    # See `per_stock.py` for the canonical data pipeline.
    stacked = build_stacked(sc, symbols, data_dir, data_start=data_start, end=end)

    num_stocks = len(symbols)

    # Cross-sectional features (canonical factor set; see `features.py`).
    features, features_series = build_features(sc, stacked, num_stocks=num_stocks, window=window)

    # `market_cap` for the cap-weighted universe, `log_adj` for the
    # daily log-return target.  Both are also built inside
    # `build_features`; the duplication is intentional to keep the
    # helper's return signature minimal.
    market_cap = sc.add_operator(Multiply(stacked["close"], stacked["total_shares"]))
    log_adj = sc.add_operator(Log(stacked["adjusted_close"]))

    # Winsorize-then-demean target.  IC is rank- (and Pearson-)
    # invariant under per-row additive shifts, so the demean step
    # leaves the reported IC unchanged - the chain is here purely to
    # keep the target construction identical across examples.
    target, target_series = build_demeaned_log_return_target(sc, log_adj, num_stocks=num_stocks)

    # Daily price-limit handles (constant ±10% for now).  Only the
    # optional decile / long-short Benchmark block below actually
    # consumes them, but they live here so the trader construction
    # mirrors the other examples.
    upper_limit, lower_limit = build_price_limits(sc, stacked["close"], num_stocks=num_stocks)

    # ------------------------------------------------------------------
    # Universe and rebalance clock
    # ------------------------------------------------------------------
    #
    # Cap-weighted top-`index_size` universe used both to restrict the
    # IC evaluator to the working universe (so out-of-universe stocks
    # don't dilute the cross-sectional rank correlation) and as the
    # soft-position / rebalance signal for the optional decile
    # portfolio analysis below.

    rebalance_clock, rebalance_dates = build_rebalance_clock(sc, eval_start, end, rebalance_days)
    universe = build_cap_weighted_universe(
        sc,
        market_cap,
        rebalance_clock,
        num_stocks=num_stocks,
        index_size=index_size,
    )

    # ------------------------------------------------------------------
    # IC evaluation
    # ------------------------------------------------------------------

    eval_handles: dict[str, Handle] = {}
    for name, feature in features.items():
        aligned = sc.add_operator(Resample(rebalance_clock, feature))
        # NaN-mask out-of-universe stocks at each rebalance.  IC's
        # `np.isfinite(s)` filter then drops these slots from the
        # cross-sectional rank correlation, so the reported IC reflects
        # only the cap-weighted top-`index_size` universe.
        masked = sc.add_operator(
            Apply(
                (aligned, universe),
                lambda f, u: np.where(u > 0, f, np.nan),
                shape=(num_stocks,),
                dtype=np.float64,
            )
        )
        metric = sc.add_operator(InformationCoefficient(masked, target))
        eval_handles[name] = sc.add_operator(Record(metric))

    # ------------------------------------------------------------------
    # Per-feature decile / long-short portfolios (optional)
    # ------------------------------------------------------------------
    #
    # Builds, for each factor:
    #
    # 1. 10 `RankBucket` operators carving the universe into deciles by
    #    factor rank, each fed to a frictionless `Benchmark` to track
    #    the per-decile portfolio value (rebalanced on the same clock
    #    as the IC evaluator, marked-to-market every trading day).
    # 2. A dollar-neutral long-short portfolio constructed as
    #    `Subtract(top_decile_weights, bottom_decile_weights)` (sum 0,
    #    gross 2x) with its own `Benchmark`.  For features in
    #    `inverted_features` the legs are swapped (long D1, short D10)
    #    so the long-short curve trends *up* under the empirical
    #    direction of the alpha (small-cap premium, mean-reverting
    #    momentum, low-volatility / low-turnover / low-volume-ratio
    #    anomalies).
    #
    # The `universe` handle constructed above doubles as the rebalance
    # signal / mask for `RankBucket`.

    decile_handles: dict | None = None
    if decile_analysis:
        # Features whose long-short should be inverted (long D1 - short
        # D10) because their IC is negative -- the bottom decile is the
        # outperforming one and shorting the top decile completes the
        # arbitrage.  Names must match keys in the `features` dict.
        inverted_features: set[str] = {
            "rank_market_cap",
            f"momentum_ma_{window}",
            f"volatility_{window}",
            f"turnover_ma_{window}",
            f"volume_ratio_{window}",
        }

        n_deciles = 10
        per_feature: dict[str, dict] = {}
        for name, feature in features.items():
            decile_weights = [
                sc.add_operator(RankBucket(universe, feature, low=d / n_deciles, high=(d + 1) / n_deciles))
                for d in range(n_deciles)
            ]
            decile_value_records: list[Handle] = []
            for w in decile_weights:
                bench = sc.add_operator(
                    Benchmark(
                        w,
                        stacked["close"],
                        stacked["adjusts"],
                        upper_limit,
                        lower_limit,
                        initial_cash=initial_cash,
                        use_adjusts=True,
                    )
                )
                total = sc.add_operator(Map(bench, np.sum, shape=(), dtype=np.float64))
                decile_value_records.append(sc.add_operator(Record(total)))

            inverted = name in inverted_features
            long_leg, short_leg = (
                (decile_weights[0], decile_weights[-1])
                if inverted
                else (decile_weights[-1], decile_weights[0])
            )
            ls_weights = sc.add_operator(Subtract(long_leg, short_leg))
            ls_bench = sc.add_operator(
                Benchmark(
                    ls_weights,
                    stacked["close"],
                    stacked["adjusts"],
                    upper_limit,
                    lower_limit,
                    initial_cash=initial_cash,
                    use_adjusts=True,
                )
            )
            ls_value = sc.add_operator(Map(ls_bench, np.sum, shape=(), dtype=np.float64))

            per_feature[name] = {
                "decile_values": decile_value_records,
                "long_short_value": sc.add_operator(Record(ls_value)),
                "inverted": inverted,
            }

        decile_handles = {"features": per_feature}

    return (
        sc,
        eval_handles,
        features_series,
        target_series,
        list(features.keys()),
        rebalance_dates,
        decile_handles,
    )


def plot_individual_feature_returns(
    session,
    decile_handles: dict,
    initial_cash: float,
) -> None:
    """Plot per-feature decile / long-short cumulative log return figures.

    Reads recorded portfolio-value series from the runtime session and
    renders one figure per feature with two stacked panels sharing the
    X axis: cumulative log returns of the 10 decile portfolios and of
    the long-top / short-bottom portfolio.

    Returns are computed as ``log(value / initial_cash)``; for a
    long-short portfolio with sum-zero weights, ``value`` stays close
    to ``initial_cash`` while the spread P&L drives the curve.  Once a
    portfolio's NAV crosses zero the trader force-liquidates and emits
    ``(0, 0)``; the helper plots those ticks as `NaN` so the curve
    breaks visibly at the blowup date.
    """
    decile_cmap = plt.cm.RdYlGn  # type: ignore

    def cum_log_or_nan(values: np.ndarray) -> np.ndarray:
        """``log(values / initial_cash)``, with `NaN` wherever the
        portfolio NAV has gone to zero or below.  The trader operators
        force-liquidate (and emit `(0, 0)`) on a margin call, so the
        post-blowup tail of the time series is identically zero -
        matplotlib draws `NaN` segments as gaps, which is the
        intended "this portfolio stopped existing" reading."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log(values / initial_cash), np.nan)

    for name, handles in decile_handles["features"].items():
        decile_value_series = [session.series_view(h) for h in handles["decile_values"]]
        ls_value_series = session.series_view(handles["long_short_value"])
        inverted = handles.get("inverted", False)

        decile_ts = pd.DatetimeIndex(decile_value_series[0].timestamps())
        decile_cum_log = np.column_stack([cum_log_or_nan(s.values()) for s in decile_value_series])
        ls_ts = pd.DatetimeIndex(ls_value_series.timestamps())
        ls_cum_log = cum_log_or_nan(ls_value_series.values())

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

        n_deciles = decile_cum_log.shape[1]
        ax = axes[0]
        ax.set_title(f"Decile portfolios (equal-weighted within decile, cap-weighted universe) - {name}")
        ax.set_ylabel("Cumulative log return")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        for d in range(n_deciles):
            ax.plot(
                decile_ts,
                decile_cum_log[:, d],
                color=decile_cmap(d / (n_deciles - 1)),
                linewidth=1.0,
                label=f"D{d + 1}",
            )
        ax.legend(ncol=n_deciles, fontsize=7, loc="upper left")

        ax = axes[1]
        ls_direction = "Long bottom 10% / short top 10%" if inverted else "Long top 10% / short bottom 10%"
        ax.set_title(f"{ls_direction} - {name}")
        ax.set_ylabel("Cumulative log return")
        ax.set_xlabel("Date")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.plot(ls_ts, ls_cum_log, color="C0", linewidth=1.0)

        fig.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser, include_initial_cash=True)
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="rolling window (in samples) for windowed features; defaults to --rebalance-days",
    )
    parser.add_argument(
        "--plot-individual-feature-returns",
        action="store_true",
        help=(
            "for every factor, open a figure with two stacked panels: cumulative log returns of "
            "the 10 decile portfolios (cap-weighted top-N universe, rebalanced on the IC clock) "
            "and of the long-top-10%% / short-bottom-10%% portfolio"
        ),
    )
    args = parser.parse_args()

    data_dir, symbols = validate_data_dir(args)

    window = args.window if args.window is not None else args.rebalance_days
    (
        sc,
        eval_handles,
        features_handle,
        target_handle,
        feature_names,
        rebalance_dates,
        decile_handles,
    ) = build_scenario(
        symbols,
        data_dir,
        rebalance_days=args.rebalance_days,
        window=window,
        index_size=args.index_size,
        initial_cash=args.initial_cash,
        data_start=resolve_data_start(args.sample_begin, args.begin, max(args.rebalance_days, window)),
        eval_start=args.begin,
        end=args.end,
        decile_analysis=args.plot_individual_feature_returns,
    )

    on_flush, progress = make_progress_tracker(sc, args.begin, after_desc="Evaluating IC")
    session = sc.run(on_flush=on_flush)
    progress.close()

    # ------------------------------------------------------------------
    # Extract results with prediction timestamps
    # ------------------------------------------------------------------

    eval_data = {}
    for name, handle in eval_handles.items():
        values = session.series_view(handle).values()
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
    # Plot cumulative IC
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("Cumulative IC by factor")
    ax.set_ylabel("Cumulative IC")
    ax.set_xlabel("Date")
    for d in rebalance_dates:
        ax.axvline(d, color="lightgray", linestyle="--", linewidth=0.4, zorder=0)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    for i, (name, series) in enumerate(eval_data.items()):
        ax.plot(
            series.index,
            series.fillna(0.0).cumsum(),
            label=name,
            color=f"C{i}",
            linestyle="--" if "rank" in name else "-",
        )

    ax.legend(fontsize=9)
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Plot feature correlation matrix
    # ------------------------------------------------------------------

    # Pool every (date, stock) observation of the feature panel and
    # compute the Pearson correlation matrix (NaN handled pairwise).
    # Raw factors mix scale (e.g. log market cap) with bounded ratios
    # (percentile ranks); pooled Pearson on the panel still highlights
    # the block structure between each raw factor and its rank twin and
    # the residual cross-factor overlap that would drive multicollinearity
    # in any downstream linear model.
    feature_panel = session.series_view(features_handle).values()
    n_features = feature_panel.shape[-1]
    feature_df = pd.DataFrame(feature_panel.reshape(-1, n_features), columns=feature_names)
    corr = feature_df.corr(method="pearson")

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    im = ax_corr.imshow(corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax_corr.set_title("Feature correlation (Pearson, pooled across (date, stock) pairs)")
    ax_corr.set_xticks(range(n_features))
    ax_corr.set_yticks(range(n_features))
    ax_corr.set_xticklabels(feature_names, rotation=45, ha="right")
    ax_corr.set_yticklabels(feature_names)
    for i in range(n_features):
        for j in range(n_features):
            v = corr.values[i, j]
            ax_corr.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if abs(v) > 0.5 else "black",
                fontsize=8,
            )
    fig_corr.colorbar(im, ax=ax_corr)
    fig_corr.tight_layout()

    # ------------------------------------------------------------------
    # Per-feature decile / long-short cumulative log return figures
    # ------------------------------------------------------------------

    if decile_handles is not None:
        plot_individual_feature_returns(
            session=session,
            decile_handles=decile_handles,
            initial_cash=args.initial_cash,
        )

    plt.show()
