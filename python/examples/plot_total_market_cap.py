"""Plot the total circulating market cap and returns curve of a cap-weighted
A-shares index.

At every rebalance, the top ``--index-size`` A-shares stocks by circulating
market cap (``close * shares.circulating``) form the index universe; weights
are proportional to circulating market cap and renormalised to 1, matching
the selection / weighting convention of CSI / SSE / SZSE indices (which use
free-float-adjusted cap; circulating cap is the closest proxy available in
the crawler data).  Two views of the same index are plotted:

1. **Circulating market cap of the index** -- the sum, across the current
   index constituents, of per-stock circulating market cap
   (``close * shares.circulating``).  Updates every trading day as
   prices move; the constituent set is held fixed between rebalances.
2. **Index returns curve** -- a frictionless :class:`Benchmark` is wired
   on the same cap-weighted universe with unit cash and adjusted prices,
   so its NAV is the total-return curve of the cap-weighted index, with
   the same one-tick MOC execution semantics used by the strategy
   examples.

Requires ``pip install -e ".[examples]"`` and A-shares market data downloaded
via the crawler.  See ``python -m a_shares_crawler --help`` for configuration
and download instructions.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tradingflow import Scenario
from tradingflow.operators import Apply, Map, Record, Resample
from tradingflow.operators.num import Multiply
from tradingflow.operators.traders import Benchmark

from common import (
    add_common_arguments,
    build_cap_weighted_universe,
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
    data_start: np.datetime64,
    trading_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, np.ndarray]:
    """Build a scenario that tracks an A-shares cap-weighted index."""

    sc = Scenario()

    # Per-stock canonical handles (close, circ_shares, total_shares, adjusts, ...).
    stacked = build_stacked(sc, symbols, data_dir, data_start=data_start, end=end)
    num_stocks = len(symbols)

    # Per-stock circulating market cap, ticking every trading day.  Used
    # both to rank the universe (matching CSI / SSE / SZSE methodology,
    # which selects and weights by free-float / circulating cap rather
    # than total cap) and to plot the index's summed market cap.
    circ_market_cap = sc.add_operator(Multiply(stacked["close"], stacked["circ_shares"]))

    rebalance_clock, rebalance_dates = build_rebalance_clock(sc, trading_start, end, rebalance_days)
    universe = build_cap_weighted_universe(
        sc, circ_market_cap, rebalance_clock, num_stocks=num_stocks, index_size=index_size,
    )

    # Re-emit the rebalance-day universe weights onto the daily close pulse so
    # the constituent set is held fixed between rebalances while prices move.
    daily_universe = sc.add_operator(Resample(stacked["close"], universe))

    # Sum the circulating market cap of the current index constituents.
    index_circ_market_cap = sc.add_operator(
        Apply(
            (daily_universe, circ_market_cap),
            lambda u, c: np.nansum(np.where(u > 0, c, 0.0)),
            shape=(),
            dtype=np.float64,
        )
    )

    # Frictionless cap-weighted index NAV with unit cash and dividend
    # reinvestment, matching the strategy examples' baseline.
    upper_limit, lower_limit = build_price_limits(sc, stacked["close"], num_stocks=num_stocks)
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

    return (
        sc,
        {
            "index_circ_market_cap": sc.add_operator(Record(index_circ_market_cap)),
            "index_value": sc.add_operator(Record(index_value)),
        },
        rebalance_dates,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args()

    data_dir, symbols = validate_data_dir(args)

    sc, handles, rebalance_dates = build_scenario(
        symbols,
        data_dir,
        rebalance_days=args.rebalance_days,
        index_size=args.index_size,
        data_start=resolve_data_start(args.sample_begin, args.begin, args.rebalance_days),
        trading_start=args.begin,
        end=args.end,
    )

    on_flush, progress = make_progress_tracker(
        sc, args.begin, before_desc="Loading samples", after_desc="Running scenario"
    )
    session = sc.run(on_flush=on_flush)
    progress.close()

    # Extract results.
    index_circ_market_cap = session.series_view(handles["index_circ_market_cap"]).to_series()
    index_value = session.series_view(handles["index_value"]).to_series()

    # Restrict to the trading window (drop any warmup before `args.begin`).
    begin_ts = np.datetime64(args.begin, "ns")
    index_circ_market_cap = index_circ_market_cap[index_circ_market_cap.index >= begin_ts]
    index_value = index_value[index_value.index >= begin_ts]

    n = len(index_circ_market_cap)
    if n == 0:
        raise SystemExit("No data produced.")

    print(
        f"{n} trading days, {index_circ_market_cap.index[0].date()} "
        f"to {index_circ_market_cap.index[-1].date()}"
    )
    print(
        f"Index circulating market cap: "
        f"{index_circ_market_cap.iloc[0] / 1e12:.2f} -> "
        f"{index_circ_market_cap.iloc[-1] / 1e12:.2f} CNY (trillion)"
    )
    print(
        f"Index NAV (unit cash, total return): "
        f"{index_value.iloc[0]:.4f} -> {index_value.iloc[-1]:.4f}"
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    def draw_rebalances(ax):
        for d in rebalance_dates:
            ax.axvline(d, color="lightgray", linestyle="--", linewidth=0.4, zorder=0)

    ax = axes[0]
    ax.set_title(f"Index circulating market cap (top {args.index_size} by circulating market cap)")
    ax.set_ylabel("CNY (trillion)")
    draw_rebalances(ax)
    ax.plot(index_circ_market_cap.index, index_circ_market_cap / 1e12, color="C0", linewidth=0.8)

    ax = axes[1]
    ax.set_title("Index returns (cap-weighted, total return, unit cash)")
    ax.set_ylabel("NAV")
    ax.set_xlabel("Date")
    draw_rebalances(ax)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.plot(index_value.index, index_value, color="C0", linewidth=0.8)

    fig.tight_layout()
    plt.show()
