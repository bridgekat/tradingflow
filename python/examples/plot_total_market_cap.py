"""Plot the total market capitalisation of all A-shares stocks over time.

For each stock: market_cap = close_price * total_shares.  The per-stock
market caps are stacked into a vector and summed via a Map operator.

Requires `pip install -e ".[examples]"` and A-shares market data downloaded
via the crawler. See `python -m a_shares_crawler --help` for configuration
and download instructions.
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow.sources import CSVSource
from tradingflow.operators import Map, Record, Select, Stack
from tradingflow.operators.num import Multiply

from stocks import load_symbols


DAY_NS = 86_400_000_000_000
PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict]:
    """Build a scenario that sums per-stock market caps."""

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    market_caps: list = []
    for symbol in tqdm(symbols, desc="Building scenario"):
        price_path = history_dir / f"{symbol}.daily_prices.csv"
        equity_path = history_dir / f"{symbol}.equity_structures.csv"
        prices = sc.add_source(CSVSource(price_path, PRICE_SCHEMA, time_column="date", start=start, end=end))
        equity = sc.add_source(CSVSource(equity_path, EQUITY_SCHEMA, time_column="date", start=start, end=end))
        close = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.close")))
        shares = sc.add_operator(Select(equity, EQUITY_SCHEMA.index("shares.circulating")))
        market_cap = sc.add_operator(Multiply(close, shares))
        market_caps.append(market_cap)

    # Stack all per-stock market caps into (N,), then sum via Map.
    stacked = sc.add_operator(Stack(market_caps))
    total = sc.add_operator(Map(stacked, np.nansum, shape=(), dtype=np.float64))

    return sc, {"total_market_cap": sc.add_operator(Record(total))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument("-b", "--begin", type=np.datetime64, required=True, help="start date (e.g. 2020-01-01)")
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    symbols = load_symbols(data_dir)
    print(f"Discovered {len(symbols)} symbols.")

    # Run scenario.
    sc, handles = build_scenario(symbols, data_dir, start=args.begin, end=args.end)

    first_ns_opt, last_ns_opt = sc.time_range()
    assert first_ns_opt is not None and last_ns_opt is not None, "all sources must provide a time range"
    first_ns, last_ns = first_ns_opt, last_ns_opt

    total_days = (last_ns - first_ns) // DAY_NS
    progress = tqdm(total=total_days, unit="d", desc="Running scenario")
    sc.run(on_flush=lambda ts: progress.update((min(max(ts, first_ns), last_ns) - first_ns) // DAY_NS - progress.n))
    progress.close()

    # Extract results.
    total_market_cap = sc.series_view(handles["total_market_cap"]).to_series()

    n = len(total_market_cap)
    if n == 0:
        raise SystemExit("No data produced.")

    print(f"{n} trading days, {total_market_cap.index[0].date()} to {total_market_cap.index[-1].date()}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(total_market_cap.index, total_market_cap / 1e12, linewidth=0.8)
    ax.set_ylabel("CNY (trillion)")
    ax.set_xlabel("Date")
    ax.set_title(f"A-shares total circulating market cap ({len(symbols)} stocks)")

    fig.tight_layout()
    plt.show()
