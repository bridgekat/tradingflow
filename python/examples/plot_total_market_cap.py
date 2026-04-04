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
import csv
import numpy as np
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow.sources import CSVSource
from tradingflow.operators import Map, Record, Select, Stack
from tradingflow.operators.num import Multiply


DAY_NS = 86_400_000_000_000
PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())


def discover_symbols(data_dir: Path) -> list[str]:
    """Read stock symbols from the symbol list CSV."""

    symbol_list_path = data_dir / "symbol_list.csv"
    if not symbol_list_path.exists():
        raise SystemExit(f"Symbol list not found: {symbol_list_path}")

    with symbol_list_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        i = header.index("symbol")
        symbols = [row[i] for row in reader]

    return symbols


def estimate_date_range_ns(data_dir: Path) -> tuple[int, int]:
    """Estimate the historical date range by peeking at a sample CSV."""

    history_dir = data_dir / "a_shares_history"
    sample = next(history_dir.glob("*.daily_prices.csv"), None)
    if sample is None:
        raise SystemExit("No price CSV files found.")

    with sample.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        i = header.index("date")
        dates = [row[i] for row in reader]

    first_ns = int(np.datetime64(dates[0], "ns").view(np.int64))
    last_ns = int(np.datetime64(dates[-1], "ns").view(np.int64))
    return first_ns, last_ns


def build_scenario(symbols: list[str], data_dir: Path) -> tuple[Scenario, dict]:
    """Build a scenario that sums per-stock market caps."""

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    market_caps: list = []
    for symbol in tqdm(symbols, desc="Building scenario"):
        price_path = history_dir / f"{symbol}.daily_prices.csv"
        equity_path = history_dir / f"{symbol}.equity_structures.csv"
        prices = sc.add_source(CSVSource(price_path, PRICE_SCHEMA, time_column="date"))
        equity = sc.add_source(CSVSource(equity_path, EQUITY_SCHEMA, time_column="date"))
        close = sc.add_operator(Select(prices, [PRICE_SCHEMA.index("prices.close")]))
        shares = sc.add_operator(Select(equity, [EQUITY_SCHEMA.index("shares.circulating")]))
        market_cap = sc.add_operator(Multiply(close, shares))
        market_caps.append(market_cap)

    # Stack all per-stock market caps into (N,), then sum via Map.
    stacked = sc.add_operator(Stack(market_caps))
    total = sc.add_operator(Map(stacked, np.nansum, shape=(), dtype=np.float64))

    return sc, {"total_market_cap": sc.add_operator(Record(total))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    args = parser.parse_args()

    data_dir: Path = args.data_dir

    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    symbols = discover_symbols(data_dir)
    print(f"Discovered {len(symbols)} symbols.")

    # Run scenario.
    first_ns, last_ns = estimate_date_range_ns(data_dir)
    sc, handles = build_scenario(symbols, data_dir)
    total_days = (last_ns - first_ns) // DAY_NS
    progress = tqdm(total=total_days, unit="d", desc="Running scenario")

    def on_flush(ts_ns: int) -> None:
        elapsed_days = (min(max(ts_ns, first_ns), last_ns) - first_ns) // DAY_NS
        progress.update(elapsed_days - progress.n)

    sc.run(on_flush=on_flush)
    progress.close()

    # Extract series as DataFrames.
    total_market_cap_df = sc.series_view(handles["total_market_cap"]).to_dataframe()

    n = len(total_market_cap_df)
    if n == 0:
        raise SystemExit("No data produced.")

    print(f"{n} trading days, {total_market_cap_df.index[0].date()} to {total_market_cap_df.index[-1].date()}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(total_market_cap_df.index, total_market_cap_df / 1e12, linewidth=0.8)
    ax.set_ylabel("CNY (trillion)")
    ax.set_xlabel("Date")
    ax.set_title(f"A-shares total circulating market cap ({len(symbols)} stocks)")

    fig.tight_layout()
    plt.show()
