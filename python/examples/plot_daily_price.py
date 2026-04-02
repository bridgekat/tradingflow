"""Load daily price history for an A-shares stock, compute MA and Bollinger
Bands via TradingFlow operators, and plot.

Requires `pip install -e ".[examples]"` and A-shares market data downloaded
via the crawler. See `python -m a_shares_crawler --help` for configuration
and download instructions.
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from tradingflow import Scenario, Schema
from tradingflow.sources import CSVSource
from tradingflow.operators import Add, Last, Record, RollingMean, RollingVariance, Scale, Select, Sqrt, Subtract


PRICE_SCHEMA = Schema(["open", "close", "high", "low", "amount", "volume"])
WINDOW = 20


def build_scenario(symbol: str, data_dir: Path) -> tuple[Scenario, dict]:
    """Build a scenario with MA and Bollinger Band operators."""
    sc = Scenario()

    # Load daily price history from CSV. Shape: (6,).
    price_arr = sc.add_source(CSVSource(data_dir / "a_shares_history" / f"{symbol}.daily_prices.csv", PRICE_SCHEMA))

    # Record full price vector for the price/volume plot.
    price_series = sc.add_operator(Record(price_arr))

    # Extract close price. Shape: ().
    close_arr = sc.add_operator(Select(price_arr, [PRICE_SCHEMA.index("close")]))
    close_series = sc.add_operator(Record(close_arr))

    # 20-day moving average (Series → Series).
    ma_series = sc.add_operator(RollingMean(close_series, window=WINDOW))

    # 20-day rolling std: sqrt(variance) (Series → Series → Array → Array).
    var_series = sc.add_operator(RollingVariance(close_series, window=WINDOW))
    var_arr = sc.add_operator(Last(var_series))
    std_arr = sc.add_operator(Sqrt(var_arr))

    # Bollinger band offset: 2 × std.
    band_arr = sc.add_operator(Scale(std_arr, 2.0))

    # Upper and lower bands: MA ± 2×std (Array + Array → record → Series).
    ma_arr = sc.add_operator(Last(ma_series))
    upper_arr = sc.add_operator(Add(ma_arr, band_arr))
    lower_arr = sc.add_operator(Subtract(ma_arr, band_arr))
    upper_series = sc.add_operator(Record(upper_arr))
    lower_series = sc.add_operator(Record(lower_arr))

    handles = {
        "price": price_series,
        "ma": ma_series,
        "upper": upper_series,
        "lower": lower_series,
    }
    return sc, handles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", type=str, required=True, help='stock symbol (e.g. "000001.SZ")')
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    args = parser.parse_args()

    symbol: str = args.symbol
    data_dir: Path = args.data_dir

    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    sc, handles = build_scenario(symbol, data_dir)
    sc.run()

    price_cols = [PRICE_SCHEMA.name(i) for i in range(len(PRICE_SCHEMA))]
    price_df = sc.series_view(handles["price"]).to_dataframe(price_cols)
    ma_df = sc.series_view(handles["ma"]).to_dataframe()
    upper_df = sc.series_view(handles["upper"]).to_dataframe()
    lower_df = sc.series_view(handles["lower"]).to_dataframe()

    print(f"{symbol}: {len(price_df)} trading days, " f"{price_df.index[0].date()} to {price_df.index[-1].date()}")
    print(price_df.tail())

    # Create plot.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Price + MA + Bollinger Bands.
    ax1.plot(price_df.index, price_df["close"], linewidth=0.5, color="C0", label="Close")
    ax1.plot(ma_df.index, ma_df, linewidth=0.8, color="C1", label=f"MA{WINDOW}")
    ax1.fill_between(
        upper_df.index,
        upper_df.values.ravel(),
        lower_df.values.ravel(),
        alpha=0.15,
        color="C1",
        label=f"Bollinger ({WINDOW}, 2σ)",
    )
    ax1.set_ylabel("Price (CNY)")
    ax1.set_title(f"{symbol} daily close with MA{WINDOW} & Bollinger Bands")
    ax1.legend(loc="upper left", fontsize=8)

    # Volume.
    ax2.bar(price_df.index, price_df["volume"] / 1e6, width=1, linewidth=0, color="C0", alpha=0.6)
    ax2.set_ylabel("Volume (M)")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    plt.show()
