"""Load daily price history for an A-shares stock, compute MA and Bollinger
Bands via TradingFlow operators, and plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tradingflow import Scenario, Schema
from tradingflow.sources import CSVSource
from tradingflow.operators import Add, Last, Record, RollingMean, RollingVariance, Scale, Select, Sqrt, Subtract


DATA_DIR = Path(__file__).parent / "data" / "a_shares_history_raw"

PRICE_SCHEMA = Schema(["open", "close", "high", "low", "volume", "amount"])
WINDOW = 20


def build_scenario(symbol: str) -> tuple[Scenario, dict]:
    """Build a scenario with MA and Bollinger Band operators."""
    sc = Scenario()

    # Load daily price history from CSV. Shape: (6,).
    price_arr = sc.add_source(CSVSource(DATA_DIR / f"{symbol}_daily_price_raw.csv", PRICE_SCHEMA))

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


def extract_series(sc: Scenario, handle, columns=None) -> pd.DataFrame:
    """Extract a recorded Series as a DataFrame."""
    view = sc.series_view(handle)
    ts = view.timestamps()
    vals = view.values()
    return pd.DataFrame(vals, index=pd.DatetimeIndex(ts), columns=columns)


def main(symbol: str = "000001") -> None:
    sc, handles = build_scenario(symbol)
    sc.run()

    price_cols = [PRICE_SCHEMA.name(i) for i in range(len(PRICE_SCHEMA))]
    price_df = extract_series(sc, handles["price"], columns=price_cols)
    ma_df = extract_series(sc, handles["ma"], columns=["MA20"])
    upper_df = extract_series(sc, handles["upper"], columns=["upper"])
    lower_df = extract_series(sc, handles["lower"], columns=["lower"])

    print(f"{symbol}: {len(price_df)} trading days, " f"{price_df.index[0].date()} to {price_df.index[-1].date()}")
    print(price_df.tail())

    # --- Plot ----------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Price + MA + Bollinger Bands
    ax1.plot(price_df.index, price_df["close"], linewidth=0.5, color="C0", label="Close")
    ax1.plot(ma_df.index, ma_df["MA20"], linewidth=0.8, color="C1", label=f"MA{WINDOW}")
    ax1.fill_between(
        upper_df.index,
        upper_df["upper"].values.ravel(),
        lower_df["lower"].values.ravel(),
        alpha=0.15,
        color="C1",
        label=f"Bollinger ({WINDOW}, 2σ)",
    )
    ax1.set_ylabel("Price (CNY)")
    ax1.set_title(f"{symbol} daily close with MA{WINDOW} & Bollinger Bands")
    ax1.legend(loc="upper left", fontsize=8)

    # Volume
    ax2.bar(price_df.index, price_df["volume"] / 1e6, width=1, linewidth=0, color="C0", alpha=0.6)
    ax2.set_ylabel("Volume (M)")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "000001"
    main(symbol)
