"""Load daily price history for an A-shares stock, compute forward-adjusted
prices, MA and Bollinger Bands via TradingFlow operators, and plot.

Requires `pip install -e ".[examples]"` and A-shares market data downloaded
via the crawler. See `python -m a_shares_crawler --help` for configuration
and download instructions.
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from tradingflow import Scenario, Schema
from tradingflow.sources import CSVSource
from tradingflow.operators import Record, Select
from tradingflow.operators.num import Add, Scale, Sqrt, Subtract
from tradingflow.operators.rolling import RollingMean, RollingVariance
from tradingflow.operators.stocks import ForwardAdjust


PRICE_SCHEMA = Schema(["open", "close", "high", "low", "amount", "volume"])
DIVIDEND_SCHEMA = Schema(["share_dividends", "cash_dividends"])
WINDOW = 252
MULTIPLE = 2


def build_scenario(symbol: str, data_dir: Path) -> tuple[Scenario, dict]:
    """Build a scenario with forward-adjusted prices, MA and Bollinger Bands."""
    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    # Daily prices. Shape: (6,).
    prices = sc.add_source(CSVSource(history_dir / f"{symbol}.daily_prices.csv", PRICE_SCHEMA))

    # Dividend events. Shape: (2,).
    dividends = sc.add_source(CSVSource(history_dir / f"{symbol}.dividends.csv", DIVIDEND_SCHEMA))

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    # Extract scalar close price. Shape: ().
    closes = sc.add_operator(Select(prices, [PRICE_SCHEMA.index("close")]))

    # Forward-adjusted close price. Shape: ().
    adj_closes = sc.add_operator(ForwardAdjust(closes, dividends))
    adj_closes_series = sc.add_operator(Record(adj_closes))

    # 252-day moving average. Shape: ().
    ma = sc.add_operator(RollingMean(adj_closes_series, window=WINDOW))
    ma_series = sc.add_operator(Record(ma))

    # 252-day rolling standard deviation. Shape: ().
    var = sc.add_operator(RollingVariance(adj_closes_series, window=WINDOW))
    std = sc.add_operator(Sqrt(var))

    # Bollinger band offset: MULTIPLE × std. Shape: ().
    band = sc.add_operator(Scale(std, MULTIPLE))

    # Upper and lower bands: MA ± MULTIPLE × std. Shape: ().
    upper = sc.add_operator(Add(ma, band))
    lower = sc.add_operator(Subtract(ma, band))
    upper_series = sc.add_operator(Record(upper))
    lower_series = sc.add_operator(Record(lower))

    # Record volume for the volume subplot. Shape: ().
    volume = sc.add_operator(Select(prices, [PRICE_SCHEMA.index("volume")]))
    volume_series = sc.add_operator(Record(volume))

    handles = {
        "adj_close": adj_closes_series,
        "ma": ma_series,
        "upper": upper_series,
        "lower": lower_series,
        "volume": volume_series,
    }
    return sc, handles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
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

    adj_close_df = sc.series_view(handles["adj_close"]).to_dataframe(["adj_close"])
    ma_df = sc.series_view(handles["ma"]).to_dataframe()
    upper_df = sc.series_view(handles["upper"]).to_dataframe()
    lower_df = sc.series_view(handles["lower"]).to_dataframe()
    volume_df = sc.series_view(handles["volume"]).to_dataframe(["volume"])

    n = len(adj_close_df)
    print(f"{symbol}: {n} trading days, {adj_close_df.index[0].date()} to {adj_close_df.index[-1].date()}")
    print(adj_close_df.tail())

    # Create plot.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Forward-adjusted close + MA + Bollinger Bands.
    ax1.plot(adj_close_df.index, adj_close_df["adj_close"], linewidth=0.5, color="C0", label="Adjusted close")
    ax1.plot(ma_df.index, ma_df, linewidth=0.8, color="C1", label=f"MA{WINDOW}")
    ax1.fill_between(
        upper_df.index,
        upper_df.values.ravel(),
        lower_df.values.ravel(),
        alpha=0.15,
        color="C1",
        label=f"Bollinger ({WINDOW}, {MULTIPLE}σ)",
    )
    ax1.set_ylabel("Price (CNY)")
    ax1.set_title(f"{symbol} forward-adjusted close with MA{WINDOW} & Bollinger Bands")
    ax1.legend(loc="upper left", fontsize=8)

    # Volume.
    ax2.bar(volume_df.index, volume_df["volume"] / 1e6, width=1, linewidth=0, color="C0", alpha=0.6)
    ax2.set_ylabel("Volume (M)")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    plt.show()
