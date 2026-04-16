"""Compare covariance estimators via GMV portfolio realized variance.

For each covariance estimator (sample covariance and Ledoit-Wolf
shrinkage), a variance predictor periodically estimates the
cross-sectional covariance matrix.  A ``MinimumVariance`` evaluator
computes the global minimum variance portfolio weights from each
prediction, accumulates daily returns over the forward period, and
reports the realized variance.

Both estimators ignore features, so the features series is fed an
empty ``(num_stocks, 0)`` array per tick.  The realized volatility
(square root of variance) is plotted for each estimator over time;
lower realized volatility indicates a better covariance estimate for
the purpose of risk minimization.

Requires ``pip install -e ".[examples]"`` and A-shares market data downloaded
via the crawler.  See ``python -m a_shares_crawler --help`` for configuration
and download instructions.
"""

from pathlib import Path
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow.types import Handle
from tradingflow.sources import Clock, CSVSource, MonthlyClock
from tradingflow.operators import Clocked, Map, Record, Select, Stack
from tradingflow.operators.num import ForwardFill, Multiply
from tradingflow.operators.stocks import ForwardAdjust
from tradingflow.operators.predictors.variance import Sample, Shrinkage
from tradingflow.operators.metrics import MinimumVariance

from stocks import load_symbols, calculate_index_weights, resolve_data_start


PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())
DIVIDEND_SCHEMA = Schema(CSVSchema.dividends().iter_field_ids())


def build_scenario(
    symbols: list[str],
    data_dir: Path,
    rebalance_days: int,
    index_size: int,
    data_start: np.datetime64,
    eval_start: np.datetime64,
    end: np.datetime64,
) -> tuple[Scenario, dict, dict]:
    """Build the GMV evaluation scenario."""

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()
    num_stocks = len(symbols)

    per_stock: dict[str, list[Handle]] = {
        k: []
        for k in (
            "adjusted_close",
            "circ_shares",
            "close",
        )
    }

    for symbol in tqdm(symbols, desc="Building scenario"):
        h = history_dir / symbol

        prices = sc.add_source(
            CSVSource(f"{h}.daily_prices.csv", PRICE_SCHEMA, time_column="date", start=data_start, end=end)
        )
        equity = sc.add_source(
            CSVSource(f"{h}.equity_structures.csv", EQUITY_SCHEMA, time_column="date", start=data_start, end=end)
        )
        dividends = sc.add_source(
            CSVSource(f"{h}.dividends.csv", DIVIDEND_SCHEMA, time_column="date", start=data_start, end=end)
        )

        close = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.close")))
        adjusts = sc.add_operator(ForwardAdjust(close, dividends, output_prices=False))
        adjusted_close = sc.add_operator(Multiply(close, adjusts))
        circ_shares = sc.add_operator(Select(equity, EQUITY_SCHEMA.index("shares.circulating")))

        per_stock["adjusted_close"].append(adjusted_close)
        per_stock["circ_shares"].append(circ_shares)
        per_stock["close"].append(close)

    # ------------------------------------------------------------------
    # Cross-sectional stacking
    # ------------------------------------------------------------------

    stacked = {k: sc.add_operator(ForwardFill(sc.add_operator(Stack(v)))) for k, v in per_stock.items()}

    # ------------------------------------------------------------------
    # Features (empty — Sample and Shrinkage do not use features)
    # ------------------------------------------------------------------

    market_cap = sc.add_operator(Multiply(stacked["close"], stacked["circ_shares"]))

    # Empty features array (N, 0) ticking with adjusted_close.
    empty_features = sc.add_operator(
        Map(
            stacked["adjusted_close"],
            lambda _: np.zeros((num_stocks, 0), dtype=np.float64),
            shape=(num_stocks, 0),
            dtype=np.float64,
        )
    )
    features_series = sc.add_operator(Record(empty_features))
    adjusted_prices_series = sc.add_operator(Record(stacked["adjusted_close"]))

    # Universe: top stocks by market cap.
    monthly_clock = sc.add_source(MonthlyClock(data_start, end, tz="Asia/Shanghai"))
    universe = sc.add_operator(
        Clocked(
            monthly_clock,
            Map(
                market_cap,
                lambda m: calculate_index_weights(m, index_size),
                shape=(num_stocks,),
                dtype=np.float64,
            ),
        )
    )

    # ------------------------------------------------------------------
    # GMV evaluation
    # ------------------------------------------------------------------

    # Rebalance clock fires every `rebalance_days` calendar days from
    # `eval_start` to `end`.  A Const operator clocked by it produces
    # an Array[float64] rebalance signal that the predictors consume
    # as a regular input (observing every data tick but emitting only
    # on rebalance).
    rebalance_dates = np.arange(
        eval_start,
        end + np.timedelta64(1, "D"),
        np.timedelta64(rebalance_days, "D"),
    )
    rebalance_clock = sc.add_source(Clock(rebalance_dates))
    rebalance = rebalance_clock

    predictor_kwargs = dict(universe_size=index_size, rebalance=rebalance)

    estimators = {
        "sample": sc.add_operator(
            Sample(
                universe,
                features_series,
                adjusted_prices_series,
                max_periods=rebalance_days,
                **predictor_kwargs,
            ),
        ),
        "shrinkage": sc.add_operator(
            Shrinkage(
                universe,
                features_series,
                adjusted_prices_series,
                verbose=False,
                max_periods=rebalance_days,
                **predictor_kwargs,
            ),
        ),
    }

    eval_handles = {}
    predictions_series_handles = {}
    for name, predicted in estimators.items():
        predictions_series = sc.add_operator(Record(predicted))
        metric = sc.add_operator(
            MinimumVariance(
                predictions_series,
                adjusted_prices_series,
            )
        )
        eval_handles[name] = sc.add_operator(Record(metric))
        predictions_series_handles[name] = predictions_series

    return sc, eval_handles, predictions_series_handles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    parser.add_argument(
        "--data-begin",
        type=np.datetime64,
        default=None,
        help="data start date (default: trading begin minus --rebalance-days calendar days)",
    )
    parser.add_argument(
        "-b", "--begin", type=np.datetime64, required=True, help="evaluation start date (e.g. 2020-01-01)"
    )
    parser.add_argument("-e", "--end", type=np.datetime64, required=True, help="end date (e.g. 2025-12-31)")
    parser.add_argument("--rebalance-days", type=int, default=120, help="rebalance every N trading days")
    parser.add_argument("--index-size", type=int, default=300, help="number of stocks in the universe")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    symbols = load_symbols(data_dir)
    print(f"Discovered {len(symbols)} symbols.")

    sc, eval_handles, predictions_series_handles = build_scenario(
        symbols,
        data_dir,
        rebalance_days=args.rebalance_days,
        index_size=args.index_size,
        data_start=resolve_data_start(args.data_begin, args.begin, args.rebalance_days),
        eval_start=args.begin,
        end=args.end,
    )

    mid = args.begin
    progress = tqdm(total=sc.estimated_event_count(), unit=" events", desc="Loading data")

    def on_flush(ts_ns: int, events: int, total: int | None) -> None:
        if np.datetime64(ts_ns, "ns") > mid:
            progress.set_description("Evaluating GMV")
        if total != progress.total:
            progress.total = total
        progress.update(events - progress.n)

    sc.run(on_flush=on_flush)
    progress.close()

    # ------------------------------------------------------------------
    # Extract results with prediction timestamps
    # ------------------------------------------------------------------

    # The MinimumVariance evaluator emits the realized variance of daily
    # GMV portfolio returns within each evaluation period.
    # Annualize via the standard 252-trading-day scaling.
    TRADING_DAYS = 252

    eval_data = {}
    for name, handle in eval_handles.items():
        values = sc.series_view(handle).values()
        n = len(values)
        pred_ts = sc.series_view(predictions_series_handles[name]).timestamps()[-(n + 1) : -1]
        series = pd.Series(values, index=pd.DatetimeIndex(pred_ts), name=name)
        eval_data[name] = series

        finite = series[np.isfinite(series)]
        if len(finite) > 0:
            mean_ann_vol = np.sqrt(finite.mean() * TRADING_DAYS)
            print(f"{name}: mean annualized volatility={mean_ann_vol:.4f} ({n} periods)")
        else:
            print(f"{name}: no valid values")

    # ------------------------------------------------------------------
    # Plot annualized realized volatility over time
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("GMV annualized realized volatility by covariance estimator")
    ax.set_ylabel(f"Annualized (× √{TRADING_DAYS}) realized volatility")
    ax.set_xlabel("Date")

    for i, (name, series) in enumerate(eval_data.items()):
        ann_vol = np.sqrt(series.clip(lower=0.0) * TRADING_DAYS)
        ax.plot(ann_vol.index, ann_vol, label=name, color=f"C{i}", marker="o", markersize=3)

    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
