#!/usr/bin/env python3
"""Re-create the long-format Parquet panels from `a-shares-crawler` CSV data.

Reads the per-symbol CSV histories written by the `a-shares-crawler` package
(`<data-dir>/a_shares_history/<symbol>.<kind>.csv`) and writes one long-format
Parquet table per kind, in the layout the `tradingflow` panel source expects:
one timestamp column (sorted ascending), a dictionary-encoded `symbol` index
column, and value columns.

Two families of kinds are handled differently:

* **Event kinds** (`daily_prices`, `dividends`, `equity_structures`): the
  CSV's `date` is the market event time (trading day, ex-dividend day, change
  day), so rows are only globally sorted by `(date, symbol)` and written as-is.

* **Report kinds** (`income_statements`, `cash_flow_statements`,
  `indirect_statements`, `balance_sheets`): the CSV's `date` is the report's
  *period end*, which is not when the market learns of it. For look-ahead-safe
  backtesting each row is re-keyed to its **effective date**:

      effective = max(report_date, notice_date)

  A notice date is believed only when it is plausibly the report's
  *first* publication, i.e. no later than that period's disclosure
  deadline. The A-share deadlines differ by period kind, so there is one
  lag per kind: `--quarterly-lag-days` (default 30),
  `--semi-annual-lag-days` (default 60) and `--annual-lag-days`
  (default 180). The period kind comes from the period-end month, the
  fiscal year being the calendar year.

  Each lag is both the plausibility cut and the fallback, so a notice
  that is missing and one that is implausibly late are treated
  identically: the period becomes visible at `report_date + lag`. This
  matters because the crawler keeps only the *newest* notice per period,
  so a notice long after the period end belongs to a later filing that
  restated the period as a comparative, and says nothing about when the
  market first saw it.

  Rows are then sorted by effective date per symbol, and **restated reports
  are dropped**: walking a symbol's rows in effective order, a row is kept
  only if its report date strictly advances the running maximum. A
  restatement (same or older period published later) never fires an event, so
  a backtest sees each period exactly once, at first publication.

  The emitted `date` column *is* the effective date (the panel source fires
  on it); the period end is kept as `report_date`, along with `report_year` /
  `report_day_of_year` (1-based) value columns for YTD annualization. When an
  annual report and a Q1 report go public on the same day (the common April
  cluster), both rows share one timestamp; sorted by ascending `report_date`,
  the later period lands last in the cross-section cell.

All timestamp-like columns are written as Parquet `date32`; the panel source
converts them on read. Value columns are written as `float64` (`error` stays
`bool`), and `symbol` is dictionary-encoded.

Usage:

    python examples/export_parquet.py \
        --crawler-data path/to/a-shares-crawler/data \
        --out path/to/output/dir
"""

import argparse
import sys
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EVENT_KINDS = ["daily_prices", "dividends", "equity_structures"]
REPORT_KINDS = [
    "income_statements",
    "cash_flow_statements",
    "indirect_statements",
    "balance_sheets",
]

DATE_COLUMNS = ["date", "notice_date", "report_date"]
"""Period-end months of the annual and semi-annual reports (the A-share
fiscal year is the calendar year); every other period end is quarterly."""
ANNUAL_MONTH = 12
SEMI_ANNUAL_MONTH = 6
ROW_GROUP_SIZE = 1 << 18


def iter_symbol_csvs(history_dir: Path, kind: str):
    """`(symbol, csv_path)` for every per-symbol history of `kind`, sorted."""
    suffix = f".{kind}.csv"
    for path in tqdm(sorted(history_dir.glob(f"*{suffix}"))):
        yield path.name.removesuffix(suffix), path


def read_history(path: Path, has_notice: bool) -> pd.DataFrame:
    dates = ["date", "notice_date"] if has_notice else ["date"]
    df = pd.read_csv(path, parse_dates=dates)
    if df["date"].isna().any():
        raise ValueError(f"{path}: null in date column")
    return df


def prepare_report(
    df: pd.DataFrame,
    quarterly_lag_days: int,
    semi_annual_lag_days: int,
    annual_lag_days: int,
) -> pd.DataFrame:
    """Re-key one symbol's reports to effective dates and drop restatements."""
    report_date = df["date"]
    notice_date = df["notice_date"]
    # One disclosure lag per period kind, serving as both the plausibility cut
    # on the notice and the fallback when it fails. The crawler keeps only the
    # newest notice per period, so a notice past the deadline belongs to a
    # later filing that restated the period as a comparative and says nothing
    # about when the market first saw it — exactly as uninformative as a
    # missing one, and treated the same.
    month = report_date.dt.month
    lag = pd.to_timedelta(
        np.select(
            [month == ANNUAL_MONTH, month == SEMI_ANNUAL_MONTH],
            [annual_lag_days, semi_annual_lag_days],
            default=quarterly_lag_days,
        ),
        unit="D",
    )
    deadline = report_date + lag
    usable = notice_date.notna() & (notice_date <= deadline)
    effective = pd.Series(
        np.where(usable, np.maximum(report_date, notice_date.fillna(deadline)), deadline),
        index=df.index,
    )

    # Stable (effective, report_date) order, so simultaneous publications
    # keep ascending periods and the high-water walk below is well-defined.
    order = np.lexsort((report_date.to_numpy(), effective.to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    report_date = df["date"]
    effective = pd.Series(effective.to_numpy()[order])

    # A report fires only if its period strictly advances the newest period
    # already published; anything else is a restatement, which a live
    # backtest must not see as a fresh event.
    high_water = report_date.cummax().shift(1)
    keep = high_water.isna() | (report_date > high_water)

    out = pd.DataFrame()
    out["date"] = effective
    out["report_date"] = report_date
    out["notice_date"] = df["notice_date"]
    out["report_year"] = report_date.dt.year.astype(np.int32)
    out["report_day_of_year"] = report_date.dt.dayofyear.astype(np.int32)
    for column in df.columns:
        if column not in ("date", "notice_date"):
            out[column] = df[column]
    return out[keep.to_numpy()]


def export_kind(
    history_dir: Path,
    out_dir: Path,
    kind: str,
    quarterly_lag_days: int,
    semi_annual_lag_days: int,
    annual_lag_days: int,
) -> None:
    is_report = kind in REPORT_KINDS
    has_notice = kind != "daily_prices"

    frames = []
    symbols = []
    num_input_rows = 0
    for symbol, path in iter_symbol_csvs(history_dir, kind):
        df = read_history(path, has_notice)
        if df.empty:
            continue
        num_input_rows += len(df)
        if is_report:
            df = prepare_report(df, quarterly_lag_days, semi_annual_lag_days, annual_lag_days)
        if not df.empty:
            frames.append(df)
            symbols.extend([symbol] * len(df))

    if not frames:
        print(f"{kind}: no rows, skipped", file=sys.stderr)
        return

    table = pd.concat(frames, ignore_index=True)
    table.insert(1, "symbol", pd.Categorical(symbols))

    # Global stable sort: the panel source streams the table in timestamp
    # order. Within a timestamp, symbols ascend; within a symbol, report
    # periods ascend (so the latest period wins the cross-section cell).
    keys = [table["symbol"].cat.codes.to_numpy(), table["date"].to_numpy()]
    if is_report:
        keys.insert(0, table["report_date"].to_numpy())
    table = table.iloc[np.lexsort(tuple(keys))].reset_index(drop=True)

    arrow = pa.Table.from_pandas(table, preserve_index=False)
    for name in DATE_COLUMNS:
        if name in table.columns:
            i = arrow.schema.get_field_index(name)
            column = pc.cast(arrow.column(i), pa.date32())
            arrow = arrow.set_column(i, pa.field(name, pa.date32()), column)

    # Pandas categoricals convert to `dictionary<int16, large_string>`; pin
    # the canonical `dictionary<int32, string>` instead — `large_string`
    # labels are not accepted by the panel source's index resolution.
    symbol_type = pa.dictionary(pa.int32(), pa.string())
    i = arrow.schema.get_field_index("symbol")
    column = pc.cast(arrow.column(i), symbol_type)
    arrow = arrow.set_column(i, pa.field("symbol", symbol_type), column)

    out_path = out_dir / f"{kind}.parquet"
    pq.write_table(arrow, out_path, row_group_size=ROW_GROUP_SIZE, compression="zstd")
    dropped = num_input_rows - len(table)
    note = f" ({dropped} restated rows dropped)" if is_report else ""
    print(f"{kind}: {len(table)} rows -> {out_path}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a-shares-crawler CSV data as long-format Parquet panels.")
    parser.add_argument(
        "--crawler-data",
        type=Path,
        required=True,
        metavar="DIR",
        help="the crawler's data directory (containing a_shares_history/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="output directory for the Parquet files",
    )
    parser.add_argument(
        "--quarterly-lag-days",
        type=int,
        default=31,
        metavar="DAYS",
        help="disclosure lag for a quarterly period: a notice later than "
        "this is discarded, and the period instead becomes visible this "
        "long after its end (default: 31)",
    )
    parser.add_argument(
        "--semi-annual-lag-days",
        type=int,
        default=62,
        metavar="DAYS",
        help="the same, for an interim (June) period end (default: 62)",
    )
    parser.add_argument(
        "--annual-lag-days",
        type=int,
        default=121,
        metavar="DAYS",
        help="the same, for an annual (December) period end (default: 121)",
    )
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=EVENT_KINDS + REPORT_KINDS,
        choices=EVENT_KINDS + REPORT_KINDS,
        metavar="KIND",
        help="subset of kinds to export (default: all)",
    )
    args = parser.parse_args()

    history_dir = args.crawler_data / "a_shares_history"
    if not history_dir.is_dir():
        parser.error(f"{history_dir} is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)

    for kind in args.kinds:
        export_kind(
            history_dir,
            args.out,
            kind,
            args.quarterly_lag_days,
            args.semi_annual_lag_days,
            args.annual_lag_days,
        )


if __name__ == "__main__":
    main()
