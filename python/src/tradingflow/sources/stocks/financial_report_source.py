"""Financial report CSV source with two-timestamp logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import Schema
from ...source import NativeSource
from ...data import coerce_timestamp


class FinancialReportSource(NativeSource):
    """Historical source for financial report CSVs with two date columns.

    Reads a CSV with a report-date column, an optional notice-date column,
    and numeric value columns.

    When `use_effective_date` is `True` (the default), the event timestamp
    for each row is `max(report_date, notice_date)`, falling back to
    `report_date + notice_date_fallback` when the notice date is missing.
    Reports whose report date is earlier than a previously emitted report
    date are dropped (retrospective updates).  This mode ensures backtesting
    correctness — data cannot be used before publication.

    When `use_effective_date` is `False`, the report date is used directly
    as the event timestamp.  This is useful for analysis that should align
    with reporting periods rather than publication dates.

    When `with_report_date` is `True`, the output array is prepended with
    two extra elements `[year, day_of_year]` derived from the report date,
    for use with the [`Annualize`][tradingflow.operators.stocks.annualize.Annualize]
    operator.

    Parameters
    ----------
    path
        Path to the CSV file.
    schema
        Column names to load as values (determines element shape and order).
    report_date_column
        Name of the report-date column.
    notice_date_column
        Name of the notice-date column.  If the column is absent from the
        CSV, every row falls back to `report_date + notice_date_fallback`.
    with_report_date
        If `True`, prepend `[year, day_of_year]` to the output array.
        Required when piping into `Annualize`.
    use_effective_date
        If `True` (default), use `max(report_date, notice_date)` as the
        event timestamp and drop retrospective updates.  If `False`, use
        the report date directly.
    notice_date_fallback
        Offset added to the report date when the notice date is missing
        (only used when `use_effective_date` is `True`).  Defaults to
        90 days.
    is_utc
        If `True` (default), date strings are interpreted as UTC
        wall-clock instants and converted to this crate's TAI timeline
        via the IERS leap-second table.  If `False`, date strings are
        treated as TAI wall-clock directly.
    tz_offset
        Offset of the date-string timezone from the reference timescale.
        Defaults to zero.
    start
        Optional inclusive start bound.  Rows before this timestamp are
        dropped and the reported time range is clamped.
    end
        Optional inclusive end bound.  Rows after this timestamp are
        dropped and the reported time range is clamped.
    name
        Optional source name.
    """

    def __init__(
        self,
        path: str | Path,
        schema: Schema,
        *,
        report_date_column: str,
        notice_date_column: str,
        with_report_date: bool = False,
        use_effective_date: bool = True,
        notice_date_fallback: np.timedelta64 = np.timedelta64(90, "D"),
        is_utc: bool = True,
        tz_offset: np.timedelta64 = np.timedelta64(0, "ns"),
        start: np.datetime64 | None = None,
        end: np.datetime64 | None = None,
        name: str | None = None,
    ) -> None:
        stride = len(schema)
        if with_report_date:
            shape = (2 + stride,)
        else:
            shape = () if stride == 1 else (stride,)

        fallback_ns = int(np.timedelta64(notice_date_fallback, "ns").astype(np.int64))
        tz_offset_ns = int(np.timedelta64(tz_offset, "ns").astype(np.int64))

        params: dict = {
            "path": str(Path(path).resolve()),
            "report_date_column": report_date_column,
            "notice_date_column": notice_date_column,
            "value_columns": schema.names,
            "with_report_date": with_report_date,
            "use_effective_date": use_effective_date,
            "notice_date_fallback_ns": fallback_ns,
            "is_utc": is_utc,
            "tz_offset_ns": tz_offset_ns,
        }
        if start is not None:
            params["start_ns"] = int(coerce_timestamp(start))
        if end is not None:
            params["end_ns"] = int(coerce_timestamp(end))

        super().__init__(
            native_id="financial_report",
            dtype="float64",
            shape=shape,
            params=params,
            name=name,
        )
