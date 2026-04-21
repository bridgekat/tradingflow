"""Clock sources — scheduling triggers at fixed timestamps.

A clock is a [`NodeKind.UNIT`][tradingflow.data.types.NodeKind] source: its output
node carries no data, only a produce bit.  Downstream operators consume
it as a trigger to gate their compute — typically via the
[`Clocked`][tradingflow.operators.clocked.Clocked] transformer or as an extra
input on performance-metric operators that fire on a schedule.

Calendar-aligned clocks (`DailyClock`, `MonthlyClock`) are constructed in
Python via the standard-library `zoneinfo` module: timestamps are
pre-computed here and passed to the native `clock` source as a list.
This keeps the Rust core free of timezone data; updates to IANA tzdb
flow through Python's standard library.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np

from .. import NodeKind
from ..source import NativeSource
from ..data import coerce_timestamp


class Clock(NativeSource):
    """Clock source from explicit timestamps.

    Emits unit events (`NodeKind.UNIT`, no payload) at the supplied
    timestamps.  The output handle exists solely to be wired into
    downstream operators as a trigger input.

    Parameters
    ----------
    timestamps
        Sequence of `datetime64` timestamps at which the clock fires.
    """

    def __init__(self, timestamps: list[np.datetime64] | np.ndarray) -> None:
        ts_ns = np.asarray(timestamps, dtype="datetime64[ns]")
        ts_i64 = ts_ns.view("int64").tolist()
        super().__init__(native_id="clock", kind=NodeKind.UNIT, dtype="", params={"timestamps": ts_i64})


class DailyClock(Clock):
    """Daily clock (midnight in the given timezone).

    Timestamps are computed at construction time using `zoneinfo`.

    Parameters
    ----------
    start
        Start timestamp (inclusive).
    end
        End timestamp (inclusive).
    tz
        IANA timezone name (e.g. `"Asia/Shanghai"`, `"US/Eastern"`).
    """

    def __init__(
        self,
        start: np.datetime64,
        end: np.datetime64,
        tz: str = "UTC",
    ) -> None:
        super().__init__(_calendar_midnights(start, end, tz, monthly=False))


class MonthlyClock(Clock):
    """Monthly clock (first day of each month in the given timezone).

    Timestamps are computed at construction time using `zoneinfo`.

    Parameters
    ----------
    start
        Start timestamp (inclusive).
    end
        End timestamp (inclusive).
    tz
        IANA timezone name (e.g. `"Asia/Shanghai"`, `"US/Eastern"`).
    """

    def __init__(
        self,
        start: np.datetime64,
        end: np.datetime64,
        tz: str = "UTC",
    ) -> None:
        super().__init__(_calendar_midnights(start, end, tz, monthly=True))


def _calendar_midnights(
    start: np.datetime64,
    end: np.datetime64,
    tz: str,
    *,
    monthly: bool,
) -> np.ndarray:
    """Generate local-midnight timestamps in `[start, end]` for a given tz.

    `monthly=False` advances daily; `monthly=True` advances to the first
    day of each subsequent month.
    """
    zone = ZoneInfo(tz)
    start_ns = int(coerce_timestamp(start))
    end_ns = int(coerce_timestamp(end))
    start_dt = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).astimezone(zone)
    end_dt = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).astimezone(zone)

    # Align to first-of-month for monthly cadence.
    date = start_dt.date()
    if monthly and date.day != 1:
        date = (date.replace(day=1) + timedelta(days=32)).replace(day=1)

    out: list[np.datetime64] = []
    while date <= end_dt.date():
        local_midnight = datetime(date.year, date.month, date.day, tzinfo=zone)
        ts_ns = int(local_midnight.timestamp() * 1e9)
        if start_ns <= ts_ns <= end_ns:
            out.append(np.datetime64(ts_ns, "ns"))
        if monthly:
            date = (date.replace(day=28) + timedelta(days=4)).replace(day=1)
        else:
            date += timedelta(days=1)
    return np.asarray(out, dtype="datetime64[ns]") if out else np.empty(0, dtype="datetime64[ns]")
