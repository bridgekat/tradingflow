"""Clock sources — scheduling triggers at fixed timestamps.

Clock sources produce no data; they exist purely as triggers for
periodic operators. They delegate to Rust `clock`, `daily_clock`,
and `monthly_clock` implementations via `Scenario.add_source`.
"""

from __future__ import annotations

from ..source import NativeSource

import numpy as np


def clock(timestamps: list[np.datetime64] | np.ndarray) -> NativeSource:
    """Create a clock source from explicit timestamps.

    Parameters
    ----------
    timestamps
        Sequence of `datetime64` timestamps at which the clock fires.
    """
    ts_ns = np.asarray(timestamps, dtype="datetime64[ns]")
    ts_i64 = ts_ns.view("int64").tolist()
    return NativeSource("clock", params={"timestamps": ts_i64})


def daily_clock(
    start: np.datetime64,
    end: np.datetime64,
    tz: str = "UTC",
) -> NativeSource:
    """Create a daily clock (midnight in the given timezone).

    Parameters
    ----------
    start
        Start timestamp (inclusive).
    end
        End timestamp (inclusive).
    tz
        IANA timezone name (e.g. `"Asia/Shanghai"`, `"US/Eastern"`).
    """
    start_ns = int(start.astype("datetime64[ns]").view("int64"))
    end_ns = int(end.astype("datetime64[ns]").view("int64"))
    return NativeSource("daily_clock", params={"start_ns": start_ns, "end_ns": end_ns, "tz": tz})


def monthly_clock(
    start: np.datetime64,
    end: np.datetime64,
    tz: str = "UTC",
) -> NativeSource:
    """Create a monthly clock (first day of each month in the given timezone).

    Parameters
    ----------
    start
        Start timestamp (inclusive).
    end
        End timestamp (inclusive).
    tz
        IANA timezone name (e.g. `"Asia/Shanghai"`, `"US/Eastern"`).
    """
    start_ns = int(start.astype("datetime64[ns]").view("int64"))
    end_ns = int(end.astype("datetime64[ns]").view("int64"))
    return NativeSource("monthly_clock", params={"start_ns": start_ns, "end_ns": end_ns, "tz": tz})
