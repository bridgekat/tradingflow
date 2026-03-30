"""Clock sources -- scheduling triggers at fixed timestamps."""

from __future__ import annotations

import numpy as np

from ..source import NativeSource


class Clock(NativeSource):
    """Clock source from explicit timestamps.

    Parameters
    ----------
    timestamps
        Sequence of `datetime64` timestamps at which the clock fires.
    """

    def __init__(self, timestamps: list[np.datetime64] | np.ndarray) -> None:
        ts_ns = np.asarray(timestamps, dtype="datetime64[ns]")
        ts_i64 = ts_ns.view("int64").tolist()
        super().__init__("clock", params={"timestamps": ts_i64})


class DailyClock(NativeSource):
    """Daily clock (midnight in the given timezone).

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
        start_ns = int(start.astype("datetime64[ns]").view("int64"))
        end_ns = int(end.astype("datetime64[ns]").view("int64"))
        super().__init__("daily_clock", params={"start_ns": start_ns, "end_ns": end_ns, "tz": tz})


class MonthlyClock(NativeSource):
    """Monthly clock (first day of each month in the given timezone).

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
        start_ns = int(start.astype("datetime64[ns]").view("int64"))
        end_ns = int(end.astype("datetime64[ns]").view("int64"))
        super().__init__("monthly_clock", params={"start_ns": start_ns, "end_ns": end_ns, "tz": tz})
