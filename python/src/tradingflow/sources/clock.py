"""Clock sources — scheduling triggers at fixed timestamps.

Clock sources produce no data; they exist purely as triggers for
periodic operators. They delegate to Rust `clock`, `daily_clock`,
and `monthly_clock` implementations via `Scenario.add_source`.
"""

from __future__ import annotations

import numpy as np


class NativeSource:
    """Descriptor for a Rust-implemented source.

    Analogous to `NativeOperator` — carries `kind` + `params` and is
    dispatched entirely on the native side. Not a `Source` subclass
    (no Python async iterators).
    """

    __slots__ = ("_kind", "_dtype", "_shape", "_params", "_name")

    def __init__(
        self,
        kind: str,
        *,
        dtype: str = "float64",
        shape: tuple[int, ...] = (),
        params: dict | None = None,
        name: str | None = None,
    ) -> None:
        self._kind = kind
        self._dtype = dtype
        self._shape = shape
        self._params = params or {}
        self._name = name or kind

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def params(self) -> dict:
        return self._params

    @property
    def name(self) -> str:
        return self._name


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
    start_ns = int(np.datetime64(start, "ns").view("int64"))
    end_ns = int(np.datetime64(end, "ns").view("int64"))
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
    start_ns = int(np.datetime64(start, "ns").view("int64"))
    end_ns = int(np.datetime64(end, "ns").view("int64"))
    return NativeSource("monthly_clock", params={"start_ns": start_ns, "end_ns": end_ns, "tz": tz})
