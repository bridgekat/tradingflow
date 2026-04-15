"""Internal helpers shared across the package."""

from __future__ import annotations

import numpy as np

from tradingflow._native import tai_to_utc as _tai_to_utc_scalar
from tradingflow._native import utc_to_tai as _utc_to_tai_scalar


def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Return *arr* as a C-contiguous array, preserving shape.

    Unlike ``np.ascontiguousarray``, this does **not** promote 0-d arrays
    to 1-d.  If the array is already C-contiguous, it is returned as-is
    (no copy).
    """
    if arr.flags["C_CONTIGUOUS"]:
        return arr
    else:
        assert arr.ndim > 0
        return np.ascontiguousarray(arr)


def coerce_timestamp(ts: np.datetime64 | int | np.integer) -> np.int64:
    """Coerce a timestamp to int64 nanoseconds without any conversion.

    TradingFlow uses TAI throughout, on both sides of the PyO3 bridge.
    A `datetime64[ns]` value is reinterpreted as its stored `int64`
    directly — no timescale math, no leap-second correction.  The
    resulting integer is an [`Instant`][tradingflow.Instant]'s numeric
    value: SI nanoseconds since the PTP epoch (1970-01-01 00:00:00 TAI).

    Because NumPy / pandas `datetime64` arithmetic is itself naïve of
    leap seconds, this matches numpy's own semantics exactly.  The only
    caveat: a string like `"2024-01-01"` parsed by NumPy labels an
    instant 37 s *earlier* in wall-clock UTC than the eponymous UTC
    midnight.  To convert to/from the UTC wall-clock convention for
    plotting or interoperability with external systems, use
    [`utc_to_tai`][tradingflow.utils.utc_to_tai] or
    [`tai_to_utc`][tradingflow.utils.tai_to_utc].

    Accepts `datetime64` (any precision; coerced to ns), plain `int`, or
    `np.integer`.
    """
    if isinstance(ts, (int, np.integer)):
        return np.int64(ts)
    return ts.astype("datetime64[ns]").view("int64")


def utc_to_tai(ts: np.datetime64 | int | np.integer | np.ndarray) -> np.ndarray | np.int64:
    """Convert UTC-convention nanoseconds to TAI nanoseconds.

    Accepts a scalar (`datetime64`, `int`, or `np.integer`) or a numpy
    array of any integer / datetime64 dtype.  Returns the same kind:
    scalars return `np.int64`; arrays return a contiguous `int64` array
    in TAI ns.  Reinterpret the result as `datetime64[ns]` for display.

    The conversion applies the current TAI−UTC offset via hifitime's
    IERS leap-second table (37 s for any date from 2017 to present; 0 s
    for pre-1972; integer seconds for dates in between).
    """
    if isinstance(ts, np.ndarray):
        flat = ts.astype("datetime64[ns]", copy=False).view("int64").ravel()
        out = np.fromiter((_utc_to_tai_scalar(int(x)) for x in flat), dtype=np.int64, count=flat.size)
        return out.reshape(ts.shape)
    return np.int64(_utc_to_tai_scalar(int(coerce_timestamp(ts))))


def tai_to_utc(ts: np.datetime64 | int | np.integer | np.ndarray) -> np.ndarray | np.int64:
    """Convert TAI nanoseconds (this crate's native timeline) to
    UTC-convention nanoseconds (UNIX time, as consumed by most external
    systems).

    Accepts a scalar or a numpy array; returns the same kind.  Useful
    for plot axis labels when the user wants UTC wall-clock dates
    instead of the default TAI display.
    """
    if isinstance(ts, np.ndarray):
        flat = ts.astype("datetime64[ns]", copy=False).view("int64").ravel()
        out = np.fromiter((_tai_to_utc_scalar(int(x)) for x in flat), dtype=np.int64, count=flat.size)
        return out.reshape(ts.shape)
    return np.int64(_tai_to_utc_scalar(int(coerce_timestamp(ts))))
