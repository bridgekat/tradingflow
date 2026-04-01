"""Internal helpers shared across the package."""

from __future__ import annotations

import numpy as np


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


def coerce_timestamp(ts: np.datetime64) -> np.int64:
    """Coerce a timestamp to `datetime64[ns]` precision and return as `int64`."""
    return ts.astype("datetime64[ns]").view("int64")
