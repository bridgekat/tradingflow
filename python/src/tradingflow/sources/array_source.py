"""Array-bundle historical source dispatched to Rust."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ..data import ensure_contiguous
from ..source import NativeSource


class ArraySource(NativeSource):
    """Historical source backed by `(timestamps, values)` array bundles.

    Dispatched entirely to the Rust `ArraySource` implementation for
    maximum throughput — no Python async iterators or channel overhead.

    Parameters
    ----------
    timestamps
        1-D array-like of `datetime64`-compatible timestamps in
        non-decreasing order.
    values
        Array-like of values; first dimension must match `timestamps`.
    dtype
        Optional NumPy dtype to cast `values` to.
    name
        Optional source name.
    """

    def __init__(
        self,
        timestamps: ArrayLike,
        values: ArrayLike,
        *,
        dtype: type | np.dtype | None = None,
        name: str | None = None,
    ) -> None:
        ts = np.asarray(timestamps, dtype="datetime64[ns]").view("int64")
        raw = np.asarray(values)
        dt = np.dtype(dtype) if dtype is not None else raw.dtype
        vals = raw.astype(dt)

        if ts.ndim != 1:
            raise ValueError(f"timestamps must be 1-dimensional, got ndim={ts.ndim}")
        if vals.ndim < 1:
            raise ValueError(f"values must have at least 1 dimension, got ndim={vals.ndim}")
        if len(ts) != len(vals):
            raise ValueError(f"timestamps length {len(ts)} does not match values length {len(vals)}")

        # Validate non-decreasing timestamps.
        if len(ts) > 1 and np.any(np.diff(ts) < 0):
            raise ValueError("timestamps must be non-decreasing")

        shape = vals.shape[1:]

        super().__init__(
            native_id="array",
            dtype=str(dt),
            shape=shape,
            params={
                "timestamps": ensure_contiguous(ts),
                "values": ensure_contiguous(vals),
                "shape": list(shape),
            },
            name=name,
        )
