"""Array-bundle historical source dispatched to Rust."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike

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
        Array-like of values; first dimension must match *timestamps*.
    dtype
        Optional NumPy dtype to cast *values* to.
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
        ts = np.asarray(timestamps, dtype="datetime64[ns]")
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
        ts_i64 = ts.view("int64")
        if len(ts_i64) > 1 and np.any(np.diff(ts_i64) < 0):
            raise ValueError("timestamps must be non-decreasing")

        shape = cast(tuple[int, ...], vals.shape[1:])
        stride = int(np.prod(shape)) if shape else 1
        values_bytes = np.ascontiguousarray(vals).tobytes()

        super().__init__(
            "array",
            dtype=str(dt),
            shape=shape,
            params={
                "timestamps": ts_i64.tolist(),
                "values_bytes": values_bytes,
                "stride": stride,
            },
            name=name,
        )

    @classmethod
    def from_arrays(
        cls,
        timestamps: ArrayLike,
        values: ArrayLike,
        *,
        dtype: type | np.dtype | None = None,
        name: str | None = None,
    ) -> ArraySource:
        """Construct from in-memory arrays.

        Convenience alias for the constructor — parameters are identical
        to [`ArraySource.__init__`][tradingflow.sources.ArraySource.__init__].
        """
        return cls(timestamps=timestamps, values=values, dtype=dtype, name=name)
