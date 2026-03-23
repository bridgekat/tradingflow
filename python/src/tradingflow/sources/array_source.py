"""Array-bundle historical source."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from ..source import Source, empty_live_gen


class ArraySource(Source):
    """Historical source backed by `(timestamps, values)` array bundles.

    Parameters
    ----------
    timestamps
        1-D array-like of `datetime64`-compatible timestamps.
    values
        Array-like of values; first dimension must match *timestamps*.
    dtype
        Optional NumPy dtype to cast *values* to.
    initial
        Optional initial value.
    name
        Optional source name.
    """

    __slots__ = ("_timestamps", "_values")

    def __init__(
        self,
        timestamps: ArrayLike,
        values: ArrayLike,
        *,
        dtype: type | np.dtype | None = None,
        initial: ArrayLike | None = None,
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

        shape = cast(tuple[int, ...], vals.shape[1:])
        super().__init__(shape, dt, initial=initial, name=name)
        self._timestamps = ts
        self._values = vals

    @classmethod
    def from_arrays(
        cls,
        timestamps: ArrayLike,
        values: ArrayLike,
        *,
        dtype: type | np.dtype | None = None,
        initial: ArrayLike | None = None,
        name: str | None = None,
    ) -> ArraySource:
        """Construct from in-memory arrays."""
        return cls(timestamps=timestamps, values=values, dtype=dtype, initial=initial, name=name)

    def subscribe(self) -> tuple[AsyncIterator[tuple[np.datetime64, Any]], AsyncIterator[Any]]:
        return self._historical_gen(), empty_live_gen()

    async def _historical_gen(self) -> AsyncIterator[tuple[np.datetime64, Any]]:
        for i in range(len(self._timestamps)):
            yield self._timestamps[i], self._values[i]
