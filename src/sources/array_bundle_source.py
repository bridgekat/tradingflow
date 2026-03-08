"""Array-bundle historical source."""

from __future__ import annotations

import pickle
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from ..series import AnyShape, Array
from ..source import Source, empty_live_gen


class ArrayBundleSource[Shape: AnyShape, T: np.generic](Source[Shape, T]):
    """Historical source backed by ``(timestamps, values)`` array bundles.

    The element shape and dtype are inferred from *values*.

    Parameters
    ----------
    timestamps
        1-D array-like of ``datetime64``-compatible timestamps.
    values
        Array-like of values; first dimension must match *timestamps* and
        trailing dimensions determine the emitted element shape.
    dtype
        Optional NumPy dtype to cast *values* to.  Defaults to the natural
        dtype of the *values* array.
    name
        Optional source name; defaults to the class name.
    """

    __slots__ = ("_timestamps", "_values")

    _timestamps: Array[tuple[int], np.datetime64]
    _values: Array[tuple[int, *Shape], T]

    def __init__(
        self,
        timestamps: ArrayLike,
        values: ArrayLike,
        *,
        dtype: type[T] | np.dtype[T] | None = None,
        name: str | None = None,
    ) -> None:
        ts = np.asarray(timestamps, dtype="datetime64[ns]")
        raw = np.asarray(values)
        dt: np.dtype[T] = np.dtype(dtype) if dtype is not None else cast("np.dtype[T]", raw.dtype)
        vals = raw.astype(dt)

        if ts.ndim != 1:
            raise ValueError(f"timestamps must be 1-dimensional, got ndim={ts.ndim}")
        if vals.ndim < 1:
            raise ValueError(f"values must have at least 1 dimension, got ndim={vals.ndim}")
        if len(ts) != len(vals):
            raise ValueError(f"timestamps length {len(ts)} does not match values length {len(vals)}")

        shape = cast(Shape, vals.shape[1:])
        super().__init__(shape, dt, name=name)
        self._timestamps = cast(Array[tuple[int], np.datetime64], ts)
        self._values = cast("Array[tuple[int, *Shape], T]", vals)

    @classmethod
    def from_arrays(
        cls,
        timestamps: ArrayLike,
        values: ArrayLike,
        *,
        dtype: type[T] | np.dtype[T] | None = None,
        name: str | None = None,
    ) -> ArrayBundleSource[Shape, T]:
        """Constructs from in-memory arrays."""
        return cls(timestamps=timestamps, values=values, dtype=dtype, name=name)

    @classmethod
    def from_pickle(
        cls,
        path: str | Path,
        *,
        dtype: type[T] | np.dtype[T] | None = None,
        timestamps_key: str = "timestamps",
        values_key: str = "values",
        name: str | None = None,
    ) -> ArrayBundleSource[Shape, T]:
        """Constructs from a pickled mapping containing timestamps and values arrays."""
        with Path(path).open("rb") as file:
            payload = pickle.load(file)

        if not isinstance(payload, Mapping):
            raise ValueError("Pickle payload must be a mapping containing timestamps and values.")
        if timestamps_key not in payload or values_key not in payload:
            raise ValueError(f"Pickle payload must contain keys '{timestamps_key}' and '{values_key}'.")

        return cls(
            timestamps=cast(Any, payload[timestamps_key]),
            values=cast(Any, payload[values_key]),
            dtype=dtype,
            name=name,
        )

    def subscribe(self) -> tuple[AsyncIterator[tuple[np.datetime64, Any]], AsyncIterator[Any]]:
        """Returns a ``(historical, live)`` iterator pair; the live iterator is empty."""
        return self._historical_gen(), empty_live_gen()

    async def _historical_gen(self) -> AsyncIterator[tuple[np.datetime64, Any]]:
        for i in range(len(self._timestamps)):
            yield self._timestamps[i], self._values[i]
