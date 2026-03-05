"""Array-bundle payload-timestamp source."""

from __future__ import annotations

import pickle
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from ..series import AnyShape, Array, Series
from ..source import Source, SourceItem


class ArrayBundleSource[Shape: AnyShape, T: np.generic](Source[Shape, T]):
    """Source backed by ``(timestamps, values)`` array bundles."""

    __slots__ = ("_timestamps", "_values")

    _timestamps: Array[tuple[int], np.datetime64]
    _values: Array[tuple[int, *Shape], T]

    def __init__(
        self,
        timestamps: ArrayLike,
        values: ArrayLike,
        series: Series[Shape, T],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(series, name=name, timestamp_mode="payload")
        ts = np.asarray(timestamps, dtype="datetime64[ns]")
        vals = np.asarray(values, dtype=series.dtype)

        if ts.ndim != 1:
            raise ValueError(f"timestamps must be 1-dimensional, got ndim={ts.ndim}")
        if vals.ndim < 1:
            raise ValueError(f"values must have at least 1 dimension, got ndim={vals.ndim}")
        if len(ts) != len(vals):
            raise ValueError(f"timestamps length {len(ts)} does not match values length {len(vals)}")
        if vals.shape[1:] != series.shape:
            raise ValueError(f"values shape tail {vals.shape[1:]} does not match series shape {series.shape}")

        self._timestamps = cast(Array[tuple[int], np.datetime64], ts)
        self._values = cast("Array[tuple[int, *Shape], T]", vals)

    @classmethod
    def from_arrays(
        cls,
        timestamps: ArrayLike,
        values: ArrayLike,
        series: Series[Shape, T],
        *,
        name: str | None = None,
    ) -> ArrayBundleSource[Shape, T]:
        """Constructs from in-memory arrays."""
        return cls(timestamps=timestamps, values=values, series=series, name=name)

    @classmethod
    def from_pickle(
        cls,
        path: str | Path,
        series: Series[Shape, T],
        *,
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
            series=series,
            name=name,
        )

    async def stream(self) -> AsyncIterator[SourceItem[Shape, T]]:
        for i in range(len(self._timestamps)):
            yield SourceItem(value=self._values[i], timestamp=self._timestamps[i])
