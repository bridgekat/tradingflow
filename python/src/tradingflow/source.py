"""Source interface for data feeding into the computation graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class Source(ABC):
    """Abstract base for data sources.

    Parameters
    ----------
    dtype
        NumPy dtype for emitted values.
    shape
        Shape of each emitted value element. Use `()` for scalars.
    initial
        Initial value. Defaults to NaN for floats, zero otherwise.
    name
        Optional human-readable name.
    """

    __slots__ = ("_dtype", "_shape", "_initial", "_name")

    def __init__(
        self,
        dtype: type | np.dtype,
        shape: tuple[int, ...],
        *,
        initial: ArrayLike | None = None,
        name: str | None = None,
    ) -> None:
        self._dtype = np.dtype(dtype)
        self._shape = shape
        if initial is not None:
            self._initial = np.asarray(initial, dtype=self._dtype)
        elif np.issubdtype(self._dtype, np.floating):
            self._initial = np.full(shape, np.nan, dtype=self._dtype)
        else:
            self._initial = np.zeros(shape, dtype=self._dtype)
        self._name = name or type(self).__name__

    @abstractmethod
    def init(self, timestamp: int) -> tuple[
        AsyncIterator[tuple[np.datetime64, ArrayLike]],
        AsyncIterator[tuple[np.datetime64, ArrayLike]],
    ]:
        """Return a ``(historical, live)`` async-iterator pair.

        Both iterators yield ``(timestamp, value)`` tuples, matching the
        Rust ``Source::init`` which returns two ``Receiver<(i64, Event)>``
        channels.

        Parameters
        ----------
        timestamp
            Initial timestamp (nanoseconds since epoch).
        """
        ...

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype for emitted values."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape of each emitted value."""
        return self._shape

    @property
    def initial(self) -> np.ndarray:
        """Initial value array."""
        return self._initial

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name


async def empty_historical_gen() -> AsyncIterator[tuple[np.datetime64, Any]]:
    """Immediately-exhausting historical async generator."""
    return
    yield


async def empty_live_gen() -> AsyncIterator[tuple[np.datetime64, Any]]:
    """Immediately-exhausting live async generator."""
    return
    yield


class NativeSource:
    """Descriptor for a Rust-implemented source.

    Analogous to [`NativeOperator`][tradingflow.NativeOperator] -- carries
    `native_id` + `params` and is dispatched entirely on the native side. Not a
    [`Source`][tradingflow.Source] subclass (no Python async iterators).

    Parameters
    ----------
    native_id
        Source native dispatch string on the Rust side.
    dtype
        NumPy dtype string (default `"float64"`).
    shape
        Element shape (default `()`).
    params
        Source-specific parameters.
    name
        Optional human-readable name (defaults to *native_id*).
    """

    __slots__ = ("_native_id", "_dtype", "_shape", "_params", "_name")

    def __init__(
        self,
        native_id: str,
        *,
        dtype: str = "float64",
        shape: tuple[int, ...] = (),
        params: dict | None = None,
        name: str | None = None,
    ) -> None:
        self._native_id = native_id
        self._dtype = dtype
        self._shape = shape
        self._params = params or {}
        self._name = name or native_id

    @property
    def native_id(self) -> str:
        """Source native dispatch string."""
        return self._native_id

    @property
    def dtype(self) -> str:
        """NumPy dtype string."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape."""
        return self._shape

    @property
    def params(self) -> dict:
        """Source-specific parameters."""
        return self._params

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name
