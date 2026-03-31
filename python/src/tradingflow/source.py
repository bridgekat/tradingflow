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
    shape
        Shape of each emitted value element. Use `()` for scalars.
    dtype
        NumPy dtype for emitted values.
    initial
        Initial value. Defaults to NaN for floats, zero otherwise.
    name
        Optional human-readable name.
    """

    __slots__ = ("_shape", "_dtype", "_initial", "_name")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: type | np.dtype,
        *,
        initial: ArrayLike | None = None,
        name: str | None = None,
    ) -> None:
        self._shape = shape
        self._dtype = np.dtype(dtype)
        if initial is not None:
            self._initial = np.asarray(initial, dtype=self._dtype)
        elif np.issubdtype(self._dtype, np.floating):
            self._initial = np.full(shape, np.nan, dtype=self._dtype)
        else:
            self._initial = np.zeros(shape, dtype=self._dtype)
        self._name = name or type(self).__name__

    @abstractmethod
    def init(self) -> tuple[
        AsyncIterator[tuple[np.datetime64, ArrayLike]],
        AsyncIterator[tuple[np.datetime64, ArrayLike]],
    ]:
        """Return a ``(historical, live)`` async-iterator pair.

        Both iterators yield ``(timestamp, value)`` tuples, matching the
        Rust ``Source::init`` which returns two ``Receiver<(i64, Event)>``
        channels.
        """
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape of each emitted value."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype for emitted values."""
        return self._dtype

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
    `kind` + `params` and is dispatched entirely on the native side. Not a
    [`Source`][tradingflow.Source] subclass (no Python async iterators).

    Parameters
    ----------
    kind
        Source kind string dispatched on the Rust side.
    dtype
        NumPy dtype string (default `"float64"`).
    shape
        Element shape (default `()`).
    params
        Source-specific parameters.
    name
        Optional human-readable name (defaults to *kind*).
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
        """Source kind string."""
        return self._kind

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
