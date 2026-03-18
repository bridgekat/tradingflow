"""Core interface for data sources which generate observable values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .series import AnyShape, Array


class Source[Shape: AnyShape, T: np.generic](ABC):
    """Abstract base class for sources that produce observable values.

    A source declares the element `shape`, `dtype`, and *initial value* of the
    values it will emit.  [`Scenario`][tradingflow.Scenario] creates the target
    [`Observable`][tradingflow.Observable] when the source is registered via
    [`Scenario.add_source`][tradingflow.Scenario.add_source].

    Subclasses implement [`subscribe`][.subscribe] to return a `(historical, live)`
    async-iterator pair.  The historical iterator must yield
    `(timestamp, value)` pairs in strictly increasing timestamp order.
    The live iterator must yield raw values, which will be stamped at ingest
    time by [`Scenario`][tradingflow.Scenario].  The two iterators must cover
    complementary, non-overlapping segments of the same time series, split
    at some instant during the execution of [`subscribe`][.subscribe].

    Parameters
    ----------
    shape
        Shape of each emitted value element.  Use `()` for scalars.
    dtype
        NumPy dtype for the emitted values (e.g. `np.float64`).
    initial
        Initial value for the observable.  Defaults to NaN for floating-point
        dtypes and zero for others.
    name
        Optional human-readable name used in diagnostics and error messages;
        defaults to the class name.
    """

    __slots__ = ("_shape", "_dtype", "_initial", "_name")

    _shape: Shape
    _dtype: np.dtype[T]
    _initial: Array[Shape, T]
    _name: str

    def __init__(
        self,
        shape: Shape,
        dtype: type[T] | np.dtype[T],
        *,
        initial: ArrayLike | None = None,
        name: str | None = None,
    ) -> None:
        self._shape = shape
        self._dtype = np.dtype(dtype)
        if initial is not None:
            self._initial = np.asarray(initial, dtype=self._dtype)  # type: ignore[assignment]
        elif np.issubdtype(self._dtype, np.floating):
            self._initial = np.full(shape, np.nan, dtype=self._dtype)  # type: ignore[assignment]
        else:
            self._initial = np.zeros(shape, dtype=self._dtype)  # type: ignore[assignment]
        self._name = name or type(self).__name__

    @abstractmethod
    def subscribe(self) -> tuple[AsyncIterator[tuple[np.datetime64, ArrayLike]], AsyncIterator[ArrayLike]]:
        """Returns a `(historical, live)` async-iterator pair."""
        raise NotImplementedError

    @property
    def shape(self) -> Shape:
        """Element shape of each emitted value."""
        return self._shape

    @property
    def dtype(self) -> np.dtype[T]:
        """NumPy dtype of emitted values."""
        return self._dtype

    @property
    def initial(self) -> Array[Shape, T]:
        """Initial value for the observable."""
        return self._initial

    @property
    def name(self) -> str:
        """Human-readable name for debugging."""
        return self._name


async def empty_historical_gen() -> AsyncIterator[tuple[np.datetime64, Any]]:
    """Immediately-exhausting historical async generator.

    Returns an async generator that yields nothing.  Intended for use in
    [`Source.subscribe`][tradingflow.Source.subscribe] implementations of purely-live sources.
    """
    return
    yield  # Required to make this an async generator


async def empty_live_gen() -> AsyncIterator[Any]:
    """Immediately-exhausting live async generator.

    Returns an async generator that yields nothing.  Intended for use in
    [`Source.subscribe`][tradingflow.Source.subscribe] implementations of purely-historical sources.
    """
    return
    yield  # Required to make this an async generator
