"""Source interface for data feeding into the computation graph.

[`Source`][tradingflow.Source] is the abstract base for Python-implemented
sources that produce events via async iterators. The Scenario registers them
as channel-based sources on the Rust side.
"""

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
    def subscribe(self) -> tuple[
        AsyncIterator[tuple[np.datetime64, ArrayLike]],
        AsyncIterator[ArrayLike],
    ]:
        """Return a `(historical, live)` async-iterator pair."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def initial(self) -> np.ndarray:
        return self._initial

    @property
    def name(self) -> str:
        return self._name


async def empty_historical_gen() -> AsyncIterator[tuple[np.datetime64, Any]]:
    """Immediately-exhausting historical async generator."""
    return
    yield


async def empty_live_gen() -> AsyncIterator[Any]:
    """Immediately-exhausting live async generator."""
    return
    yield
