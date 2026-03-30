"""Iterator-based source -- feeds events from a Python iterable."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..source import Source, empty_live_gen


class IterSource(Source):
    """Source driven by an iterable of `(timestamp, value)` pairs.

    More flexible than `ArraySource` — supports lazy or computed
    timestamp sequences and arbitrary value shapes.

    Parameters
    ----------
    iterable
        Iterable of `(datetime64, array_like)` pairs. Timestamps must be
        in non-decreasing order. The iterable is materialised into a list
        at construction time, so it can be replayed across multiple
        `subscribe` calls.
    shape
        Shape of each emitted value element.
    dtype
        NumPy dtype for emitted values.
    initial
        Optional initial value. Defaults to NaN for floats, zero otherwise.
    name
        Optional human-readable name.
    """

    __slots__ = ("_iterable_factory",)

    def __init__(
        self,
        iterable: Iterable[tuple[np.datetime64, ArrayLike]] | Iterable[tuple[Any, ArrayLike]],
        shape: tuple[int, ...],
        dtype: type | np.dtype,
        *,
        initial: ArrayLike | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(shape, dtype, initial=initial, name=name)
        # Store as a list so it can be iterated multiple times (repeated runs).
        self._iterable_factory = list(iterable)

    def subscribe(self) -> tuple[AsyncIterator[tuple[np.datetime64, Any]], AsyncIterator[Any]]:
        return self._historical_gen(), empty_live_gen()

    async def _historical_gen(self) -> AsyncIterator[tuple[np.datetime64, Any]]:
        for ts, val in self._iterable_factory:
            yield np.datetime64(ts, "ns"), val
