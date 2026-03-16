"""Async-callable live source."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable

import numpy as np
from numpy.typing import ArrayLike

from ..series import AnyShape
from ..source import Source, empty_historical_gen

type _AsyncFactory = Callable[[], AsyncIterable[ArrayLike]]


class AsyncCallableSource[Shape: AnyShape, T: np.generic](Source[Shape, T]):
    """Live source wrapping a user-provided async iterable of raw values.

    The runtime assigns ingest timestamps when each item arrives (wall-clock time).

    Parameters
    ----------
    shape
        Shape of each emitted value element.  Use `()` for scalars.
    dtype
        NumPy dtype for the emitted values.
    factory
        Callable returning an `AsyncIterable[ArrayLike]` of raw values.
        Called once per [`subscribe`][.subscribe] invocation.
    name
        Optional source name; defaults to the class name.
    """

    __slots__ = ("_factory",)

    _factory: _AsyncFactory

    def __init__(
        self,
        shape: Shape,
        dtype: type[T] | np.dtype[T],
        factory: _AsyncFactory,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(shape, dtype, name=name)
        self._factory = factory

    def subscribe(self) -> tuple[AsyncIterator[tuple[np.datetime64, ArrayLike]], AsyncIterator[ArrayLike]]:
        """Returns a `(historical, live)` iterator pair; the historical iterator is empty."""
        return empty_historical_gen(), self._live_gen()

    async def _live_gen(self) -> AsyncIterator[ArrayLike]:
        iterable = self._factory()
        if not isinstance(iterable, AsyncIterable):
            raise TypeError(
                f"AsyncCallableSource '{self.name}' factory must return an AsyncIterable, "
                f"got {type(iterable).__name__}."
            )
        async for value in iterable:
            yield value
