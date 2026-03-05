"""Async-callable ingest-timestamp source."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable

import numpy as np
from numpy.typing import ArrayLike

from ..series import AnyShape, Series
from ..source import Source, SourceItem

type _AsyncFactory = Callable[[], AsyncIterable[ArrayLike]]


class AsyncCallableSource[Shape: AnyShape, T: np.generic](Source[Shape, T]):
    """Wraps a user-provided async iterable of raw values."""

    __slots__ = ("_factory",)

    _factory: _AsyncFactory

    def __init__(
        self,
        series: Series[Shape, T],
        factory: _AsyncFactory,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(series, name=name, timestamp_mode="ingest")
        self._factory = factory

    async def stream(self) -> AsyncIterator[SourceItem[Shape, T]]:
        iterable = self._factory()
        if not isinstance(iterable, AsyncIterable):
            raise TypeError(
                f"AsyncCallableSource '{self.name}' factory must return an AsyncIterable, "
                f"got {type(iterable).__name__}."
            )
        async for value in iterable:
            yield SourceItem(value=value)
