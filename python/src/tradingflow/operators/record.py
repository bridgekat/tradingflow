"""Record operator — accumulates Array values into a Series."""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


class Record(NativeOperator):
    """Record an Array node into a Series (accumulates history).

    The output is a `Series<T>` node that appends the input's value
    at each timestamp.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="record", inputs=(a,), shape=a.shape, dtype=a.dtype)
