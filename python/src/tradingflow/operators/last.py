"""Last operator — extracts the most recent element from a Series."""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


class Last(NativeOperator):
    """Extract the most recent value from a Series as an Array.

    If the series is empty, the output is filled with *fill*.

    Parameters
    ----------
    a
        Handle to a Series node.
    fill
        Value used when the series is empty (default ``0``).
    """

    def __init__(self, a: Handle, *, fill: float | int = 0) -> None:
        params = {"fill": fill} if fill != 0 else {}
        super().__init__(kind="last", inputs=(a,), shape=a.shape, dtype=a.dtype, params=params)
