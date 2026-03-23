"""Lag operator — outputs the value from N steps ago."""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


def lag(a: Handle, offset: int = 1, *, fill: float | int = 0) -> NativeOperator:
    """Output the value from *offset* steps ago.

    If fewer than ``offset + 1`` values have been recorded, the output
    is filled with *fill*.

    Parameters
    ----------
    a
        Handle to a Series node.
    offset
        Number of steps to look back (default ``1``).
    fill
        Value used when history is insufficient (default ``0``).
    """
    params: dict = {"offset": offset}
    if fill != 0:
        params["fill"] = fill
    return NativeOperator(kind="lag", inputs=(a,), shape=a.shape, dtype=a.dtype, params=params)
