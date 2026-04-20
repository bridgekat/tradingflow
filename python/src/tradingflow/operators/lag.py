"""Lag operator — outputs the value from N steps ago."""

from __future__ import annotations

from .. import Handle, NativeOperator, NodeKind


class Lag(NativeOperator):
    """Output the value from *offset* steps ago.

    If fewer than `offset + 1` values have been recorded, the output
    is filled with *fill*.

    Parameters
    ----------
    a
        Handle to a Series node.
    offset
        Number of steps to look back (default `1`).
    fill
        Value used when history is insufficient (default `0`).
    """

    def __init__(self, a: Handle, offset: int = 1, *, fill: float | int = 0) -> None:
        params: dict = {"offset": offset}
        if fill != 0:
            params["fill"] = fill
        super().__init__(
            native_id="lag", inputs=(a,), kind=NodeKind.SERIES, dtype=a.dtype, shape=a.shape, params=params
        )
