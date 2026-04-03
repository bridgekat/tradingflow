"""Rolling mean operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class RollingMean(NativeOperator):
    """Element-wise rolling mean of the last *window* values.

    Takes a Series input and outputs an Array.
    If any value in the window is NaN, the output for that element is NaN.
    """

    def __init__(self, a: Handle, window: int) -> None:
        super().__init__(
            native_id="rolling_mean",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params={"window": window},
        )
