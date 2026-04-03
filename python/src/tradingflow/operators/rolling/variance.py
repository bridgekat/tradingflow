"""Rolling variance operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class RollingVariance(NativeOperator):
    """Element-wise rolling population variance of the last *window* values.

    Takes a Series input and outputs an Array.
    If any value in the window is NaN, the output for that element is NaN.
    """

    def __init__(self, a: Handle, window: int) -> None:
        super().__init__(
            native_id="rolling_variance",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params={"window": window},
        )
