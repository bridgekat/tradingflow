"""Rolling sum operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle


class RollingSum(NativeOperator):
    """Element-wise rolling sum of the last *window* values.

    Takes a Series input and outputs an Array.
    If any value in the window is NaN, the output for that element is NaN.
    """

    def __init__(self, a: Handle, window: int) -> None:
        super().__init__(
            kind="rolling_sum",
            inputs=(a,),
            shape=a.shape,
            dtype=a.dtype,
            params={"window": window},
        )
