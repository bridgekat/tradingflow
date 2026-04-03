"""Forward-fill operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle


class ForwardFill(NativeOperator):
    """Forward-fill NaN values with the last valid observation (element-wise).

    Takes a Series input and outputs an Array.
    If no valid value has been seen yet for an element, the output is NaN.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(
            kind="forward_fill",
            inputs=(a,),
            shape=a.shape,
            dtype=a.dtype,
        )
