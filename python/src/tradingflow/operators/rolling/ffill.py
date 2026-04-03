"""Forward-fill operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class ForwardFill(NativeOperator):
    """Forward-fill NaN values with the last valid observation (element-wise).

    Takes a Series input and outputs an Array.
    If no valid value has been seen yet for an element, the output is NaN.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(
            native_id="forward_fill",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
        )
