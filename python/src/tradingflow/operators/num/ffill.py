"""Forward-fill operator."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class ForwardFill(NativeOperator):
    """Forward-fill NaN values with the last valid observation (element-wise).

    Takes an Array input and outputs an Array of the same shape.
    On each tick, replaces NaN elements with the most recently seen
    non-NaN value for that position.  If no valid value has been seen
    yet for an element, the output is NaN.

    Parameters
    ----------
    a
        Handle to an Array node.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(
            native_id="forward_fill",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
        )
