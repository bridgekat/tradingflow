"""Stack operator."""

from __future__ import annotations

from collections.abc import Sequence

from .. import Handle, NativeOperator, NodeKind


class Stack(NativeOperator):
    """Stack N arrays along a new axis.

    All inputs must have the same dtype and shape. The output shape
    inserts a new dimension of size `len(inputs)` at *axis*.

    Parameters
    ----------
    inputs
        Sequence of upstream handles (at least one).
    axis
        Position of the new axis (default `0`).
    """

    def __init__(self, inputs: Sequence[Handle], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("Stack requires at least one input.")
        base_shape = list(inputs[0].shape)
        out_shape = base_shape[:axis] + [len(inputs)] + base_shape[axis:]
        super().__init__(
            native_id="stack",
            inputs=tuple(inputs),
            kind=NodeKind.ARRAY,
            dtype=inputs[0].dtype,
            shape=tuple(out_shape),
            params={"axis": axis},
        )
