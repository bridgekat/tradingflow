"""Stack operator."""

from __future__ import annotations

from collections.abc import Sequence

from ..operator import NativeOperator
from ..types import Handle


class Stack(NativeOperator):
    """Stack N arrays along a new axis.

    All inputs must have the same dtype and shape. The output shape
    inserts a new dimension of size ``len(inputs)`` at *axis*.

    Parameters
    ----------
    inputs
        Sequence of upstream handles (at least one).
    axis
        Position of the new axis (default ``0``).
    """

    def __init__(self, inputs: Sequence[Handle], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("Stack requires at least one input.")
        base_shape = list(inputs[0].shape)
        out_shape = base_shape[:axis] + [len(inputs)] + base_shape[axis:]
        super().__init__(
            kind="stack",
            inputs=tuple(inputs),
            shape=tuple(out_shape),
            dtype=inputs[0].dtype,
            params={"axis": axis},
        )
