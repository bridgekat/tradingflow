"""Concatenation operator."""

from __future__ import annotations

from collections.abc import Sequence

from ..operator import NativeOperator
from ..types import Handle


class Concat(NativeOperator):
    """Concatenate N arrays along an existing axis.

    All inputs must have the same dtype and matching shapes on every axis
    except *axis*.
    """

    def __init__(self, inputs: Sequence[Handle], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("Concat requires at least one input.")
        shapes = [inp.shape for inp in inputs]
        ndim = len(shapes[0])
        if not 0 <= axis < ndim:
            raise ValueError(f"axis {axis} is out of bounds for {ndim}-dimensional inputs.")
        out_shape = list(shapes[0])
        out_shape[axis] = sum(sh[axis] for sh in shapes)
        super().__init__(
            kind="concat",
            inputs=tuple(inputs),
            shape=tuple(out_shape),
            dtype=inputs[0].dtype,
            params={"axis": axis},
        )
