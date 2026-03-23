"""Stack operator factory."""

from __future__ import annotations

from collections.abc import Sequence

from ..operator import NativeOperator
from ..types import Handle


def stack(inputs: Sequence[Handle], *, axis: int = 0) -> NativeOperator:
    """Stack N arrays along a new axis."""
    if not inputs:
        raise ValueError("stack requires at least one input.")
    base_shape = list(inputs[0].shape)
    out_shape = base_shape[:axis] + [len(inputs)] + base_shape[axis:]
    return NativeOperator(
        kind="stack",
        inputs=tuple(inputs),
        shape=tuple(out_shape),
        dtype=inputs[0].dtype,
        params={"axis": axis},
    )
