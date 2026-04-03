"""Shift operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class Shift(NativeOperator):
    """Element-wise shift: `a + c`."""

    def __init__(self, a: Handle, c: float) -> None:
        super().__init__(native_id="shift", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape, params={"c": c})
