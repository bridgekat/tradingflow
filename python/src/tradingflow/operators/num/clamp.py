"""Clamp operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class Clamp(NativeOperator):
    """Element-wise clamp to `[lo, hi]`."""

    def __init__(self, a: Handle, lo: float, hi: float) -> None:
        super().__init__(native_id="clamp", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape, params={"lo": lo, "hi": hi})
