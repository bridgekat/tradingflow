"""Clamp operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle


class Clamp(NativeOperator):
    """Element-wise clamp to `[lo, hi]`."""

    def __init__(self, a: Handle, lo: float, hi: float) -> None:
        super().__init__(kind="clamp", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"lo": lo, "hi": hi})
