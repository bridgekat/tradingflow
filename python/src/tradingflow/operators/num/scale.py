"""Scale operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle


class Scale(NativeOperator):
    """Element-wise scale: `a * c`."""

    def __init__(self, a: Handle, c: float) -> None:
        super().__init__(kind="scale", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"c": c})
