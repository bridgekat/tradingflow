"""Power operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class Pow(NativeOperator):
    """Element-wise power: `a ** n`."""

    def __init__(self, a: Handle, n: float) -> None:
        super().__init__(native_id="pow", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape, params={"n": n})
