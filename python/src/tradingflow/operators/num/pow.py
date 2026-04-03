"""Power operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle


class Pow(NativeOperator):
    """Element-wise power: `a ** n`."""

    def __init__(self, a: Handle, n: float) -> None:
        super().__init__(kind="pow", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"n": n})
