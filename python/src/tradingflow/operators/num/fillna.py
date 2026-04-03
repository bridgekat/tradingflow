"""Fill-NaN operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle


class Fillna(NativeOperator):
    """Replace NaN with `val`."""

    def __init__(self, a: Handle, val: float) -> None:
        super().__init__(kind="nan_to_num", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"val": val})
