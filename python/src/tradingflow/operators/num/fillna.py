"""Fill-NaN operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class Fillna(NativeOperator):
    """Replace NaN with `val`."""

    def __init__(self, a: Handle, val: float) -> None:
        super().__init__(native_id="nan_to_num", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape, params={"val": val})
