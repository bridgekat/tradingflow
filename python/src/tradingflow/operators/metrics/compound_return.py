"""Compound return since inception."""

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class CompoundReturn(NativeOperator):
    """Compound return: ``(current / first)^(1/n) - 1``.

    Parameters
    ----------
    a
        Scalar Array input (e.g. portfolio market value).
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="compound_return", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=())
