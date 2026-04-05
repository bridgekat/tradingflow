"""Average return since inception."""

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class AverageReturn(NativeOperator):
    """Arithmetic mean of period returns since inception.

    Parameters
    ----------
    a
        Scalar Array input (e.g. portfolio market value).
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="average_return", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=())
