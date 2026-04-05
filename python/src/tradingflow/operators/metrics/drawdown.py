"""Drawdown from previous high since inception."""

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class Drawdown(NativeOperator):
    """Drawdown: ``(current - running_max) / running_max``.

    Always non-positive.  Zero when at a new high.

    Parameters
    ----------
    a
        Scalar Array input (e.g. portfolio market value).
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="drawdown", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=())
