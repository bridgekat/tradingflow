"""Drawdown from previous high since inception."""

from ... import Handle, NativeOperator, NodeKind


class Drawdown(NativeOperator):
    r"""Drawdown: \((P_t - M_t) / M_t\) where \(M_t = \max_{s \le t} P_s\).

    Always non-positive.  Zero when at a new high.

    Parameters
    ----------
    a
        Scalar Array input (e.g. portfolio market value).
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="drawdown", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=())
