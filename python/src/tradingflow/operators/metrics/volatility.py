"""Volatility since inception."""

from ... import Handle, NativeOperator, NodeKind


class Volatility(NativeOperator):
    """Population standard deviation of period returns since inception.

    Parameters
    ----------
    a
        Scalar Array input (e.g. portfolio market value).
    clock
        Clock source handle. The operator only emits on clock ticks.
    """

    def __init__(self, a: Handle, clock: Handle) -> None:
        super().__init__(native_id="volatility", inputs=(a, clock), kind=NodeKind.ARRAY, dtype=a.dtype, shape=())
