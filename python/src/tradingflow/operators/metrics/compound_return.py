"""Compound return since inception."""

from ... import Handle, NativeOperator, NodeKind


class CompoundReturn(NativeOperator):
    r"""Compound return: \((P_t / P_0)^{1/n} - 1\).

    Parameters
    ----------
    a
        Scalar Array input (e.g. portfolio market value).
    clock
        Clock source handle. The operator only emits on clock ticks.
    """

    def __init__(self, a: Handle, clock: Handle) -> None:
        super().__init__(native_id="compound_return", inputs=(a, clock), kind=NodeKind.ARRAY, dtype=a.dtype, shape=())
