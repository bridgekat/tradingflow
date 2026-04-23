"""Percentage-change operator — element-wise linear returns across ticks."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class PctChange(NativeOperator):
    """Element-wise one-step linear return across ticks.

    Emits `a / a_prev - 1` on every tick, maintaining the previous input
    array.  The output is `NaN` on the first tick (no previous value).

    This is the linear-return counterpart of
    [`Diff`][tradingflow.operators.num.diff.Diff]: `PctChange` yields
    `p_t / p_{t-1} - 1`, while `Log -> Diff` yields `log p_t - log p_{t-1}`.

    Parameters
    ----------
    a
        Handle to a float Array node.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(
            native_id="pct_change",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
        )
