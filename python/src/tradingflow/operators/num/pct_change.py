"""Percentage-change operator — element-wise linear returns across ticks."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class PctChange(NativeOperator):
    """Element-wise linear return across ticks.

    Emits `a / a_{offset steps ago} - 1` on every tick, maintaining a
    ring buffer of the last `offset` input arrays.  Output is `NaN` for
    the first `offset` ticks.  `offset` must be at least `1`.

    This is the linear-return counterpart of
    [`Diff`][tradingflow.operators.num.diff.Diff]: `PctChange` yields
    `p_t / p_{t-k} - 1`, while `Log -> Diff` yields `log p_t - log p_{t-k}`.

    Parameters
    ----------
    a
        Handle to a float Array node.
    offset
        Number of ticks to look back.  Default `1`.
    """

    def __init__(self, a: Handle, offset: int = 1) -> None:
        super().__init__(
            native_id="pct_change",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params={"offset": offset},
        )
