"""Diff operator — element-wise first difference across ticks."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class Diff(NativeOperator):
    """Element-wise first difference across ticks.

    Emits `a - a_{offset steps ago}` on every tick, maintaining a ring
    buffer of the last `offset` input arrays.  Output is `NaN` for the
    first `offset` ticks.  `offset` must be at least `1`.

    Combined with [`Log`][tradingflow.operators.num.arithmetic.Log]
    upstream this produces log returns: `Log -> Diff`.

    Parameters
    ----------
    a
        Handle to a float Array node.
    offset
        Number of ticks to look back.  Default `1`.
    """

    def __init__(self, a: Handle, offset: int = 1) -> None:
        super().__init__(
            native_id="diff",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params={"offset": offset},
        )
