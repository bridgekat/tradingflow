"""Diff operator — element-wise first difference across ticks."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class Diff(NativeOperator):
    """Element-wise one-step difference across ticks.

    Emits `a - a_prev` on every tick, maintaining the previous input
    array.  The output is `NaN` on the first tick (no previous value).

    This is the unit-step counterpart of a k-step diff: multi-step
    differences can be expressed as `Subtract(a, Last(Lag(Record(a), k)))`
    if needed — in practice, predictor alignment is handled by the
    `target_offset` parameter on
    [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor]
    and friends, not by varying the diff stride.

    Parameters
    ----------
    a
        Handle to a float Array node.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(
            native_id="diff",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
        )
