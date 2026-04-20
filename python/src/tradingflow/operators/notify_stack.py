"""Notify-aware Stack operator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import Handle, NativeOperator, NodeKind


class NotifyStack(NativeOperator):
    """Stack N arrays along a new axis, filling non-produced slots with NaN.

    Message-passing counterpart to
    [`Stack`][tradingflow.operators.Stack]: on every compute, slots of
    inputs that did not produce in the current flush cycle are filled
    with `NaN`.  This separates freshly-updated state from stale
    carry-over.

    Typical use: pair with
    [`ForwardFill`][tradingflow.operators.num.ForwardFill] downstream
    to get "fresh where available, last-known otherwise" semantics —
    correct for suspended stocks, multi-frequency sensors, sparse
    event streams, and any cross-sectional aggregation where inputs
    update at heterogeneous cadences.

    Float dtype only: `NaN` is used as the "no update" sentinel.

    Parameters
    ----------
    inputs
        Sequence of upstream handles (at least one).  All must share
        the same float dtype and shape.
    axis
        Position of the new axis (default `0`).
    """

    def __init__(self, inputs: Sequence[Handle], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("NotifyStack requires at least one input.")
        dtype = np.dtype(inputs[0].dtype)
        if dtype.kind != "f":
            raise TypeError(f"NotifyStack requires a float dtype (got {dtype}); NaN is used as the no-update sentinel.")
        base_shape = list(inputs[0].shape)
        out_shape = base_shape[:axis] + [len(inputs)] + base_shape[axis:]
        super().__init__(
            native_id="notify_stack",
            inputs=tuple(inputs),
            kind=NodeKind.ARRAY,
            dtype=dtype,
            shape=tuple(out_shape),
            params={"axis": axis},
        )
