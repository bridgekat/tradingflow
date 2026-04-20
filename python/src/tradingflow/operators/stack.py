"""Stack operators — stack N arrays along a new axis.

- [`Stack`][tradingflow.operators.Stack] — time-series semantics:
  copies all inputs on every trigger.
- [`StackSync`][tradingflow.operators.StackSync] — message-passing
  semantics: fills non-produced input slots with `NaN` (float-only).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import Handle, NativeOperator, NodeKind


class Stack(NativeOperator):
    """Stack N arrays along a new axis.

    All inputs must have the same dtype and shape. The output shape
    inserts a new dimension of size `len(inputs)` at *axis*.

    Parameters
    ----------
    inputs
        Sequence of upstream handles (at least one).
    axis
        Position of the new axis (default `0`).
    """

    def __init__(self, inputs: Sequence[Handle], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("Stack requires at least one input.")
        base_shape = list(inputs[0].shape)
        out_shape = base_shape[:axis] + [len(inputs)] + base_shape[axis:]
        super().__init__(
            native_id="stack",
            inputs=tuple(inputs),
            kind=NodeKind.ARRAY,
            dtype=inputs[0].dtype,
            shape=tuple(out_shape),
            params={"axis": axis},
        )


class StackSync(NativeOperator):
    """Stack N arrays along a new axis, filling non-produced slots with NaN.

    Message-passing counterpart to
    [`Stack`][tradingflow.operators.Stack]: on every compute, slots of
    inputs that did not produce in the current flush cycle are filled
    with `NaN`, so downstream sees only the *synchronised* slice of
    inputs that fired together.  This separates freshly-updated state
    from stale carry-over.

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
            raise ValueError("StackSync requires at least one input.")
        dtype = np.dtype(inputs[0].dtype)
        if dtype.kind != "f":
            raise TypeError(f"StackSync requires a float dtype (got {dtype}); NaN is used as the no-update sentinel.")
        base_shape = list(inputs[0].shape)
        out_shape = base_shape[:axis] + [len(inputs)] + base_shape[axis:]
        super().__init__(
            native_id="stack_sync",
            inputs=tuple(inputs),
            kind=NodeKind.ARRAY,
            dtype=dtype,
            shape=tuple(out_shape),
            params={"axis": axis},
        )
