"""Concat operators — concatenate N arrays along an existing axis.

- [`Concat`][tradingflow.operators.Concat] — time-series semantics:
  copies all inputs on every trigger.
- [`ConcatSync`][tradingflow.operators.ConcatSync] — message-passing
  semantics: fills non-produced input slots with `NaN` (float-only).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import Handle, NativeOperator, NodeKind


class Concat(NativeOperator):
    """Concatenate N arrays along an existing axis.

    Time-series semantics: on every trigger, the latest value of each
    input is copied into the output, regardless of which inputs actually
    produced this cycle.  See
    [`ConcatSync`][tradingflow.operators.ConcatSync] for the
    message-passing variant that fills non-produced slots with `NaN`.

    All inputs must have the same dtype and matching shapes on every axis
    except *axis*.
    """

    def __init__(self, inputs: Sequence[Handle], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("Concat requires at least one input.")
        shapes = [inp.shape for inp in inputs]
        ndim = len(shapes[0])
        if not 0 <= axis < ndim:
            raise ValueError(f"axis {axis} is out of bounds for {ndim}-dimensional inputs.")
        out_shape = list(shapes[0])
        out_shape[axis] = sum(sh[axis] for sh in shapes)
        super().__init__(
            native_id="concat",
            inputs=tuple(inputs),
            kind=NodeKind.ARRAY,
            dtype=inputs[0].dtype,
            shape=tuple(out_shape),
            params={"axis": axis},
        )


class ConcatSync(NativeOperator):
    """Concatenate N arrays along an existing axis, filling non-produced
    slots with NaN.

    Message-passing counterpart to
    [`Concat`][tradingflow.operators.Concat]: slots of inputs that did
    not produce in the current flush cycle are filled with `NaN`, so
    downstream sees only the *synchronised* slice of inputs that fired
    together.  See [`StackSync`][tradingflow.operators.StackSync] for
    the general motivation.

    Float dtype only: `NaN` is used as the "no update" sentinel.

    Parameters
    ----------
    inputs
        Sequence of upstream handles (at least one).  All must share
        the same float dtype and matching shapes on every axis except
        *axis*.
    axis
        Axis along which to concatenate (default `0`).
    """

    def __init__(self, inputs: Sequence[Handle], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("ConcatSync requires at least one input.")
        dtype = np.dtype(inputs[0].dtype)
        if dtype.kind != "f":
            raise TypeError(
                f"ConcatSync requires a float dtype (got {dtype}); NaN is used as the no-update sentinel."
            )
        shapes = [inp.shape for inp in inputs]
        ndim = len(shapes[0])
        if not 0 <= axis < ndim:
            raise ValueError(f"axis {axis} is out of bounds for {ndim}-dimensional inputs.")
        out_shape = list(shapes[0])
        out_shape[axis] = sum(sh[axis] for sh in shapes)
        super().__init__(
            native_id="concat_sync",
            inputs=tuple(inputs),
            kind=NodeKind.ARRAY,
            dtype=dtype,
            shape=tuple(out_shape),
            params={"axis": axis},
        )
