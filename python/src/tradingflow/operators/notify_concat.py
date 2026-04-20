"""Notify-aware Concat operator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import Handle, NativeOperator, NodeKind


class NotifyConcat(NativeOperator):
    """Concatenate N arrays along an existing axis, filling non-produced
    slots with NaN.

    Message-passing counterpart to
    [`Concat`][tradingflow.operators.Concat]: slots of inputs that did
    not produce in the current flush cycle are filled with `NaN`.
    See [`NotifyStack`][tradingflow.operators.NotifyStack] for the
    general motivation.

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
            raise ValueError("NotifyConcat requires at least one input.")
        dtype = np.dtype(inputs[0].dtype)
        if dtype.kind != "f":
            raise TypeError(
                f"NotifyConcat requires a float dtype (got {dtype}); NaN is used as the no-update sentinel."
            )
        shapes = [inp.shape for inp in inputs]
        ndim = len(shapes[0])
        if not 0 <= axis < ndim:
            raise ValueError(f"axis {axis} is out of bounds for {ndim}-dimensional inputs.")
        out_shape = list(shapes[0])
        out_shape[axis] = sum(sh[axis] for sh in shapes)
        super().__init__(
            native_id="notify_concat",
            inputs=tuple(inputs),
            kind=NodeKind.ARRAY,
            dtype=dtype,
            shape=tuple(out_shape),
            params={"axis": axis},
        )
