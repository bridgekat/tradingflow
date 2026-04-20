"""Rolling sum operator."""

from __future__ import annotations

import numpy as np

from ... import Handle, NativeOperator, NodeKind
from ._window import window_params as _window_params


class RollingSum(NativeOperator):
    """Element-wise rolling sum.

    Takes a Series input and outputs an Array.
    If any value in the window is NaN, the output for that element is NaN.

    Parameters
    ----------
    a
        Series input handle.
    window
        Window size: an `int` for a count-based window (last N elements),
        or a `numpy.timedelta64` for a time-delta-based window (all
        elements within the given duration of the most recent timestamp).
    """

    def __init__(self, a: Handle, window: int | np.timedelta64) -> None:
        super().__init__(
            native_id="rolling_sum",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params=_window_params(window),
        )
