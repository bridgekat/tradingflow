"""Rolling mean operator."""

from __future__ import annotations

import numpy as np

from ...operator import NativeOperator
from ...types import Handle, NodeKind
from ._window import window_params as _window_params


class RollingMean(NativeOperator):
    """Element-wise rolling mean.

    Takes a Series input and outputs an Array.
    If any value in the window is NaN, the output for that element is NaN.

    Parameters
    ----------
    a
        Series input handle.
    window
        Window size: an `int` for a count-based window (last N elements),
        or a `numpy.timedelta64` for a time-delta-based window.
    """

    def __init__(self, a: Handle, window: int | np.timedelta64) -> None:
        super().__init__(
            native_id="rolling_mean",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params=_window_params(window),
        )
