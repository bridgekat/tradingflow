"""Rolling variance operator."""

from __future__ import annotations

import numpy as np

from ...operator import NativeOperator
from ...types import Handle, NodeKind
from ._window import window_params as _window_params


class RollingVariance(NativeOperator):
    """Element-wise rolling population variance.

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
            native_id="rolling_variance",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params=_window_params(window),
        )
