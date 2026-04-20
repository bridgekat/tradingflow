"""Rolling covariance operator."""

from __future__ import annotations

import numpy as np

from ...operator import NativeOperator
from ...types import Handle, NodeKind
from ._window import window_params as _window_params


class RollingCovariance(NativeOperator):
    """Pairwise rolling covariance matrix.

    Takes a Series input and outputs an Array.
    Input must be 1-D with shape `(K,)`. Output shape is `(K, K)`.
    If any value in the window is NaN, the affected covariance entries are NaN.

    Parameters
    ----------
    a
        Series input handle (must be 1-D).
    window
        Window size: an `int` for a count-based window (last N elements),
        or a `numpy.timedelta64` for a time-delta-based window.
    """

    def __init__(self, a: Handle, window: int | np.timedelta64) -> None:
        if len(a.shape) != 1:
            raise ValueError("RollingCovariance requires 1-D input")
        k = a.shape[0]
        super().__init__(
            native_id="rolling_covariance",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=(k, k),
            params=_window_params(window),
        )
