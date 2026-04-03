"""Rolling covariance operator."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle


class RollingCovariance(NativeOperator):
    """Pairwise rolling covariance matrix of the last *window* values.

    Takes a Series input and outputs an Array.
    Input must be 1-D with shape ``(K,)``. Output shape is ``(K, K)``.
    If any value in the window is NaN, the affected covariance entries are NaN.
    """

    def __init__(self, a: Handle, window: int) -> None:
        if len(a.shape) != 1:
            raise ValueError("RollingCovariance requires 1-D input")
        k = a.shape[0]
        super().__init__(
            kind="rolling_covariance",
            inputs=(a,),
            shape=(k, k),
            dtype=a.dtype,
            params={"window": window},
        )
