"""Rolling window operators (Series to Series, float dtypes only)."""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


class RollingSum(NativeOperator):
    """Element-wise rolling sum of the last *window* values.

    If any value in the window is NaN, the output for that element is NaN.
    """

    def __init__(self, a: Handle, window: int) -> None:
        super().__init__(
            kind="rolling_sum",
            inputs=(a,),
            shape=a.shape,
            dtype=a.dtype,
            params={"window": window},
        )


class RollingMean(NativeOperator):
    """Element-wise rolling mean of the last *window* values.

    If any value in the window is NaN, the output for that element is NaN.
    """

    def __init__(self, a: Handle, window: int) -> None:
        super().__init__(
            kind="rolling_mean",
            inputs=(a,),
            shape=a.shape,
            dtype=a.dtype,
            params={"window": window},
        )


class RollingVariance(NativeOperator):
    """Element-wise rolling population variance of the last *window* values.

    If any value in the window is NaN, the output for that element is NaN.
    """

    def __init__(self, a: Handle, window: int) -> None:
        super().__init__(
            kind="rolling_variance",
            inputs=(a,),
            shape=a.shape,
            dtype=a.dtype,
            params={"window": window},
        )


class RollingCovariance(NativeOperator):
    """Pairwise rolling covariance matrix of the last *window* values.

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


class EMA(NativeOperator):
    """Window-normalized exponential moving average.

    Exactly one of *alpha*, *span*, or *half_life* must be provided:

    - ``alpha`` — explicit smoothing factor in ``(0, 1]``.
    - ``span`` — equivalent to ``alpha = 2 / (span + 1)`` (pandas convention).
    - ``half_life`` — equivalent to ``alpha = 1 - exp(-ln2 / half_life)``.

    Parameters
    ----------
    a
        Handle to a Series node.
    window
        Number of past ticks to keep in the normalisation window.
    alpha, span, half_life
        Smoothing specification (mutually exclusive).
    """

    def __init__(
        self,
        a: Handle,
        window: int,
        *,
        alpha: float | None = None,
        span: int | None = None,
        half_life: float | None = None,
    ) -> None:
        count = sum(x is not None for x in (alpha, span, half_life))
        if count != 1:
            raise ValueError("exactly one of alpha, span, or half_life must be provided")
        params: dict = {"window": window}
        if alpha is not None:
            params["alpha"] = alpha
        elif span is not None:
            params["span"] = span
        else:
            params["half_life"] = half_life
        super().__init__(
            kind="ema",
            inputs=(a,),
            shape=a.shape,
            dtype=a.dtype,
            params=params,
        )


class ForwardFill(NativeOperator):
    """Forward-fill NaN values with the last valid observation (element-wise).

    If no valid value has been seen yet for an element, the output is NaN.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(
            kind="forward_fill",
            inputs=(a,),
            shape=a.shape,
            dtype=a.dtype,
        )
