"""Rolling operator factories.

All rolling operators take a Series input and produce a Series output.
They require a floating-point dtype (``float32`` or ``float64``).
"""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


def rolling_sum(a: Handle, window: int) -> NativeOperator:
    """Element-wise rolling sum of the last *window* values.

    If any value in the window is NaN, the output for that element is NaN.
    """
    return NativeOperator(
        kind="rolling_sum", inputs=(a,), shape=a.shape, dtype=a.dtype,
        params={"window": window},
    )


def rolling_mean(a: Handle, window: int) -> NativeOperator:
    """Element-wise rolling mean of the last *window* values.

    If any value in the window is NaN, the output for that element is NaN.
    """
    return NativeOperator(
        kind="rolling_mean", inputs=(a,), shape=a.shape, dtype=a.dtype,
        params={"window": window},
    )


def rolling_variance(a: Handle, window: int) -> NativeOperator:
    """Element-wise rolling population variance of the last *window* values.

    If any value in the window is NaN, the output for that element is NaN.
    """
    return NativeOperator(
        kind="rolling_variance", inputs=(a,), shape=a.shape, dtype=a.dtype,
        params={"window": window},
    )


def rolling_covariance(a: Handle, window: int) -> NativeOperator:
    """Pairwise rolling covariance matrix of the last *window* values.

    Input must be 1-D with shape ``(K,)``. Output shape is ``(K, K)``.
    If any value in the window is NaN, the affected covariance entries are NaN.
    """
    if len(a.shape) != 1:
        raise ValueError("rolling_covariance requires 1-D input")
    k = a.shape[0]
    return NativeOperator(
        kind="rolling_covariance", inputs=(a,), shape=(k, k), dtype=a.dtype,
        params={"window": window},
    )


def ema(
    a: Handle,
    window: int,
    *,
    alpha: float | None = None,
    span: int | None = None,
    half_life: float | None = None,
) -> NativeOperator:
    """Window-normalised exponential moving average.

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
    return NativeOperator(
        kind="ema", inputs=(a,), shape=a.shape, dtype=a.dtype,
        params=params,
    )


def forward_fill(a: Handle) -> NativeOperator:
    """Forward-fill NaN values with the last valid observation (element-wise).

    If no valid value has been seen yet for an element, the output is NaN.
    """
    return NativeOperator(
        kind="forward_fill", inputs=(a,), shape=a.shape, dtype=a.dtype,
    )
