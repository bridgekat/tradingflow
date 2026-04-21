"""Exponential moving average operator."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class EMA(NativeOperator):
    r"""Window-normalized exponential moving average.

    Takes a Series input and outputs an Array.

    Exactly one of *alpha*, *span*, or *half_life* must be provided:

    - `alpha` — explicit smoothing factor \(\alpha \in (0, 1]\).
    - `span` — equivalent to \(\alpha = 2 / (\text{span} + 1)\) (pandas convention).
    - `half_life` — equivalent to \(\alpha = 1 - \exp(-\ln 2 / \text{half\_life})\).

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
            native_id="ema",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params=params,
        )
