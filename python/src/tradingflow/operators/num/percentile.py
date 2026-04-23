"""Percentile operator — NaN-preserving cross-sectional rank-to-percentile transform."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class Percentile(NativeOperator):
    """Cross-sectional rank-to-percentile transform on a 1-D float array.

    Non-NaN elements are ranked ascending and mapped to
    ``(rank + 0.5) / n_valid`` in `(0, 1)`.  NaN inputs propagate to NaN
    outputs — they do not occupy ranks, so the denominator only counts
    finite values and the percentile distribution is not squeezed by
    missing entries.

    Identical sort / NaN logic to
    [`Gaussianize`][tradingflow.operators.num.gaussianize.Gaussianize],
    just without the final ``Φ⁻¹`` step.

    Parameters
    ----------
    a
        Handle to a 1-D float Array node.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(
            native_id="percentile",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
        )
