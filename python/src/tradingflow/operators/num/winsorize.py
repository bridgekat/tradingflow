"""Winsorize operator — cross-sectional percentile clipping."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class Winsorize(NativeOperator):
    """Cross-sectional percentile clipping on a 1-D float array.

    Non-NaN entries are sorted; values below the p-quantile are clipped
    up to the p-quantile itself (and symmetrically for values above the
    (1-p)-quantile).  NaN inputs propagate to NaN outputs.  Identical
    sort / NaN logic to
    [`Percentile`][tradingflow.operators.num.percentile.Percentile],
    but preserves magnitudes — values within the central `(p, 1-p)`
    range pass through unchanged.

    Typical use: cap tail leverage on daily cross-sectional factor
    values or returns before a pooled OLS fit.  `p = 0.01` (1st / 99th
    percentile) and `p = 0.025` (2.5 / 97.5) are common choices.  The
    clip bounds adapt to each day's cross-section automatically, so
    high-vol days winsorize at wider absolute bounds than quiet days.

    Parameters
    ----------
    a
        Handle to a 1-D float Array node.
    p
        Quantile threshold in `[0, 0.5)`.  `p = 0` is a no-op.
    """

    def __init__(self, a: Handle, *, p: float = 0.01) -> None:
        super().__init__(
            native_id="winsorize",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
            params={"p": p},
        )
