"""Gaussianize operator — cross-sectional rank-to-Gaussian transform."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class Gaussianize(NativeOperator):
    """Cross-sectional rank-to-Gaussian transform of a 1-D array.

    Each non-NaN element is replaced with
    ``Φ⁻¹((rank + 0.5) / n_valid)``, where `rank` is its 0-based
    ascending rank among non-NaN elements and `n_valid` is their count.
    NaN inputs propagate to NaN outputs, so they do not skew downstream
    regressions.

    Output has the same dtype and shape as the input.  The inverse CDF
    is a rational approximation (Acklam, max rel error ~1.15e-9)
    evaluated in ``f64``.

    Parameters
    ----------
    a
        Handle to a 1-D Array node with a float dtype.
    """

    def __init__(self, a: Handle) -> None:
        assert len(a.shape) == 1, "Gaussianize requires a 1-D input"
        super().__init__(
            native_id="gaussianize",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
        )
