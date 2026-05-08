"""Standardize operator - cross-sectional z-score transform."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind


class Standardize(NativeOperator):
    """Cross-sectional z-score of a 1-D array.

    Each non-NaN element is replaced with ``(x - mean) / std``, where
    `mean` and `std` are the cross-sectional mean and population
    standard deviation of the non-NaN entries.  NaN inputs propagate to
    NaN outputs.  When fewer than two non-NaN entries are present, or
    all non-NaN entries are equal (`std == 0`), every output slot is
    NaN - the z-score is undefined and downstream `np.isfinite` masks
    will skip the row.

    Output has the same dtype and shape as the input.  Same shape and
    NaN convention as
    [`Gaussianize`][tradingflow.operators.num.gaussianize.Gaussianize]
    and
    [`Percentile`][tradingflow.operators.num.percentile.Percentile].
    The difference: `Standardize` is a linear rescale that preserves
    the input's distribution shape, whereas `Gaussianize` remaps ranks
    to a standard normal.  Pick `Standardize` when downstream code
    consumes magnitudes (linear regression, mean-variance objective)
    and the input is already approximately Gaussian.

    Parameters
    ----------
    a
        Handle to a 1-D Array node with a float dtype.
    """

    def __init__(self, a: Handle) -> None:
        assert len(a.shape) == 1, "Standardize requires a 1-D input"
        super().__init__(
            native_id="standardize",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=a.shape,
        )
