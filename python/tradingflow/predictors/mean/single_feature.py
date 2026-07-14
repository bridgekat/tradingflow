"""Single-feature pass-through mean predictor (pyhost contract)."""

import numpy as np

from ._base import MeanPredictor


class SingleFeature(MeanPredictor[None]):
    """Return a single feature column directly as the prediction.

    A trivial pass-through predictor that selects one feature column and
    outputs it unchanged.  No fitting is performed.  ``max_periods`` and
    ``min_periods`` are fixed to ``1`` since only the latest feature row
    is used.  The ``target_series`` input is still required for alignment
    but is not consulted.
    """

    def __init__(self, *, feature_index: int = 0, **kwargs) -> None:
        super().__init__(
            fit_fn=_fit_fn,
            predict_fn=lambda state, features, params: features[:, feature_index],
            max_periods=1,
            min_periods=1,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray) -> None:
    """No-op: SingleFeature does not fit any parameters."""
    return None


def build(**kwargs) -> SingleFeature:
    """Construct a :class:`SingleFeature` mean predictor.

    Build kwargs
    ------------
    num_stocks : int
    num_features : int
    universe_size : int
    target_offset : int
    feature_index : int, optional (default 0)
        Index of the feature column to return as the prediction.
    refit_every : int, optional (default 1)

    Note: ``max_periods`` and ``min_periods`` are fixed to ``1``.
    """
    return SingleFeature(**kwargs)
