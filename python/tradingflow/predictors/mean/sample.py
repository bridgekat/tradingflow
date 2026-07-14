"""Sample mean predictor (pyhost contract)."""

import numpy as np

from ._base import MeanPredictor, MeanPredictorState


class Sample(MeanPredictor[np.ndarray]):
    """Predict the target as the sample mean of past target values.

    Ignores features entirely.  Useful as a baseline.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            fit_fn=_fit_fn,
            predict_fn=_predict_fn,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample mean of target per stock (NaN-robust). Ignores features."""
    # y: (T, N)
    return np.nanmean(y, axis=0)  # (N,)


def _predict_fn(state: MeanPredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Return the sample mean directly."""
    return params


def build(**kwargs) -> Sample:
    """Construct a :class:`Sample` mean predictor.

    Build kwargs
    ------------
    num_stocks : int
    num_features : int
    universe_size : int
    target_offset : int
    refit_every : int, optional (default 1)
    max_periods : int | None, optional
    min_periods : int | None, optional
    """
    return Sample(**kwargs)
