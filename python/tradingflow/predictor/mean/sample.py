import numpy as np

from ._base import MeanPredictor


def fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-stock sample mean of the target over the window, ignoring features."""
    return np.nanmean(y, axis=0)


def predict(features: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params


def build(*, retain_features: bool = False, **kwargs) -> MeanPredictor:
    """Constructs a sample-mean predictor — the do-nothing baseline every other
    mean predictor has to beat.

    It fits from the target alone, so it retains no feature history and does
    not withhold a stock whose features are unusable.
    """
    return MeanPredictor(
        fit=fit,
        predict=predict,
        retain_features=retain_features,
        **kwargs,
    )
