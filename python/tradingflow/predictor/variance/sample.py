import numpy as np

from ._base import VariancePredictor, covariance_predictor
from ._common import sample_covariance


def fit(y: np.ndarray) -> np.ndarray:
    """NaN-robust sample covariance over pairwise complete observations."""
    return sample_covariance(y)[0]


def build(**kwargs) -> VariancePredictor:
    """Constructs a sample covariance predictor — the *Markowitz* direct
    estimator of Pantaleo et al. (2010), and the baseline the structured
    estimators improve on."""
    return covariance_predictor(fit, **kwargs)
