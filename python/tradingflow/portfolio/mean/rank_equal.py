import numpy as np

from .._base import MeanPortfolio, mean_portfolio


def weights_of(mu: np.ndarray, top_fraction: float) -> np.ndarray:
    """Equal weight across the top `top_fraction` of positively-predicted stocks."""
    weights = np.zeros_like(mu)
    positive = int((mu > 0).sum())
    if positive == 0:
        return weights

    k = round(top_fraction * positive)
    weights[np.argsort(-np.maximum(mu, 0.0))[:k]] = 1.0
    total = weights.sum()
    return weights / total if total > 0 else weights


def build(*, top_fraction: float = 0.1, **kwargs) -> MeanPortfolio:
    """Constructs an equal-weight top-fraction portfolio.

    Uses the predictions only to *rank*, then discards their magnitudes. That
    throws away information, which is the point: a predictor may order stocks
    far better than it estimates how much they will return, and equal weighting
    is immune to a single wild prediction.
    """
    assert 0.0 < top_fraction <= 1.0, "top_fraction must be in (0, 1]"
    return mean_portfolio(lambda mu: weights_of(mu, top_fraction), **kwargs)
