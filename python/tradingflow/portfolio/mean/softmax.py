import numpy as np

from .._base import MeanPortfolio, mean_portfolio


def weights_of(mu: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax of the non-negative part of the predicted returns."""
    weights = np.exp(np.maximum(mu, 0.0) / temperature)
    total = weights.sum()
    return weights / total if total > 0 else weights


def build(*, temperature: float = 1.0, **kwargs) -> MeanPortfolio:
    """Constructs a softmax-weight portfolio.

    `temperature` sets how sharply conviction translates into weight: as it
    approaches zero the book collapses onto the single best-predicted stock,
    and as it grows the weights flatten toward equal. Unlike
    `proportional` every eligible stock keeps a positive weight, so the
    portfolio stays diversified even when one prediction dominates.
    """
    assert temperature > 0.0, "temperature must be positive"
    return mean_portfolio(lambda mu: weights_of(mu, temperature), **kwargs)
