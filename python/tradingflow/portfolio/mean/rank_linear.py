import numpy as np

from .._base import MeanPortfolio, mean_portfolio


def weights_of(mu: np.ndarray, top_fraction: float) -> np.ndarray:
    """Linearly decaying weights across the top `top_fraction` of stocks."""
    weights = np.zeros_like(mu)
    positive = int((mu > 0).sum())
    if positive == 0:
        return weights

    k = round(top_fraction * positive)
    weights[np.argsort(-np.maximum(mu, 0.0))[:k]] = np.linspace(k, 1, k)
    total = weights.sum()
    return weights / total if total > 0 else weights


def build(*, top_fraction: float = 0.1, **kwargs) -> MeanPortfolio:
    """Constructs a rank-linear top-fraction portfolio.

    Like `rank_equal` it uses predictions only to rank, but tilts weight toward
    the top of that ranking rather than spreading it evenly — the best-ranked
    stock gets `k` units against the last one's `1`. A middle ground between
    trusting the ordering and trusting the magnitudes.
    """
    assert 0.0 < top_fraction <= 1.0, "top_fraction must be in (0, 1]"
    return mean_portfolio(lambda mu: weights_of(mu, top_fraction), **kwargs)
