import numpy as np

from .._base import MeanPortfolio, mean_portfolio


def weights_of(mu: np.ndarray) -> np.ndarray:
    """Weight proportional to predicted return, shorting nothing."""
    weights = np.maximum(mu, 0.0)
    total = weights.sum()
    return weights / total if total > 0 else weights


def build(**kwargs) -> MeanPortfolio:
    """Constructs a proportional-weight portfolio.

    Stocks with a non-positive predicted return get nothing; the rest split the
    book in proportion to what they are expected to return. Conviction-weighted
    and trivially cheap, but with no risk model it will happily concentrate the
    whole portfolio into one volatile name that happens to score highest.
    """
    return mean_portfolio(weights_of, **kwargs)
