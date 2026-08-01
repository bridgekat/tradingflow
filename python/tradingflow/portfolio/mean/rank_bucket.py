import numpy as np

from .._base import MeanPortfolio, mean_portfolio


def weights_of(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Equal weight across the stocks whose percentile rank falls in `[low, high)`."""
    n = len(values)
    if n == 0:
        return np.zeros(0)

    # Mid-bin percentile rank, so the smallest stock lands at 0.5/n and the
    # largest at (n - 0.5)/n — every stock sits strictly inside a bucket
    # rather than on its edge.
    order = np.argsort(values, kind="stable")
    rank = np.empty(n)
    rank[order] = (np.arange(n) + 0.5) / n

    weights = ((rank >= low) & (rank < high)).astype(np.float64)
    total = weights.sum()
    return weights / total if total > 0 else weights


def build(*, low: float, high: float, logarithmic: bool = False, **kwargs) -> MeanPortfolio:
    """Constructs an equal-weight percentile-bucket portfolio.

    The tool for measuring a feature rather than trading a prediction: run one
    per decile (`low=d/10, high=(d+1)/10`) and the spread between the top and
    bottom buckets is the feature's return, monotonicity and all. The input is
    read as a raw feature, so `logarithmic` defaults to `False` — only the
    ranking matters, which no monotone transform can change.
    """
    assert 0.0 <= low < high <= 1.0, f"expected 0 <= low < high <= 1, got {low} and {high}"
    return mean_portfolio(
        lambda values: weights_of(values, low, high),
        logarithmic=logarithmic,
        **kwargs,
    )
