"""Ledoit-Wolf linear shrinkage covariance estimator."""

import numpy as np

from ..variance_predictor import VariancePredictor


class Shrinkage(VariancePredictor[np.ndarray]):
    """Ledoit-Wolf linear shrinkage covariance estimator.

    Computes ``Σ = α * F + (1 - α) * S`` where ``S`` is the
    sample covariance matrix and ``F`` is a structured target (scaled
    identity matching the average variance).  The shrinkage intensity
    ``α`` is estimated analytically following Ledoit & Wolf (2004).

    Parameters
    ----------
    features
        Stacked features, shape ``(num_stocks, num_features)``.
        Passed through but not used by this estimator.
    adjusted_prices
        Stacked forward-adjusted close prices, shape ``(num_stocks,)``.
    rebalance_period
        Produce output every N ticks.
    verbose
        If ``True``, print shrinkage diagnostics to stdout.
    """

    def __init__(
        self,
        universe,
        features,
        adjusted_prices,
        *,
        rebalance_period: int,
        max_samples: int,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            universe,
            features,
            adjusted_prices,
            fit_fn=lambda x, y: _fit_fn(y, verbose=verbose),
            predict_fn=lambda state, x, params: params,
            max_samples=max_samples,
            rebalance_period=rebalance_period,
        )


def _fit_fn(y: np.ndarray, *, verbose: bool = False) -> np.ndarray:
    """Ledoit-Wolf linear shrinkage toward scaled identity.

    Parameters
    ----------
    y
        Cross-sectional return matrix of shape ``(T, N)``.

    Returns
    -------
    np.ndarray
        Shrunk covariance matrix of shape ``(N, N)``.
    """
    T, N = y.shape

    # NaN-robust sample covariance (pairwise complete observations).
    mean = np.nanmean(y, axis=0)
    centered = y - mean
    finite = np.isfinite(centered)
    centered = np.where(finite, centered, 0.0)
    indicator = finite.astype(np.float64)
    counts = indicator.T @ indicator  # (N, N) pairwise counts
    S = (centered.T @ centered) / np.maximum(counts - 1, 1.0)

    # Shrinkage target: scaled identity (average variance on diagonal).
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # Shrinkage intensity (Ledoit-Wolf analytical formula).
    # delta = (1/T_eff) * sum_t ||x_t x_t' - S||_F^2
    # Use zero-filled centered rows so NaN positions contribute nothing.
    delta = 0.0
    T_eff = 0
    for t in range(T):
        if not finite[t].any():
            continue
        T_eff += 1
        xt = centered[t : t + 1].T  # (N, 1)
        M = xt @ xt.T - S
        delta += np.sum(M * M)
    if T_eff > 0:
        delta /= T_eff

    # Optimal shrinkage intensity.
    denom = np.sum((S - F) ** 2)
    if denom < 1e-30:
        alpha = 1.0
    else:
        alpha = min(max(delta / (T_eff * denom) if T_eff > 0 else 1.0, 0.0), 1.0)

    if verbose:
        print(f"  shrinkage: {T_eff}/{T} valid samples, {N} stocks, alpha={alpha:.4f}")

    return alpha * F + (1.0 - alpha) * S
