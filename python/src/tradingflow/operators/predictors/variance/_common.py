"""Shared utilities for variance (covariance) estimators.

All helpers are NaN-robust: observations flagged non-finite are
excluded from per-pair statistics.  Intended for internal use by the
concrete estimators in this package.
"""

import numpy as np


def sample_covariance(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-robust sample covariance using pairwise complete observations.

    Parameters
    ----------
    y
        Cross-sectional return matrix of shape ``(T, N)``.

    Returns
    -------
    S : np.ndarray
        ``(N, N)`` sample covariance matrix.
    centered : np.ndarray
        ``(T, N)`` mean-centered returns with non-finite entries
        replaced by 0 (so they contribute nothing to sums).
    finite : np.ndarray
        ``(T, N)`` boolean mask of originally-finite entries.
    """
    mean = np.nanmean(y, axis=0)
    centered = y - mean
    finite = np.isfinite(centered)
    centered = np.where(finite, centered, 0.0)
    indicator = finite.astype(np.float64)
    counts = indicator.T @ indicator
    S = (centered.T @ centered) / np.maximum(counts - 1.0, 1.0)
    return S, centered, finite


def correlation_from_covariance(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample correlation matrix and standard deviations from a covariance.

    Parameters
    ----------
    S
        ``(N, N)`` covariance matrix.

    Returns
    -------
    C : np.ndarray
        ``(N, N)`` correlation matrix, with diagonal forced to 1 and
        entries clipped to ``[-1, 1]``.
    stds : np.ndarray
        ``(N,)`` sample standard deviations.
    """
    diag = np.maximum(np.diag(S), 0.0)
    stds = np.sqrt(diag)
    stds_safe = np.where(stds > 0, stds, 1.0)
    C = S / np.outer(stds_safe, stds_safe)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0), stds


def single_index_covariance(y: np.ndarray) -> np.ndarray:
    """Single-index factor-model covariance estimator.

    Fits the factor model ``r_i(t) = alpha_i + beta_i * f(t) + eps_i(t)``
    stock-by-stock against the equal-weighted cross-sectional mean return
    ``f(t)``, and returns

        Sigma = sigma_f^2 * beta @ beta.T + diag(sigma_eps^2).

    All statistics are computed NaN-robustly from pairs of finite
    observations.

    Parameters
    ----------
    y
        Cross-sectional return matrix of shape ``(T, N)``.

    Returns
    -------
    np.ndarray
        ``(N, N)`` single-index covariance matrix.  Zero matrix if the
        factor is never observable.
    """
    _, N = y.shape

    # Equal-weighted cross-sectional mean as the market-factor proxy.
    f = np.nanmean(y, axis=1)

    # Time-series centering.
    y_c = y - np.nanmean(y, axis=0)
    f_c = f - np.nanmean(f)

    # Keep only rows where the factor is observable.
    f_valid = np.isfinite(f_c)
    if not f_valid.any():
        return np.zeros((N, N), dtype=np.float64)
    y_c = y_c[f_valid]
    f_c = f_c[f_valid]

    # Per-stock finiteness (the factor is already finite on these rows).
    valid = np.isfinite(y_c)
    y_fill = np.where(valid, y_c, 0.0)
    f_mat = np.where(valid, f_c[:, None], 0.0)

    # OLS beta per stock using only pairs where y_i is observed.
    num = (y_fill * f_mat).sum(axis=0)
    den = (f_mat * f_mat).sum(axis=0)
    beta = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)

    # Residual (idiosyncratic) variances.
    resid = y_fill - beta[None, :] * f_mat
    counts = valid.sum(axis=0)
    resid_ss = (resid * resid).sum(axis=0)
    sigma_eps_sq = np.where(counts > 2, resid_ss / np.maximum(counts - 2.0, 1.0), 0.0)

    # Market-factor variance.
    sigma_f_sq = float((f_c * f_c).sum() / max(len(f_c) - 1, 1))

    F = sigma_f_sq * np.outer(beta, beta)
    F[np.diag_indices(N)] += sigma_eps_sq
    return F


def ledoit_wolf_alpha(
    S: np.ndarray,
    F: np.ndarray,
    centered: np.ndarray,
    finite: np.ndarray,
) -> tuple[float, int]:
    """Ledoit-Wolf optimal linear-shrinkage intensity.

    Computes ``alpha = delta / (T_eff * ||S - F||_F^2)`` clipped to
    ``[0, 1]`` where ``delta = (1/T_eff) Σ_t ||x_t x_t^T - S||_F^2``.
    This is the general formula from Ledoit & Wolf (2004) Lemma 3.3
    that does not depend on the specific target ``F``.

    Parameters
    ----------
    S
        Sample covariance matrix.
    F
        Target matrix (same shape as ``S``).
    centered
        Mean-centered returns ``(T, N)`` with non-finite rows zeroed.
    finite
        Boolean finiteness mask ``(T, N)``.

    Returns
    -------
    alpha : float
        Optimal shrinkage intensity in ``[0, 1]``.
    T_eff : int
        Number of rows with at least one finite observation.
    """
    T = centered.shape[0]
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
    denom = float(np.sum((S - F) ** 2))
    if denom < 1e-30 or T_eff == 0:
        return 1.0, T_eff
    return float(np.clip(delta / (T_eff * denom), 0.0, 1.0)), T_eff
