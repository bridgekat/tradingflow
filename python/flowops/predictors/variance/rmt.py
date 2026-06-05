r"""Random matrix theory (RMT) covariance predictors (flowops host).

Both estimators diagonalize the sample correlation matrix and reduce the
contribution of eigenvalues below the Laloux-corrected Marchenko-Pastur
upper bound

\[
\lambda_{\max} = \sigma^{2}\left(1 + N/T + 2\sqrt{N/T}\right),
\]

with \(\sigma^{2} = 1 - \lambda_1/N\).  ``RMT0`` zeros sub-threshold
eigenvalues (Rosenow et al., 2002); ``RMTM`` replaces them with their mean
(Potters et al., 2005).  Both rescale by the sample standard deviations.
"""

import numpy as np

from ._base import VariancePredictor
from ._common import correlation_from_covariance, sample_covariance


class RMT0(VariancePredictor[np.ndarray]):
    """RMT covariance estimator with zero replacement (Rosenow et al.).

    Zeros out every eigenvalue of the sample correlation matrix below the
    Marchenko-Pastur bound, reconstructs a filtered correlation matrix,
    forces its diagonal to 1, and rescales by the sample standard
    deviations.  Ignores features.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            fit_fn=lambda x, y: _rmt_fit(y, mode="zero"),
            predict_fn=lambda state, x, params: params,
            **kwargs,
        )


class RMTM(VariancePredictor[np.ndarray]):
    """RMT covariance estimator with mean replacement (Potters et al.).

    Replaces every eigenvalue below the Marchenko-Pastur bound with the
    mean of the sub-threshold block - preserving the trace - then
    renormalizes off-diagonals so the filtered correlation has unit
    diagonal, and rescales by the sample standard deviations.  Ignores
    features.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            fit_fn=lambda x, y: _rmt_fit(y, mode="mean"),
            predict_fn=lambda state, x, params: params,
            **kwargs,
        )


def _rmt_fit(y: np.ndarray, *, mode: str) -> np.ndarray:
    """RMT-filter the sample correlation and return the filtered covariance.

    Parameters
    ----------
    y
        Target panel `(T, N)`.
    mode
        `"zero"` for RMT-0, `"mean"` for RMT-M.
    """
    T, N = y.shape
    S, _, _ = sample_covariance(y)
    C, stds = correlation_from_covariance(S)

    # Symmetrize for numerical safety before eigendecomposition.
    C = (C + C.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    lam1 = float(eigvals[-1])

    # Laloux-corrected Marchenko-Pastur bound for the null variance
    # sigma^2 = 1 - lambda_1 / N (removes the market mode from the null).
    if T <= 0:
        return S
    q = N / T
    sigma2 = max(1.0 - lam1 / N, 0.0)
    lam_max = sigma2 * (1.0 + q + 2.0 * np.sqrt(q))

    below = eigvals < lam_max
    eigvals_new = eigvals.copy()
    if mode == "zero":
        eigvals_new[below] = 0.0
    elif mode == "mean":
        if below.any():
            eigvals_new[below] = eigvals[below].mean()
    else:
        raise ValueError(f"unknown mode {mode!r}")

    H = (eigvecs * eigvals_new) @ eigvecs.T

    if mode == "zero":
        # Force the correlation diagonal to 1.
        C_filt = H.copy()
        np.fill_diagonal(C_filt, 1.0)
    else:
        # Renormalize so that C_filt has unit diagonal; preserves the
        # off-diagonal structure H produced by the mean replacement.
        h_diag = np.sqrt(np.maximum(np.diag(H), 1e-30))
        C_filt = H / np.outer(h_diag, h_diag)
        np.fill_diagonal(C_filt, 1.0)

    return C_filt * np.outer(stds, stds)


def build_rmt0(**kwargs) -> RMT0:
    """Construct an :class:`RMT0` covariance predictor."""
    return RMT0(**kwargs)


def build_rmtm(**kwargs) -> RMTM:
    """Construct an :class:`RMTM` covariance predictor."""
    return RMTM(**kwargs)


_VARIANTS = {"zero": RMT0, "RMT0": RMT0, "mean": RMTM, "RMTM": RMTM}


def build(**kwargs):
    """Construct an RMT covariance predictor.

    Build kwargs
    ------------
    mode : str, optional (default "zero")
        ``"zero"``/``"RMT0"`` selects :class:`RMT0`, ``"mean"``/``"RMTM"``
        selects :class:`RMTM`.
    num_stocks : int
    num_features : int
    universe_size : int
    target_offset : int
    refit_every : int, optional (default 1)
    max_periods : int | None, optional
    min_periods : int | None, optional
    """
    mode = kwargs.pop("mode", "zero")
    cls = _VARIANTS[mode]
    return cls(**kwargs)
