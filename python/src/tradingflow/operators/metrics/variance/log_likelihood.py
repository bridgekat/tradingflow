"""Gaussian log-likelihood evaluator for covariance-matrix predictions."""

from dataclasses import dataclass

import numpy as np

from ....views import ArrayView
from ....operator import Operator
from ....types import Handle, NodeKind


@dataclass(slots=True)
class LogLikelihoodState:
    num_stocks: int
    initialized: bool = False
    sigma_inv: np.ndarray | None = None
    prev_prices: np.ndarray | None = None
    log_det: float = 0.0
    sum_quad: float = 0.0
    count: int = 0


class LogLikelihood(
    Operator[
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        LogLikelihoodState,
    ]
):
    """Gaussian negative-log-likelihood evaluator for covariance predictions.

    Evaluates covariance prediction quality.  On each prediction
    emission, caches ``log |Σ|`` and the Moore-Penrose pseudo-inverse
    ``Σ⁺`` (via a single SVD on the symmetrized Σ, equivalent to
    [`scipy.linalg.pinv`][scipy.linalg.pinv] on symmetric PSD inputs
    and handling rank-deficient Σ gracefully) and begins accumulating
    the daily Mahalanobis quadratic ``rᵀ Σ⁺ r``.  When the next
    prediction arrives, emits the period-averaged negative log-
    likelihood ``log |Σ| + (1/T) Σₜ rₜᵀ Σ⁺ rₜ`` — the
    multivariate-normal log-density with the ``N log(2π)`` constant
    and the ``1/2`` prefactor dropped (lower is better) — and updates
    the cache.  Matches the ``ll_metric`` in
    <https://osquant.com/papers/a-quants-guide-to-covariance-matrix-estimation/>.

    Output is a scalar (the period-averaged negative log-likelihood
    over one evaluation period).  ``Record(output)`` produces a
    directly plottable time series.

    Alignment guarantee
    -------------------
    After the initial warmup (first prediction caches Σ⁺/log|Σ| without
    emitting), the operator emits exactly once per prediction emission
    — **the same cadence as**
    [`MinimumVariance`][tradingflow.operators.metrics.variance.MinimumVariance],
    so corresponding records line up element-by-element.  Output is 0
    if no daily return was accumulated during the period (e.g. no
    stocks had finite covariance diagonal).

    Memory
    ------
    Both ``predictions`` and ``prices`` are ``Array`` inputs — the
    operator reads only the latest covariance and the latest cross-
    section of prices, caching one previous price tick in state to
    compute one-period returns.  No ``Record`` is required upstream.

    Parameters
    ----------
    predictions
        Live predicted covariance matrix from a variance predictor,
        shape ``(N, N)``.  Stocks excluded by the variance predictor
        have NaN on the diagonal.
    prices
        Live forward-adjusted close prices, shape ``(N,)``.
    """

    def __init__(
        self,
        predictions: Handle,
        prices: Handle,
    ) -> None:
        assert len(predictions.shape) == 2
        assert predictions.shape[0] == predictions.shape[1]
        assert len(prices.shape) == 1
        assert predictions.shape[0] == prices.shape[0]

        self._num_stocks = predictions.shape[0]

        super().__init__(
            inputs=(predictions, prices),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        timestamp: int,
    ) -> LogLikelihoodState:
        return LogLikelihoodState(
            num_stocks=self._num_stocks,
        )

    @staticmethod
    def _set_prediction(state: LogLikelihoodState, sigma: np.ndarray) -> None:
        """Compute and store Σ⁺ and log|Σ| from a covariance matrix.

        Stocks with non-finite covariance diagonal are excluded; their
        rows and columns of Σ⁺ remain zero so their returns contribute
        zero to the quadratic form.  If no stocks are eligible, Σ⁺ is
        all zeros and log|Σ| is zero.
        """
        n = state.num_stocks
        mask = np.isfinite(np.diag(sigma))
        sigma_inv = np.zeros((n, n), dtype=np.float64)
        log_det = 0.0
        if mask.any():
            log_det, sub_inv = _log_pdet_and_pinv(sigma[np.ix_(mask, mask)])
            sigma_inv[np.ix_(mask, mask)] = sub_inv
        state.sigma_inv = sigma_inv
        state.log_det = log_det

    @staticmethod
    def compute(
        state: LogLikelihoodState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        predictions, prices = inputs
        predictions_produced, prices_produced = produced

        # Accumulate one-period Mahalanobis quadratic on price ticks.
        if prices_produced:
            prices_new = np.where(prices.value() > 0, prices.value(), np.nan)
            if state.sigma_inv is not None and state.prev_prices is not None:
                r = prices_new / state.prev_prices - 1.0
                r = np.where(np.isfinite(r), r, 0.0)
                state.sum_quad += float(r @ state.sigma_inv @ r)
                state.count += 1
            state.prev_prices = prices_new

        # Gate: new prediction?
        if not predictions_produced:
            return False

        # First prediction stores cache without emitting.
        if not state.initialized:
            state.initialized = True
            LogLikelihood._set_prediction(state, predictions.value())
            return False

        # Emit period-averaged negative log-likelihood.
        ll = state.log_det + state.sum_quad / max(state.count, 1)
        output.write(np.array(ll, dtype=np.float64))

        # Update cache and reset accumulators.
        LogLikelihood._set_prediction(state, predictions.value())
        state.sum_quad = 0.0
        state.count = 0

        return True


def _log_pdet_and_pinv(sigma: np.ndarray) -> tuple[float, np.ndarray]:
    """Log-pseudo-determinant and Moore-Penrose pseudo-inverse via one SVD.

    For a real symmetric matrix the singular values equal the absolute
    eigenvalues (and equal the eigenvalues themselves when the matrix
    is PSD), so a single SVD of a symmetrized ``sigma`` yields both
    outputs directly:

    - the log-pseudo-determinant is ``sum(log(s[s > cutoff]))``;
    - the Moore-Penrose pseudo-inverse is
      ``V @ diag(1/s[retained]) @ U^T`` on the retained subspace and
      zero elsewhere — equivalent to
      [`scipy.linalg.pinv`][scipy.linalg.pinv] on symmetric PSD inputs.

    The cutoff ``max(M, N) * eps * s_max`` matches scipy's default
    ``rcond``; numerical-artifact negative eigenvalues (e.g. from
    pairwise-deletion sample covariance or the Hausdorff filter) lie
    far below it and are discarded, correctly restricting the Gaussian
    log-likelihood to the PSD subspace.
    """
    sym = 0.5 * (sigma + sigma.T)
    U, s, Vt = np.linalg.svd(sym)
    s_max = float(s[0]) if s.size else 0.0
    cutoff = max(sym.shape) * np.finfo(sym.dtype).eps * s_max
    mask = s > cutoff
    log_pdet = float(np.log(s[mask]).sum()) if mask.any() else 0.0
    inv_s = np.where(mask, 1.0 / np.where(mask, s, 1.0), 0.0)
    pinv = (Vt.T * inv_s) @ U.T
    return log_pdet, pinv
