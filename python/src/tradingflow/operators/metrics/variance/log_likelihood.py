"""Gaussian log-likelihood evaluator for covariance-matrix predictions."""

from dataclasses import dataclass

import numpy as np

from .... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class LogLikelihoodState:
    num_stocks: int
    initialized: bool = False
    sigma_inv: np.ndarray | None = None
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
    r"""Gaussian negative-log-likelihood evaluator for covariance predictions.

    Evaluates covariance prediction quality.  On each prediction
    emission, caches \(\log |\Sigma|\) and the Moore-Penrose
    pseudo-inverse \(\Sigma^+\) (via a single SVD on the symmetrized
    \(\Sigma\), equivalent to
    [`scipy.linalg.pinv`][scipy.linalg.pinv] on symmetric PSD inputs
    and handling rank-deficient \(\Sigma\) gracefully) and begins
    accumulating the daily Mahalanobis quadratic \(r^T \Sigma^+ r\)
    from the target ticks that follow.  When the next prediction arrives,
    emits the period-averaged negative log-likelihood

    \[
    \log |\Sigma| + \frac{1}{T} \sum_t r_t^T \Sigma^+ r_t
    \]

    — the multivariate-normal log-density with the \(N \log(2\pi)\)
    constant and the \(1/2\) prefactor dropped (lower is better) — and
    updates the cache.  Matches the `ll_metric` in
    <https://osquant.com/papers/a-quants-guide-to-covariance-matrix-estimation/>.

    Output is a scalar (the period-averaged negative log-likelihood
    over one evaluation period).  `Record(output)` produces a
    directly plottable time series.

    Notes
    -----
    **Alignment guarantee.** After the initial warmup (first prediction
    caches \(\Sigma^+\) / \(\log |\Sigma|\) without emitting), the
    operator emits exactly once per prediction emission — **the same
    cadence as**
    [`MinimumVariance`][tradingflow.operators.metrics.variance.minimum_variance.MinimumVariance],
    so corresponding records line up element-by-element.  Output is 0
    if no target tick was accumulated during the period (e.g. no
    stocks had finite covariance diagonal).

    **Memory.** Both `predictions` and `target` are `Array` inputs —
    the operator only reads the latest cross-section of each.
    No `Record` is required upstream.

    Parameters
    ----------
    predictions
        Live predicted covariance matrix from a variance predictor,
        shape `(N, N)`.  Stocks excluded by the variance predictor
        have NaN on the diagonal.
    target
        Live cross-sectional realized target values, shape `(N,)`,
        produced at every tick (e.g. a `PctChange` node).  Non-finite
        entries contribute zero to the Mahalanobis quadratic.
    """

    def __init__(self, predictions: Handle, target: Handle) -> None:
        num_stocks = predictions.shape[0]

        assert predictions.shape == (num_stocks, num_stocks)
        assert target.shape == (num_stocks,)

        self._num_stocks = num_stocks

        super().__init__(
            inputs=(predictions, target),
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
        return LogLikelihoodState(num_stocks=self._num_stocks)

    @staticmethod
    def _set_prediction(state: LogLikelihoodState, sigma: np.ndarray) -> None:
        r"""Compute and store \(\Sigma^+\) and \(\log |\Sigma|\) from a covariance matrix.

        Stocks with non-finite covariance diagonal are excluded; their
        rows and columns of \(\Sigma^+\) remain zero so their returns
        contribute zero to the quadratic form.  If no stocks are
        eligible, \(\Sigma^+\) is all zeros and \(\log |\Sigma|\) is zero.
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
        predictions, target = inputs
        predictions_produced, target_produced = produced

        # Accumulate one-period Mahalanobis quadratic on each target tick.
        if target_produced and state.initialized:
            r = target.value()
            r = np.where(np.isfinite(r), r, 0.0)
            state.sum_quad += float(r @ state.sigma_inv @ r)
            state.count += 1

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
    r"""Log-pseudo-determinant and Moore-Penrose pseudo-inverse via one SVD.

    For a real symmetric matrix the singular values equal the absolute
    eigenvalues (and equal the eigenvalues themselves when the matrix
    is PSD), so a single SVD of a symmetrized `sigma` yields both
    outputs directly:

    - the log-pseudo-determinant is \(\sum_{s_i > \text{cutoff}} \log s_i\);
    - the Moore-Penrose pseudo-inverse is
      \(V \, \operatorname{diag}(1/s_{\text{retained}}) \, U^T\) on the
      retained subspace and zero elsewhere — equivalent to
      [`scipy.linalg.pinv`][scipy.linalg.pinv] on symmetric PSD inputs.

    The cutoff \(\max(M, N) \cdot \varepsilon \cdot s_{\max}\) matches
    scipy's default `rcond`; numerical-artifact negative eigenvalues
    (e.g. from pairwise-deletion sample covariance or the Hausdorff
    filter) lie far below it and are discarded, correctly restricting
    the Gaussian log-likelihood to the PSD subspace.
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
