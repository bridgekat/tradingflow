"""Global Minimum Variance (GMV) portfolio realized variance evaluator."""

from dataclasses import dataclass

import numpy as np
import scipy as sp

from .... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class MinimumVarianceState:
    num_stocks: int
    initialized: bool = False
    weights: np.ndarray | None = None
    sum_r: float = 0.0
    sum_r_sq: float = 0.0
    count: int = 0


class MinimumVariance(
    Operator[
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        MinimumVarianceState,
    ]
):
    r"""GMV portfolio realized variance evaluator.

    Evaluates covariance prediction quality.  On each prediction
    emission, computes GMV weights
    \(w = \Sigma^+ \mathbf{1} / (\mathbf{1}^T \Sigma^+ \mathbf{1})\)
    (using the SVD-based pseudo-inverse for numerical stability with
    rank-deficient \(\Sigma\)) and begins accumulating daily portfolio
    linear returns \(r_p = w^T r_\text{lin}\) from the target ticks
    that follow.  When the next prediction arrives, emits the realized
    variance of the accumulated portfolio returns and updates the
    weights.

    Output is a scalar (the realized variance over one evaluation
    period).  `Record(output)` produces a directly plottable time
    series.

    ## Expected prediction semantics

    Mirrors the
    [`VariancePortfolio`][tradingflow.operators.portfolios.variance_portfolio.VariancePortfolio]
    contract: `predictions` is a **log-return** covariance matrix
    (zero-mean lognormal specialisation), converted internally to
    linear-return units before solving GMV:

        mu_lin[i]       = exp(½ Sigma_log[i, i]) - 1
        Sigma_lin[i, j] = (1 + mu_lin[i]) (1 + mu_lin[j])
                          · (exp(Sigma_log[i, j]) - 1)

    `target` is also expected to be a **log-return** vector
    (`Diff(Log(prices))` upstream).  The metric maps it to linear
    returns elementwise (`exp(r_log) - 1`) before accumulating the
    portfolio return, so the reported realized variance matches the
    linear-return units that the GMV objective actually minimises.

    Notes
    -----
    **Alignment guarantee.** After the initial warmup (first prediction
    sets weights without emitting), the operator emits exactly once per
    prediction emission.  Output is 0 if no target tick was accumulated
    during the period (e.g. no stocks had finite covariance diagonal).

    **Memory.** Both `predictions` and `target` are `Array` inputs —
    the operator only reads the latest cross-section of each.
    No `Record` is required upstream.

    Parameters
    ----------
    predictions
        Live predicted log-return covariance matrix from a variance
        predictor, shape `(N, N)`.  Stocks excluded by the variance
        predictor have NaN on the diagonal.
    target
        Live cross-sectional realized log-returns, shape `(N,)`,
        produced at every tick (e.g. a `Diff(Log(prices))` node).
        Non-finite entries contribute zero to the portfolio return.
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
    ) -> MinimumVarianceState:
        return MinimumVarianceState(num_stocks=self._num_stocks)

    @staticmethod
    def _set_weights(state: MinimumVarianceState, sigma_log: np.ndarray) -> None:
        """Compute and store GMV weights from a log-return covariance
        matrix (converted to linear returns before solving GMV — zero-mean
        specialisation of the lognormal moment map).

        Stocks with non-finite covariance diagonal are excluded; their
        weights remain zero.  If no stocks are eligible, all weights
        are zero.
        """
        n = state.num_stocks
        mask = np.isfinite(np.diag(sigma_log))
        weights = np.zeros(n, dtype=np.float64)
        if mask.any():
            sub_log = sigma_log[np.ix_(mask, mask)]
            sub_mu_linear = np.expm1(0.5 * np.diag(sub_log))
            factor = 1.0 + sub_mu_linear
            sub_sigma_linear = np.outer(factor, factor) * np.expm1(sub_log)
            weights[mask] = _gmv_weights(sub_sigma_linear)
        state.weights = weights

    @staticmethod
    def compute(
        state: MinimumVarianceState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        predictions, target = inputs
        predictions_produced, target_produced = produced

        # Accumulate one-period portfolio linear return on each target
        # tick.  Target arrives as log returns; convert elementwise so
        # that the reported realized variance matches the linear-return
        # objective the GMV weights actually minimise.  Non-finite
        # entries contribute zero (exp(0) - 1 == 0).
        if target_produced and state.initialized:
            r_log = target.value()
            r_log = np.where(np.isfinite(r_log), r_log, 0.0)
            r_linear = np.expm1(r_log)
            r_p = float(state.weights @ r_linear)
            state.sum_r += r_p
            state.sum_r_sq += r_p * r_p
            state.count += 1

        # Gate: new prediction?
        if not predictions_produced:
            return False

        # First prediction stores weights without emitting.
        if not state.initialized:
            state.initialized = True
            MinimumVariance._set_weights(state, predictions.value())
            return False

        # Emit realized variance over the evaluation period.
        mean = state.sum_r / max(state.count, 1)
        variance = state.sum_r_sq / max(state.count, 1) - mean * mean
        output.write(np.array(variance, dtype=np.float64))

        # Update weights and reset accumulators.
        MinimumVariance._set_weights(state, predictions.value())
        state.sum_r = 0.0
        state.sum_r_sq = 0.0
        state.count = 0

        return True


def _gmv_weights(sigma: np.ndarray) -> np.ndarray:
    r"""Compute GMV portfolio weights from a covariance matrix.

    Closed-form solution to the equality-constrained quadratic program

    \[
    \begin{aligned}
    \text{minimize} \quad & w^T \Sigma w \\
    \text{subject to} \quad & \mathbf{1}^T w = 1
    \end{aligned}
    \]

    given by \(w = \Sigma^+ \mathbf{1} / (\mathbf{1}^T \Sigma^+ \mathbf{1})\).
    No non-negativity constraint is imposed (weights may be negative,
    i.e. short positions are allowed).

    Uses the SVD-based Moore-Penrose pseudo-inverse \(\Sigma^+\) instead
    of a direct solve, which gracefully handles rank-deficient covariance
    matrices (e.g. sample covariance with N > T) by returning the
    minimum-norm solution.  If the denominator
    \(\mathbf{1}^T \Sigma^+ \mathbf{1}\) is zero, the returned weights
    contain non-finite entries (NaN or inf).
    """
    n = sigma.shape[0]
    ones = np.ones(n, dtype=np.float64)
    sigma_pinv = sp.linalg.pinv(sigma)
    w = sigma_pinv @ ones
    return w / (ones @ w)
