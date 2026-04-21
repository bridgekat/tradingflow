"""Markowitz mean-variance portfolio optimization."""

from typing import Any
from enum import IntEnum

import numpy as np
import scipy as sp
import cvxpy as cp

from ..mean_variance_portfolio import MeanVariancePortfolio


class Mode(IntEnum):
    r"""Markowitz optimization mode.

    All modes share the long-only (optional) and budget constraints
    \(\mathbf{1}^T x = 1\), \(x \geq 0\).  The `bound` parameter's meaning is
    mode-dependent.

    - [`MIN_VARIANCE_GIVEN_RETURN`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode]
      — minimize \(x^T \Sigma x\) subject to \(\mu^T x \geq \text{bound}\).  `bound` is
      the minimum admissible expected return \(\mu_{\min}\).
    - [`MAX_RETURN_GIVEN_STD_DEV`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode]
      — maximize \(\mu^T x\) subject to \(\sqrt{x^T \Sigma x} \leq \text{bound}\).  `bound`
      is the maximum admissible portfolio standard deviation \(\sigma_{\max}\).
    - [`MIN_MEAN_VARIANCE`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode]
      — maximize \(\mu^T x - \text{bound} \cdot x^T \Sigma x\).  `bound` is the
      variance-penalty coefficient \(\delta\) (quadratic risk aversion).
    - [`MIN_MEAN_STD_DEV`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode]
      — maximize \(\mu^T x - \text{bound} \cdot \sqrt{x^T \Sigma x}\).  `bound` is the
      std-dev-penalty coefficient \(\delta\) (linear risk aversion).
    """

    MIN_VARIANCE_GIVEN_RETURN = 1
    MAX_RETURN_GIVEN_STD_DEV = 2
    MIN_MEAN_VARIANCE = 3
    MIN_MEAN_STD_DEV = 4


class Markowitz(MeanVariancePortfolio):
    r"""Markowitz mean-variance optimization with a pluggable mode.

    Solves one of four equivalent reformulations of the mean-variance
    trade-off, selected by the `mode` parameter (see
    [`Mode`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode]).  The
    budget constraint (\(\mathbf{1}^T x = 1\) when `full_position`, else \(\mathbf{1}^T x \leq 1\))
    and the long-only constraint (\(x \geq 0\) when `long_only`) apply to
    every mode; the scalar `bound` parameterizes whichever knob the
    chosen mode uses.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
    predicted_returns
        Handle to predicted returns, shape `(num_stocks,)`.
    covariance
        Handle to covariance matrix, shape `(num_stocks, num_stocks)`.
    mode
        Optimization mode, a member of
        [`Mode`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode].
    bound
        Scalar parameter whose meaning depends on `mode` — minimum
        return \(\mu_{\min}\), maximum standard deviation \(\sigma_{\max}\), or
        risk-aversion coefficient \(\delta\).  See `Mode` for details.
        If the resulting problem is infeasible (e.g. \(\mu_{\min}\) above
        every attainable return, or \(\sigma_{\max}\) below the GMV
        volatility), the operator falls back to equal weights.
    long_only
        If `True` (default), enforce \(x \geq 0\).
    full_position
        If `True` (default), require full investment \(\mathbf{1}^T x = 1\).  If
        `False`, allow underinvestment \(\mathbf{1}^T x \leq 1\) (holding cash is
        permitted).
    verbose
        If `True`, print optimization diagnostics to stdout.
    **kwargs
        Forwarded to [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio].
    """

    def __init__(
        self,
        universe,
        predicted_returns,
        covariance,
        *,
        mode: Mode,
        bound: float,
        long_only: bool = True,
        full_position: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            predicted_returns,
            covariance,
            positions_fn=lambda state, mu, sigma: _solve(mu, sigma, mode, bound, long_only, full_position, verbose),
            **kwargs,
        )


def _solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    mode: Mode,
    bound: float,
    long_only: bool,
    full_position: bool,
    verbose: bool,
) -> np.ndarray:
    """Solve the Markowitz mean-variance optimization problem."""
    N = len(mu)

    if verbose:
        print(f"  markowitz: mu has shape {mu.shape} and range [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  markowitz: sigma has shape {sigma.shape} and range [{sigma.min():.4f}, {sigma.max():.4f}]")

    # LDL decomposition: sigma = L @ D @ L.T, where D diagonal and L[perm, :] lower-triangular.
    L, D, perm = sp.linalg.ldl(sigma)
    L = L * np.sqrt(np.maximum(np.diag(D), 0.0)).reshape(1, N)

    if verbose:
        error = np.max(np.abs(sigma - L @ L.T))
        print(f"  markowitz: L has shape {L.shape} and range [{L.min():.4f}, {L.max():.4f}]")
        print(f"  markowitz: LDL max error {error:.4} (non-zero may indicate non-positive-semidefinite sigma)")

    # Construct the problem.
    x = cp.Variable(N)
    constraints: list[Any] = []

    if long_only:
        constraints.append(x >= 0)

    if full_position:
        constraints.append(cp.sum(x) == 1)
    else:
        constraints.append(cp.sum(x) <= 1)

    match mode:
        case Mode.MIN_VARIANCE_GIVEN_RETURN:
            objective = cp.Minimize(cp.sum_squares(L.T @ x))
            constraints.append(mu @ x >= bound)
        case Mode.MAX_RETURN_GIVEN_STD_DEV:
            objective = cp.Maximize(mu @ x)
            constraints.append(cp.norm(L.T @ x) <= bound)
        case Mode.MIN_MEAN_VARIANCE:
            objective = cp.Maximize(mu @ x - bound * cp.sum_squares(L.T @ x))
        case Mode.MIN_MEAN_STD_DEV:
            objective = cp.Maximize(mu @ x - bound * cp.norm(L.T @ x))

    # Solve the problem.
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS)
    except cp.SolverError as e:
        print(f"  markowitz: solver failed ({e}), using equal weights")
        return np.full(N, 1.0 / N)

    if x.value is None:
        print(f"  markowitz: no solution (status={prob.status})")
        return np.full(N, 1.0 / N)

    weights = np.array(x.value, dtype=np.float64)

    if long_only:
        weights = np.maximum(weights, 0.0)

    if verbose:
        n_nonzero = (np.abs(weights) > 1e-6).sum()
        s = weights.sum()
        exp_ret = float(mu @ weights)
        exp_vol = float(np.sqrt(weights @ sigma @ weights))
        print(f"  markowitz: problem status: {prob.status}")
        print(f"  markowitz: {n_nonzero}/{N} stocks, {s:.4f} invested, E[r]={exp_ret:.4f}, vol={exp_vol:.4f}")

    return weights
