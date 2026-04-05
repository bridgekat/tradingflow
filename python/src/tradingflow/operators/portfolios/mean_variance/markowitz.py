"""Markowitz mean-variance portfolio optimization."""

from typing import Any

import numpy as np
import cvxpy as cp

from ..mean_variance_portfolio import MeanVariancePortfolio


class Markowitz(MeanVariancePortfolio):
    """Markowitz mean-variance optimization (formulation 2.4).

    Solves::

        maximize  mu' x  -  delta * sqrt(x' Sigma x)
        subject to  1' x = 1
                    x >= 0   (if long_only)

    using CVXPY.

    Parameters
    ----------
    universe
        Handle to universe weights, shape ``(num_stocks,)``.
    predicted_returns
        Handle to predicted returns, shape ``(num_stocks,)``.
    covariance
        Handle to covariance matrix, shape ``(num_stocks, num_stocks)``.
    risk_aversion
        Risk-aversion coefficient ``delta``.
    long_only
        If ``True`` (default), enforce ``x >= 0``.
    verbose
        If ``True``, print optimization diagnostics to stdout.
    """

    def __init__(
        self,
        universe,
        predicted_returns,
        covariance,
        *,
        risk_aversion: float = 1.0,
        long_only: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            universe,
            predicted_returns,
            covariance,
            positions_fn=lambda state, mu, sigma: _positions_fn(mu, sigma, risk_aversion, long_only, verbose),
        )


def _positions_fn(mu: np.ndarray, sigma: np.ndarray, delta: float, long_only: bool, verbose: bool) -> np.ndarray:
    """Solve the Markowitz optimization problem via CVXPY."""
    N = len(mu)

    try:
        L = np.linalg.cholesky(sigma + np.eye(N) * 1e-10)
    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"  markowitz: Cholesky failed ({e}), using equal weights")
        return np.full(N, 1.0 / N)

    x = cp.Variable(N)
    objective = cp.Maximize(mu @ x - delta * cp.norm(L.T @ x))
    constraints: list[Any] = [cp.sum(x) == 1]
    if long_only:
        constraints.append(x >= 0)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS)
    except cp.SolverError as e:
        if verbose:
            print(f"  markowitz: solver failed ({e}), using equal weights")
        return np.full(N, 1.0 / N)

    if x.value is None:
        if verbose:
            print(f"  markowitz: no solution (status={prob.status})")
        return np.full(N, 1.0 / N)

    weights = np.array(x.value, dtype=np.float64).ravel()
    if long_only:
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > 1e-6:
            weights /= s

    if verbose:
        n_nonzero = (np.abs(weights) > 1e-6).sum()
        exp_ret = float(mu @ weights)
        exp_vol = float(np.sqrt(weights @ sigma @ weights))
        print(f"  markowitz: {n_nonzero}/{N} stocks, E[r]={exp_ret:.4f}, vol={exp_vol:.4f}")

    return weights
