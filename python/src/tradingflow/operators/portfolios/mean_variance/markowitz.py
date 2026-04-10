"""Markowitz mean-variance portfolio optimization."""

from typing import Any

import numpy as np
import scipy as sp
import cvxpy as cp

from ..mean_variance_portfolio import MeanVariancePortfolio


class Markowitz(MeanVariancePortfolio):
    """Markowitz mean-variance optimization (formulation 2.4).

    Solves the following optimization problem:

        maximize  mu' x  -  delta * sqrt(x' Sigma x)
        subject to  1' x <= 1
                    x >= 0   (if long_only)

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
    predicted_returns
        Handle to predicted returns, shape `(num_stocks,)`.
    covariance
        Handle to covariance matrix, shape `(num_stocks, num_stocks)`.
    risk_aversion
        Risk-aversion coefficient `delta`.
    long_only
        If `True` (default), enforce `x >= 0`.
    verbose
        If `True`, print optimization diagnostics to stdout.
    **kwargs
        Forwarded to [`MeanVariancePortfolio`][tradingflow.operators.portfolios.MeanVariancePortfolio].
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
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            predicted_returns,
            covariance,
            positions_fn=lambda state, mu, sigma: _solve(mu, sigma, risk_aversion, long_only, verbose),
            **kwargs,
        )


def _solve(mu: np.ndarray, sigma: np.ndarray, delta: float, long_only: bool, verbose: bool) -> np.ndarray:
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
    objective = cp.Maximize(mu @ x - delta * cp.norm(L.T @ x))
    constraints: list[Any] = [cp.sum(x) <= 1]
    if long_only:
        constraints.append(x >= 0)

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
