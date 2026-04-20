"""Global minimum-variance portfolio optimization."""

from typing import Any

import numpy as np
import scipy as sp
import cvxpy as cp

from ..variance_portfolio import VariancePortfolio


class MinimumVariance(VariancePortfolio):
    """Global minimum-variance portfolio optimization.

    Solves the following optimization problem:

        minimize    x' Sigma x
        subject to  1' x = 1
                    x >= 0   (if long_only)

    When `long_only=False` this reproduces — modulo solver tolerance —
    the analytical GMV solution `x = Σ⁺ 1 / (1ᵀ Σ⁺ 1)` used by the
    evaluation metric
    [`MinimumVariance`][tradingflow.operators.metrics.variance.MinimumVariance].
    With `long_only=True` the no-short-selling constraint cannot in
    general be satisfied by the closed-form pseudo-inverse solution, and
    the CVXPY quadratic program is used to enforce it.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
    covariance
        Handle to covariance matrix, shape `(num_stocks, num_stocks)`.
    long_only
        If `True` (default), enforce `x >= 0`.
    verbose
        If `True`, print optimization diagnostics to stdout.
    **kwargs
        Forwarded to [`VariancePortfolio`][tradingflow.operators.portfolios.VariancePortfolio].
    """

    def __init__(
        self,
        universe,
        covariance,
        *,
        long_only: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            covariance,
            positions_fn=lambda state, sigma: _solve(sigma, long_only, verbose),
            **kwargs,
        )


def _solve(sigma: np.ndarray, long_only: bool, verbose: bool) -> np.ndarray:
    """Solve the GMV optimization problem."""
    N = sigma.shape[0]

    if verbose:
        print(f"  minimum_variance: sigma has shape {sigma.shape} and range [{sigma.min():.4e}, {sigma.max():.4e}]")

    # LDL decomposition: sigma = L @ D @ L.T, where D diagonal and L[perm, :] lower-triangular.
    L, D, perm = sp.linalg.ldl(sigma)
    L = L * np.sqrt(np.maximum(np.diag(D), 0.0)).reshape(1, N)

    if verbose:
        error = np.max(np.abs(sigma - L @ L.T))
        print(f"  minimum_variance: L has shape {L.shape} and range [{L.min():.4f}, {L.max():.4f}]")
        print(f"  minimum_variance: LDL max error {error:.4} (non-zero may indicate non-positive-semidefinite sigma)")

    # Construct the problem.  Minimising ||L' x||_2 is equivalent to
    # minimising x' Sigma x (monotonic transform, same argmin).
    x = cp.Variable(N)
    objective = cp.Minimize(cp.norm(L.T @ x))
    constraints: list[Any] = [cp.sum(x) == 1]
    if long_only:
        constraints.append(x >= 0)

    # Solve the problem.
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS)
    except cp.SolverError as e:
        print(f"  minimum_variance: solver failed ({e}), using equal weights")
        return np.full(N, 1.0 / N)

    if x.value is None:
        print(f"  minimum_variance: no solution (status={prob.status})")
        return np.full(N, 1.0 / N)

    weights = np.array(x.value, dtype=np.float64)

    if long_only:
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > 0:
            weights /= s

    if verbose:
        n_nonzero = (np.abs(weights) > 1e-6).sum()
        s = weights.sum()
        exp_vol = float(np.sqrt(max(weights @ sigma @ weights, 0.0)))
        print(f"  minimum_variance: problem status: {prob.status}")
        print(f"  minimum_variance: {n_nonzero}/{N} stocks, {s:.4f} invested, vol={exp_vol:.4e}")

    return weights
