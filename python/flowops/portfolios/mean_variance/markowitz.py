"""Markowitz mean-variance portfolio optimization (flowops host port).

NOTE (blocker): this operator depends on cvxpy, which is NOT importable on the
free-threaded interpreter (`.venv-ft`).  The solve body is ported verbatim and the
cvxpy import is deferred to call time, so the module imports fine on ft (the `Mode`
enum lives in `_modes.py`) and the operator runs unchanged once cvxpy is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy as sp

from flowops.portfolios._base import MeanVariancePortfolio
from flowops.portfolios.mean_variance._modes import Mode

__all__ = ["Markowitz", "Mode", "build"]


class Markowitz(MeanVariancePortfolio):
    r"""Markowitz mean-variance optimization with a pluggable mode.

    Inputs: ``(universe, predicted_returns, covariance)`` with shapes
    ``(num_stocks,)`` / ``(num_stocks,)`` / ``(num_stocks, num_stocks)``.
    Output: position weights, ``(num_stocks,)``.

    ``mode`` may be a `Mode` member or its integer value (the Rust host passes
    config scalars), and ``bound`` parameterises the chosen mode.
    """

    def __init__(
        self,
        *,
        mode: Mode | int,
        bound: float,
        long_only: bool = True,
        full_position: bool = True,
        verbose: bool = False,
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        mode = Mode(mode)
        super().__init__(
            positions_fn=lambda state, mu, sigma, x_bm: _solve(
                mu, sigma, mode, bound, long_only, full_position, verbose
            ),
            num_stocks=num_stocks,
            logarithmic=logarithmic,
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
    import cvxpy as cp

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


def build(**kwargs) -> Markowitz:
    return Markowitz(**kwargs)
