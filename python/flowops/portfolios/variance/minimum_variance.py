"""Global minimum-variance portfolio optimization (flowops host port).

NOTE (blocker): this operator depends on cvxpy, which is NOT importable on the
free-threaded interpreter (`.venv-ft`).  The body is ported verbatim so it will
run unchanged once cvxpy is available; the cvxpy import is deferred to call time
so the module can still be imported (and the rank/mean leaves tested) on ft.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy as sp

from flowops.portfolios._base import VariancePortfolio


class MinimumVariance(VariancePortfolio):
    r"""Global minimum-variance portfolio: minimize ``x^T Sigma x`` s.t. budget (+ long-only).

    Inputs: ``(universe, covariance)`` with shapes ``(num_stocks,)`` /
    ``(num_stocks, num_stocks)``.
    Output: position weights, ``(num_stocks,)``.
    """

    def __init__(
        self,
        *,
        long_only: bool = True,
        verbose: bool = False,
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        super().__init__(
            positions_fn=lambda state, sigma: _solve(sigma, long_only, verbose),
            num_stocks=num_stocks,
            logarithmic=logarithmic,
        )


def _solve(sigma: np.ndarray, long_only: bool, verbose: bool) -> np.ndarray:
    """Solve the GMV optimization problem."""
    import cvxpy as cp

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


def build(**kwargs) -> MinimumVariance:
    return MinimumVariance(**kwargs)
