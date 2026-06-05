"""Benchmark-relative (tracking-error-budgeted) mean-variance optimization.

flowops host port.  NOTE (blocker): depends on cvxpy (SOC constraint), which is
NOT importable on the free-threaded interpreter (`.venv-ft`).  Body ported
verbatim with a deferred cvxpy import so the module imports on ft and runs
unchanged once cvxpy is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy as sp

from flowops.portfolios._base import MeanVariancePortfolio

__all__ = ["BenchmarkRelative", "build"]


class BenchmarkRelative(MeanVariancePortfolio):
    r"""Benchmark-relative Markowitz: maximize return subject to a tracking-error budget.

    Inputs: ``(universe, predicted_returns, covariance)`` with shapes
    ``(num_stocks,)`` / ``(num_stocks,)`` / ``(num_stocks, num_stocks)``.  The
    ``universe`` doubles as the benchmark ``x_bm`` (restricted to the active subset
    and renormalised to sum to 1 by the base).
    Output: position weights, ``(num_stocks,)``.

    ``bound`` is the tracking-error budget ``gamma_TE`` (daily by default).
    """

    def __init__(
        self,
        *,
        bound: float,
        long_only: bool = True,
        full_position: bool = True,
        verbose: bool = False,
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        super().__init__(
            positions_fn=lambda state, mu, sigma, x_bm: _solve(
                mu, sigma, x_bm, bound, long_only, full_position, verbose
            ),
            num_stocks=num_stocks,
            logarithmic=logarithmic,
        )


def _solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    x_bm: np.ndarray,
    gamma_te: float,
    long_only: bool,
    full_position: bool,
    verbose: bool,
) -> np.ndarray:
    """Solve max mu^T x s.t. TE(x, x_bm) <= gamma_te, x in feasible set."""
    import cvxpy as cp

    N = len(mu)

    if not (gamma_te > 0):
        if verbose:
            print(f"  benchmark_relative: non-positive TE budget {gamma_te}, returning benchmark")
        return x_bm.copy()

    if verbose:
        print(f"  benchmark_relative: mu shape {mu.shape}, range [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  benchmark_relative: sigma shape {sigma.shape}, range [{sigma.min():.4f}, {sigma.max():.4f}]")
        print(f"  benchmark_relative: x_bm sum {x_bm.sum():.4f}, range [{x_bm.min():.4f}, {x_bm.max():.4f}]")
        print(f"  benchmark_relative: TE budget gamma={gamma_te:.6f}")

    # LDL decomposition: sigma = L D L^T, then absorb sqrt(D) into L so
    # sigma = L L^T.  Same convention as `Markowitz._solve`.
    L, D, perm = sp.linalg.ldl(sigma)
    L = L * np.sqrt(np.maximum(np.diag(D), 0.0)).reshape(1, N)

    if verbose:
        error = float(np.max(np.abs(sigma - L @ L.T)))
        print(f"  benchmark_relative: LDL max recon error {error:.4g}")

    x = cp.Variable(N)
    constraints: list[Any] = []

    if long_only:
        constraints.append(x >= 0)

    if full_position:
        constraints.append(cp.sum(x) == 1)
    else:
        constraints.append(cp.sum(x) <= 1)

    # Tracking-error constraint: ||L^T (x - x_bm)||_2 <= gamma_te.
    constraints.append(cp.norm(L.T @ (x - x_bm)) <= gamma_te)

    prob = cp.Problem(cp.Maximize(mu @ x), constraints)
    try:
        prob.solve(solver=cp.SCS)
    except cp.SolverError as e:
        print(f"  benchmark_relative: solver failed ({e}), falling back to benchmark")
        return x_bm.copy()

    if x.value is None:
        print(f"  benchmark_relative: no solution (status={prob.status}), falling back to benchmark")
        return x_bm.copy()

    weights = np.array(x.value, dtype=np.float64)

    if long_only:
        weights = np.maximum(weights, 0.0)

    if verbose:
        active = weights - x_bm
        te_realised = float(np.sqrt(max(active @ sigma @ active, 0.0)))
        exp_ret = float(mu @ weights)
        n_nonzero = int((np.abs(weights) > 1e-6).sum())
        print(
            f"  benchmark_relative: status={prob.status}, {n_nonzero}/{N} stocks, "
            f"sum={weights.sum():.4f}, E[r]={exp_ret:.4f}, TE={te_realised:.6f}/{gamma_te:.6f}"
        )

    return weights


def build(**kwargs) -> BenchmarkRelative:
    return BenchmarkRelative(**kwargs)
