"""Global minimum-variance portfolio optimization (pyhost port).

Warm-started via a factor-model risk (see `tradingflow.portfolios._factor`): the
covariance is approximated as `Sigma ~= B B^T + diag(d^2)` (top-`factor_rank` SVD
low-rank + diagonal), so the objective `x^T Sigma x = sum_squares(F @ x) +
sum_squares(d (.) x)` (with `F = B^T`) is DPP.  The problem is built once, sized to
`max_universe_size`, and re-solved each rebalance with the parameters set and the
previous positions seeded.

NOTE (blocker): depends on cvxpy, NOT importable on `.venv-ft`; the cvxpy import is
deferred to `init`/solve time.
"""

from __future__ import annotations

import numpy as np

from tradingflow.portfolios._base import VariancePortfolio
from tradingflow.portfolios._factor import assign_slots, factor_params_at


class MinimumVariance(VariancePortfolio):
    r"""Global minimum-variance portfolio: minimize ``x^T Sigma x`` s.t. budget (+ long-only).

    Inputs: ``(universe, covariance)`` with shapes ``(num_stocks,)`` /
    ``(num_stocks, num_stocks)``.
    Output: position weights, shape ``(num_stocks,)``.  ``factor_rank`` (default 20)
    is the rank of the SVD low-rank covariance approximation.
    """

    def __init__(
        self,
        *,
        long_only: bool = True,
        verbose: bool = False,
        num_stocks: int | None = None,
        max_universe_size: int | None = None,
        factor_rank: int = 20,
        logarithmic: bool = True,
    ) -> None:
        factor_rank = int(factor_rank)
        super().__init__(
            init_fn=lambda m: _build(m, factor_rank, long_only),
            # Variance base passes `sub_mu=None`; GMV ignores it.
            positions_fn=lambda state, active_idx, sub_universe, x_prev, sub_mu, sigma: _solve(
                state.solver, active_idx, sigma, long_only, verbose
            ),
            num_stocks=num_stocks,
            max_universe_size=max_universe_size,
            logarithmic=logarithmic,
        )


def _build(max_universe_size: int, factor_rank: int, long_only: bool) -> dict:
    """Build the fixed-size, DPP, factor-model GMV problem once (per `init`)."""
    import cvxpy as cp

    m, r = int(max_universe_size), int(factor_rank)
    x = cp.Variable(m)
    F = cp.Parameter((r, m))               # B.T (factor loadings), padded
    d = cp.Parameter(m)                    # idiosyncratic std-dev, padded
    active = cp.Parameter(m, nonneg=True)

    constraints = [cp.multiply(1.0 - active, x) == 0, cp.sum(x) == 1]
    if long_only:
        constraints.append(x >= 0)
    risk_var = cp.sum_squares(F @ x) + cp.sum_squares(cp.multiply(d, x))  # x^T Sigma x
    prob = cp.Problem(cp.Minimize(risk_var), constraints)
    return {"prob": prob, "x": x, "F": F, "d": d, "active": active, "m": m, "r": r, "slot_of": {}}


def _solve(
    handle: dict,
    active_idx: np.ndarray,
    sigma: np.ndarray,
    long_only: bool,
    verbose: bool,
) -> np.ndarray:
    """Map the active stocks to stable slots, set the factor-model parameters, and
    re-solve.  No `x.value` seeding (cvxpy uses its cached previous solution, kept
    aligned by the stable slots)."""
    import cvxpy as cp

    m, n = handle["m"], sigma.shape[0]
    slots = assign_slots(handle["slot_of"], active_idx, m)
    F_val, d_val, active = factor_params_at(sigma, slots, m, handle["r"])

    handle["F"].value = F_val
    handle["d"].value = d_val
    handle["active"].value = active

    try:
        handle["prob"].solve(solver=cp.SCS, warm_start=True)
    except cp.SolverError as e:
        print(f"  minimum_variance: solver failed ({e}), using equal weights")
        return np.full(n, 1.0 / n)

    if handle["x"].value is None:
        print(f"  minimum_variance: no solution (status={handle['prob'].status})")
        return np.full(n, 1.0 / n)

    weights = np.asarray(handle["x"].value[slots], dtype=np.float64)  # active order
    if long_only:
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > 0:
            weights /= s

    if verbose:
        exp_vol = float(np.sqrt(max(weights @ sigma @ weights, 0.0)))
        n_nonzero = int((np.abs(weights) > 1e-6).sum())
        print(
            f"  minimum_variance: status={handle['prob'].status}, {n_nonzero}/{n} stocks, "
            f"{weights.sum():.4f} invested, vol={exp_vol:.4e}"
        )

    return weights


def build(**kwargs) -> MinimumVariance:
    return MinimumVariance(**kwargs)
