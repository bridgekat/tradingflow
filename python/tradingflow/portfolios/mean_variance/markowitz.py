"""Markowitz mean-variance portfolio optimization (pyhost port).

Warm-started via a **factor-model risk**.  A full covariance can't be a cvxpy
`Parameter` at universe scale (the DPP map for an M x M matrix is M x M^2 -> OOM),
so each solve approximates `Sigma ~= B B^T + diag(d^2)` (top-`factor_rank`
SVD low-rank + diagonal; see `tradingflow.portfolios._factor`) and feeds the risk as

    x^T Sigma x ~= sum_squares(F @ x) + sum_squares(d (.) x)        (F = B^T)

which IS DPP.  The problem is built once at `init` time, sized to
`max_universe_size`; each rebalance only sets the parameters and warm-starts from
the previous positions, so cvxpy reuses the canonical form (no re-canonicalisation)
and SCS reuses the prior solution.

NOTE (blocker): depends on cvxpy, NOT importable on the free-threaded interpreter
(`.venv-ft`); the cvxpy import is deferred to `init`/solve time so the module
imports on ft (the `Mode` enum lives in `_modes.py`) and runs once cvxpy is present.
"""

from __future__ import annotations

import numpy as np

from tradingflow.portfolios._base import MeanVariancePortfolio
from tradingflow.portfolios._factor import assign_slots, factor_params_at
from tradingflow.portfolios.mean_variance._modes import Mode

__all__ = ["Markowitz", "Mode", "build"]


class Markowitz(MeanVariancePortfolio):
    r"""Markowitz mean-variance optimization with a pluggable mode.

    Inputs: ``(universe, predicted_returns, covariance)`` with shapes
    ``(num_stocks,)`` / ``(num_stocks,)`` / ``(num_stocks, num_stocks)``.
    Output: position weights, ``(num_stocks,)``.

    ``mode`` may be a `Mode` member or its integer value; ``bound`` parameterises
    the mode.  ``max_universe_size`` sizes the warm-started problem (defaults to
    ``num_stocks``); ``factor_rank`` is the rank of the SVD low-rank covariance
    approximation used in the solve (default 20).
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
        max_universe_size: int | None = None,
        factor_rank: int = 20,
        logarithmic: bool = True,
    ) -> None:
        mode = Mode(mode)
        bound = float(bound)
        factor_rank = int(factor_rank)
        super().__init__(
            init_fn=lambda m: _build(m, factor_rank, mode, bound, long_only, full_position),
            positions_fn=lambda state, active_idx, sub_universe, x_prev, mu, sigma: _solve(
                state.solver, active_idx, mu, sigma, long_only, verbose
            ),
            num_stocks=num_stocks,
            max_universe_size=max_universe_size,
            logarithmic=logarithmic,
        )


def _build(
    max_universe_size: int,
    factor_rank: int,
    mode: Mode,
    bound: float,
    long_only: bool,
    full_position: bool,
) -> dict:
    """Build the fixed-size, DPP, factor-model problem once (per `init`)."""
    import cvxpy as cp

    m, r = int(max_universe_size), int(factor_rank)
    x = cp.Variable(m)
    F = cp.Parameter((r, m))               # B.T (factor loadings), padded
    d = cp.Parameter(m)                    # idiosyncratic std-dev, padded
    mu = cp.Parameter(m)                   # expected returns, padded
    active = cp.Parameter(m, nonneg=True)  # 1.0 active, 0.0 inactive

    constraints = [cp.multiply(1.0 - active, x) == 0]  # pin inactive weights to 0
    if long_only:
        constraints.append(x >= 0)
    constraints.append(cp.sum(x) == 1 if full_position else cp.sum(x) <= 1)

    risk_var = cp.sum_squares(F @ x) + cp.sum_squares(cp.multiply(d, x))  # x^T Sigma x
    ret = mu @ x
    match mode:
        case Mode.MIN_VARIANCE_GIVEN_RETURN:
            objective = cp.Minimize(risk_var)
            constraints.append(ret >= bound)
        case Mode.MAX_RETURN_GIVEN_STD_DEV:
            objective = cp.Maximize(ret)
            constraints.append(risk_var <= bound * bound)  # std <= bound  <=>  var <= bound^2
        case Mode.MIN_MEAN_VARIANCE:
            objective = cp.Maximize(ret - bound * risk_var)
        case Mode.MIN_MEAN_STD_DEV:
            risk_std = cp.norm(cp.hstack([F @ x, cp.multiply(d, x)]))  # = sqrt(x^T Sigma x)
            objective = cp.Maximize(ret - bound * risk_std)

    prob = cp.Problem(objective, constraints)
    return {"prob": prob, "x": x, "F": F, "d": d, "mu": mu, "active": active, "m": m, "r": r, "slot_of": {}}


def _solve(
    handle: dict,
    active_idx: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    long_only: bool,
    verbose: bool,
) -> np.ndarray:
    """Map the active stocks to stable slots, set the factor-model parameters, and
    re-solve.  No `x.value` seeding: cvxpy re-solves from its cached previous
    solution, which the stable slots keep aligned for names that persist."""
    import cvxpy as cp

    m, n = handle["m"], len(mu)
    slots = assign_slots(handle["slot_of"], active_idx, m)
    F_val, d_val, active = factor_params_at(sigma, slots, m, handle["r"])
    mu_val = np.zeros(m)
    mu_val[slots] = mu

    handle["F"].value = F_val
    handle["d"].value = d_val
    handle["mu"].value = mu_val
    handle["active"].value = active

    try:
        handle["prob"].solve(solver=cp.SCS, warm_start=True)
    except cp.SolverError as e:
        print(f"  markowitz: solver failed ({e}), using equal weights")
        return np.full(n, 1.0 / n)

    if handle["x"].value is None:
        print(f"  markowitz: no solution (status={handle['prob'].status})")
        return np.full(n, 1.0 / n)

    weights = np.asarray(handle["x"].value[slots], dtype=np.float64)  # active order
    if long_only:
        weights = np.maximum(weights, 0.0)

    if verbose:
        exp_ret = float(mu @ weights)
        exp_vol = float(np.sqrt(max(weights @ sigma @ weights, 0.0)))
        n_nonzero = int((np.abs(weights) > 1e-6).sum())
        print(
            f"  markowitz: status={handle['prob'].status}, {n_nonzero}/{n} stocks, "
            f"{weights.sum():.4f} invested, E[r]={exp_ret:.4f}, vol={exp_vol:.4f}"
        )

    return weights


def build(**kwargs) -> Markowitz:
    return Markowitz(**kwargs)
