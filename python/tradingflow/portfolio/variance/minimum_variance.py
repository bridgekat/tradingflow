"""Global minimum-variance portfolio, solved by CVXPY on a factor-model risk."""

import numpy as np

from .._base import VariancePortfolio
from .._factor import assign_slots, factor_params_at


def build_solver(max_universe_size: int, factor_rank: int, long_only: bool) -> dict:
    """Builds the fixed-size DPP problem once, at `init`."""
    import cvxpy as cp

    m, r = int(max_universe_size), int(factor_rank)
    x = cp.Variable(m)
    factors = cp.Parameter((r, m))  # B.T, padded
    idio = cp.Parameter(m)  # idiosyncratic std-dev, padded
    active = cp.Parameter(m, nonneg=True)

    constraints = [cp.multiply(1.0 - active, x) == 0, cp.sum(x) == 1]
    if long_only:
        constraints.append(x >= 0)

    risk = cp.sum_squares(factors @ x) + cp.sum_squares(cp.multiply(idio, x))
    return {
        "problem": cp.Problem(cp.Minimize(risk), constraints),
        "x": x,
        "factors": factors,
        "idio": idio,
        "active": active,
        "size": m,
        "rank": r,
        "slots": {},
    }


def solve(handle: dict, active_indices: np.ndarray, sigma: np.ndarray, long_only: bool) -> np.ndarray:
    """Maps the active stocks to stable slots, sets the parameters, re-solves."""
    import cvxpy as cp

    n = sigma.shape[0]
    slots = assign_slots(handle["slots"], active_indices, handle["size"])
    factors, idio, active = factor_params_at(sigma, slots, handle["size"], handle["rank"])

    handle["factors"].value = factors
    handle["idio"].value = idio
    handle["active"].value = active

    try:
        handle["problem"].solve(solver=cp.SCS, warm_start=True)
    except cp.SolverError:
        return np.full(n, 1.0 / n)
    if handle["x"].value is None:
        return np.full(n, 1.0 / n)

    weights = np.asarray(handle["x"].value[slots], dtype=np.float64)
    if long_only:
        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        if total > 0:
            weights /= total
    return weights


def build(*, long_only: bool = True, factor_rank: int = 20, **kwargs) -> VariancePortfolio:
    """Constructs a global minimum-variance portfolio."""
    return VariancePortfolio(
        init_solver=lambda m: build_solver(m, factor_rank, long_only),
        solve=lambda state, active, universe, previous, mu, sigma: solve(
            state.solver, active, sigma, long_only
        ),
        **kwargs,
    )
