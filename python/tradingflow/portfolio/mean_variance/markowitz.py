"""Markowitz mean-variance optimization, solved by CVXPY on a factor-model risk."""

import numpy as np

from .._base import MeanVariancePortfolio
from .._factor import assign_slots, factor_params_at
from ._modes import Mode


def build_solver(
    max_universe_size: int,
    factor_rank: int,
    mode: Mode,
    bound: float,
    long_only: bool,
    full_position: bool,
) -> dict:
    """Builds the fixed-size DPP problem once, at `init`."""
    import cvxpy as cp

    m, r = int(max_universe_size), int(factor_rank)
    x = cp.Variable(m)
    factors = cp.Parameter((r, m))  # B.T, padded
    idio = cp.Parameter(m)  # idiosyncratic std-dev, padded
    mu = cp.Parameter(m)
    active = cp.Parameter(m, nonneg=True)

    constraints = [cp.multiply(1.0 - active, x) == 0]  # pin inactive weights to 0
    if long_only:
        constraints.append(x >= 0)
    constraints.append(cp.sum(x) == 1 if full_position else cp.sum(x) <= 1)

    variance = cp.sum_squares(factors @ x) + cp.sum_squares(cp.multiply(idio, x))
    expected = mu @ x
    match mode:
        case Mode.MIN_VARIANCE_GIVEN_RETURN:
            objective = cp.Minimize(variance)
            constraints.append(expected >= bound)
        case Mode.MAX_RETURN_GIVEN_STD_DEV:
            objective = cp.Maximize(expected)
            constraints.append(variance <= bound * bound)
        case Mode.MIN_MEAN_VARIANCE:
            objective = cp.Maximize(expected - bound * variance)
        case Mode.MIN_MEAN_STD_DEV:
            deviation = cp.norm(cp.hstack([factors @ x, cp.multiply(idio, x)]))
            objective = cp.Maximize(expected - bound * deviation)

    return {
        "problem": cp.Problem(objective, constraints),
        "x": x,
        "factors": factors,
        "idio": idio,
        "mu": mu,
        "active": active,
        "size": m,
        "rank": r,
        "slots": {},
    }


def solve(
    handle: dict,
    active_indices: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    long_only: bool,
) -> np.ndarray:
    """Maps the active stocks to stable slots, sets the parameters, re-solves.

    The variable is never seeded directly: CVXPY re-solves from its own cached
    solution, and the stable slot assignment is what keeps that cache aligned
    for the names that persist across the rebalance.
    """
    import cvxpy as cp

    n = len(mu)
    slots = assign_slots(handle["slots"], active_indices, handle["size"])
    factors, idio, active = factor_params_at(sigma, slots, handle["size"], handle["rank"])
    padded = np.zeros(handle["size"])
    padded[slots] = mu

    handle["factors"].value = factors
    handle["idio"].value = idio
    handle["mu"].value = padded
    handle["active"].value = active

    try:
        handle["problem"].solve(solver=cp.SCS, warm_start=True)
    except cp.SolverError:
        return np.full(n, 1.0 / n)
    if handle["x"].value is None:
        return np.full(n, 1.0 / n)

    weights = np.asarray(handle["x"].value[slots], dtype=np.float64)
    return np.maximum(weights, 0.0) if long_only else weights


def build(
    *,
    mode: Mode | int = Mode.MIN_MEAN_VARIANCE,
    bound: float,
    long_only: bool = True,
    full_position: bool = True,
    factor_rank: int = 20,
    **kwargs,
) -> MeanVariancePortfolio:
    """Constructs a Markowitz mean-variance portfolio."""
    mode = Mode(mode)
    bound = float(bound)
    return MeanVariancePortfolio(
        init_solver=lambda m: build_solver(
            m, factor_rank, mode, bound, long_only, full_position
        ),
        solve=lambda state, active, universe, previous, mu, sigma: solve(
            state.solver, active, mu, sigma, long_only
        ),
        **kwargs,
    )
