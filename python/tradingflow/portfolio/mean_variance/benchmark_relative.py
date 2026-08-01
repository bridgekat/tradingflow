"""Benchmark-relative optimization: maximize return within a tracking-error budget."""

import numpy as np

from .._base import MeanVariancePortfolio
from .._factor import assign_slots, factor_params_at


def build_solver(
    max_universe_size: int,
    factor_rank: int,
    budget: float,
    long_only: bool,
    full_position: bool,
) -> dict:
    r"""Builds the fixed-size DPP problem once, at `init`.

    The tracking error against a benchmark `w`,

        TE(x) = sqrt((x - w)ᵀ Σ (x - w))
              ≈ ‖[F x - F w ; d ⊙ x - d ⊙ w]‖,

    is a second-order cone constraint. `F w` and `d ⊙ w` are precomputed each
    solve and passed as their own parameters, since a `Parameter @ Parameter`
    product would not be DPP.
    """
    import cvxpy as cp

    m, r = int(max_universe_size), int(factor_rank)
    x = cp.Variable(m)
    factors = cp.Parameter((r, m))  # B.T, padded
    idio = cp.Parameter(m)  # idiosyncratic std-dev, padded
    mu = cp.Parameter(m)
    factor_offset = cp.Parameter(r)  # F @ benchmark
    idio_offset = cp.Parameter(m)  # d * benchmark
    active = cp.Parameter(m, nonneg=True)

    constraints = [cp.multiply(1.0 - active, x) == 0]  # pin inactive weights to 0
    if long_only:
        constraints.append(x >= 0)
    constraints.append(cp.sum(x) == 1 if full_position else cp.sum(x) <= 1)

    tracking = cp.norm(
        cp.hstack([factors @ x - factor_offset, cp.multiply(idio, x) - idio_offset])
    )
    constraints.append(tracking <= budget)

    return {
        "problem": cp.Problem(cp.Maximize(mu @ x), constraints),
        "x": x,
        "factors": factors,
        "idio": idio,
        "mu": mu,
        "factor_offset": factor_offset,
        "idio_offset": idio_offset,
        "active": active,
        "size": m,
        "rank": r,
        "slots": {},
    }


def solve(
    handle: dict,
    active_indices: np.ndarray,
    benchmark: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    budget: float,
    long_only: bool,
) -> np.ndarray:
    """Normalizes the benchmark, sets the parameters, re-solves."""
    import cvxpy as cp

    # The benchmark is a full position by definition: renormalize the universe
    # weights over the active subset.
    total = float(benchmark.sum())
    benchmark = benchmark / total if total > 0 else np.full(benchmark.shape, 1.0 / benchmark.size)

    if budget <= 0.0:
        # No room to deviate, so the benchmark is the answer.
        return benchmark.copy()

    n = len(mu)
    slots = assign_slots(handle["slots"], active_indices, handle["size"])
    factors, idio, active = factor_params_at(sigma, slots, handle["size"], handle["rank"])
    padded_mu = np.zeros(handle["size"])
    padded_mu[slots] = mu
    padded_benchmark = np.zeros(handle["size"])
    padded_benchmark[slots] = benchmark

    handle["factors"].value = factors
    handle["idio"].value = idio
    handle["mu"].value = padded_mu
    handle["factor_offset"].value = factors @ padded_benchmark
    handle["idio_offset"].value = idio * padded_benchmark
    handle["active"].value = active

    try:
        handle["problem"].solve(solver=cp.SCS, warm_start=True)
    except cp.SolverError:
        return benchmark.copy()
    if handle["x"].value is None:
        return benchmark.copy()

    weights = np.asarray(handle["x"].value[slots], dtype=np.float64)
    return np.maximum(weights, 0.0) if long_only else weights


def build(
    *,
    bound: float,
    long_only: bool = True,
    full_position: bool = True,
    factor_rank: int = 20,
    **kwargs,
) -> MeanVariancePortfolio:
    """Constructs a benchmark-relative portfolio, tracking the universe weights."""
    budget = float(bound)
    return MeanVariancePortfolio(
        init_solver=lambda m: build_solver(m, factor_rank, budget, long_only, full_position),
        solve=lambda state, active, universe, previous, mu, sigma: solve(
            state.solver, active, universe, mu, sigma, budget, long_only
        ),
        **kwargs,
    )
