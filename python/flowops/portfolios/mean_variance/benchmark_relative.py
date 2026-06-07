"""Benchmark-relative (tracking-error-budgeted) mean-variance optimization.

flowops host port.  Warm-started via a factor-model risk (see `markowitz` and
`flowops.portfolios._factor`): the covariance is approximated as
`Sigma ~= B B^T + diag(d^2)` (top-`factor_rank` SVD low-rank + diagonal), so the
tracking error

    TE(x) = sqrt((x - x_bm)^T Sigma (x - x_bm))
          ~= || [ F x - F x_bm ;  d (.) x - d (.) x_bm ] ||   (F = B^T)

is a DPP second-order-cone constraint.  `F x_bm` and `d (.) x_bm` are precomputed
each solve and carried as parameters (a `Parameter @ Parameter` term would
otherwise break DPP).  The problem is built once, sized to `max_universe_size`,
and re-solved with parameters set and the previous positions seeded.

NOTE (blocker): depends on cvxpy (SOC), NOT importable on `.venv-ft`; the cvxpy
import is deferred to `init`/solve time.
"""

from __future__ import annotations

import numpy as np

from flowops.portfolios._base import MeanVariancePortfolio
from flowops.portfolios._factor import assign_slots, factor_params_at

__all__ = ["BenchmarkRelative", "build"]


class BenchmarkRelative(MeanVariancePortfolio):
    r"""Benchmark-relative Markowitz: maximize return subject to a tracking-error budget.

    Inputs: ``(universe, predicted_returns, covariance)`` with shapes
    ``(num_stocks,)`` / ``(num_stocks,)`` / ``(num_stocks, num_stocks)``.  The
    ``universe`` doubles as the benchmark ``x_bm`` (restricted to the active subset
    and renormalised to sum to 1 by the base).
    Output: position weights, ``(num_stocks,)``.

    ``bound`` is the tracking-error budget ``gamma_TE``; ``factor_rank`` (default 20)
    is the rank of the SVD low-rank covariance approximation.
    """

    def __init__(
        self,
        *,
        bound: float,
        long_only: bool = True,
        full_position: bool = True,
        verbose: bool = False,
        num_stocks: int | None = None,
        max_universe_size: int | None = None,
        factor_rank: int = 20,
        logarithmic: bool = True,
    ) -> None:
        gamma_te = float(bound)
        factor_rank = int(factor_rank)
        super().__init__(
            init_fn=lambda m: _build(m, factor_rank, gamma_te, long_only, full_position),
            positions_fn=lambda state, active_idx, sub_universe, x_prev, mu, sigma: _solve(
                state.solver, active_idx, sub_universe, mu, sigma, gamma_te, long_only, verbose
            ),
            num_stocks=num_stocks,
            max_universe_size=max_universe_size,
            logarithmic=logarithmic,
        )


def _build(max_universe_size: int, factor_rank: int, gamma_te: float, long_only: bool, full_position: bool) -> dict:
    """Build the fixed-size, DPP, factor-model TE problem once (per `init`)."""
    import cvxpy as cp

    m, r = int(max_universe_size), int(factor_rank)
    x = cp.Variable(m)
    F = cp.Parameter((r, m))               # B.T (factor loadings), padded
    d = cp.Parameter(m)                    # idiosyncratic std-dev, padded
    mu = cp.Parameter(m)                   # expected returns, padded
    f_off = cp.Parameter(r)                # F @ x_bm  (precomputed; keeps TE DPP)
    i_off = cp.Parameter(m)                # d (.) x_bm
    active = cp.Parameter(m, nonneg=True)

    constraints = [cp.multiply(1.0 - active, x) == 0]  # pin inactive weights to 0
    if long_only:
        constraints.append(x >= 0)
    constraints.append(cp.sum(x) == 1 if full_position else cp.sum(x) <= 1)
    # ||[F(x - x_bm); d(.)(x - x_bm)]||_2 <= gamma_te, offsets precomputed.
    te = cp.norm(cp.hstack([F @ x - f_off, cp.multiply(d, x) - i_off]))
    constraints.append(te <= gamma_te)

    prob = cp.Problem(cp.Maximize(mu @ x), constraints)
    return {
        "prob": prob, "x": x, "F": F, "d": d, "mu": mu,
        "f_off": f_off, "i_off": i_off, "active": active, "m": m, "r": r, "slot_of": {},
    }


def _solve(
    handle: dict,
    active_idx: np.ndarray,
    x_bm: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    gamma_te: float,
    long_only: bool,
    verbose: bool,
) -> np.ndarray:
    """Map the active stocks to stable slots, set the factor-model parameters (with
    the precomputed TE offsets), and re-solve.  No `x.value` seeding (cvxpy uses its
    cached previous solution, kept aligned by the stable slots)."""
    import cvxpy as cp

    # The benchmark must be a full position: renormalise x_bm to sum to 1 over the
    # active subset (the base hands it the raw universe weights).
    s = float(x_bm.sum())
    x_bm = x_bm / s if s > 0 else np.full(x_bm.shape, 1.0 / x_bm.size)

    if not (gamma_te > 0):
        if verbose:
            print(f"  benchmark_relative: non-positive TE budget {gamma_te}, returning benchmark")
        return x_bm.copy()

    m, n = handle["m"], len(mu)
    slots = assign_slots(handle["slot_of"], active_idx, m)
    F_val, d_val, active = factor_params_at(sigma, slots, m, handle["r"])
    mu_val = np.zeros(m)
    mu_val[slots] = mu
    x_bm_val = np.zeros(m)
    x_bm_val[slots] = x_bm

    handle["F"].value = F_val
    handle["d"].value = d_val
    handle["mu"].value = mu_val
    handle["f_off"].value = F_val @ x_bm_val        # F x_bm
    handle["i_off"].value = d_val * x_bm_val         # d (.) x_bm
    handle["active"].value = active

    try:
        handle["prob"].solve(solver=cp.SCS, warm_start=True)
    except cp.SolverError as e:
        print(f"  benchmark_relative: solver failed ({e}), falling back to benchmark")
        return x_bm.copy()

    if handle["x"].value is None:
        print(f"  benchmark_relative: no solution (status={handle['prob'].status}), falling back to benchmark")
        return x_bm.copy()

    weights = np.asarray(handle["x"].value[slots], dtype=np.float64)  # active order
    if long_only:
        weights = np.maximum(weights, 0.0)

    if verbose:
        active_w = weights - x_bm
        te_realised = float(np.sqrt(max(active_w @ sigma @ active_w, 0.0)))
        exp_ret = float(mu @ weights)
        n_nonzero = int((np.abs(weights) > 1e-6).sum())
        print(
            f"  benchmark_relative: status={handle['prob'].status}, {n_nonzero}/{n} stocks, "
            f"sum={weights.sum():.4f}, E[r]={exp_ret:.4f}, TE={te_realised:.6f}/{gamma_te:.6f}"
        )

    return weights


def build(**kwargs) -> BenchmarkRelative:
    return BenchmarkRelative(**kwargs)
