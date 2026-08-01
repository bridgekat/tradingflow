"""Mean-variance optimization by matrix-free ADMM-MNR — NumPy and SciPy only.

Solves the same problem as `markowitz` in `MIN_MEAN_VARIANCE` mode, maximizing
`muᵀx - delta xᵀΣx` subject to a budget and a box, but without CVXPY. The
covariance is never formed or factorized inside the solve: it enters only as
the operator `x -> B(Bᵀx) + d²x`, so each iteration costs `O(N k)` rather than
`O(N²)`, and there is no canonicalization step to amortize.
"""

import numpy as np

from .._base import MeanVariancePortfolio
from .._factor import factor_decompose
from . import _admm


def solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    delta: float,
    long_only: bool,
    cap: float | None,
    factor_rank: int | None,
    outer_tol: float,
    max_outer: int,
) -> np.ndarray:
    """Decomposes the covariance and runs the ADMM-MNR solve."""
    n = sigma.shape[0]

    if factor_rank is None:
        # Dense path: the solver builds its own operator from an eigen split of
        # the covariance. Exact, but O(N^3) per rebalance.
        result = _admm.solve_portfolio(
            covariance=sigma,
            expected_returns=mu,
            delta=delta,
            lower=0.0 if long_only else -np.inf,
            cap=cap,
            outer_tol=outer_tol,
            max_outer=max_outer,
        )
    else:
        loadings, idio = factor_decompose(sigma, factor_rank)
        result = _admm.solve_portfolio(
            factor_loadings=loadings,
            specific_var=idio**2,
            expected_returns=mu,
            delta=delta,
            lower=0.0 if long_only else -np.inf,
            cap=cap,
            outer_tol=outer_tol,
            max_outer=max_outer,
        )

    weights = np.asarray(result["weights"], dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        return np.full(n, 1.0 / n)
    return np.maximum(weights, 0.0) if long_only else weights


def build(
    *,
    delta: float,
    long_only: bool = True,
    cap: float | None = None,
    factor_rank: int | None = 20,
    outer_tol: float = 1e-3,
    max_outer: int = 400,
    **kwargs,
) -> MeanVariancePortfolio:
    """Constructs a matrix-free ADMM mean-variance portfolio.

    `delta` is the risk-aversion coefficient, `cap` an optional per-name upper
    bound, and `factor_rank` the rank of the covariance approximation the
    matrix-free operator is built from — `None` switches to the exact but cubic
    dense path.
    """
    assert delta > 0.0, "delta must be positive"
    return MeanVariancePortfolio(
        solve=lambda state, active, universe, previous, mu, sigma: solve(
            mu,
            sigma,
            delta=float(delta),
            long_only=long_only,
            cap=cap,
            factor_rank=factor_rank,
            outer_tol=outer_tol,
            max_outer=max_outer,
        ),
        **kwargs,
    )
