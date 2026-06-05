"""Markowitz mean-variance portfolio optimization via augmented-saddle ADMM.

flowops host port.  NumPy/SciPy only (no cvxpy) - fully runnable on the
free-threaded interpreter.  Restricted to `Mode.MIN_MEAN_VARIANCE`.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as spla

from flowops.portfolios._base import MeanVariancePortfolio
from flowops.portfolios.mean_variance.augmented_saddle_admm import solve_admm
from flowops.portfolios.mean_variance._modes import Mode

__all__ = ["MarkowitzADMM", "Mode", "build"]


class MarkowitzADMM(MeanVariancePortfolio):
    r"""Markowitz mean-variance optimization via augmented-saddle ADMM.

    Inputs: ``(universe, predicted_returns, covariance)`` with shapes
    ``(num_stocks,)`` / ``(num_stocks,)`` / ``(num_stocks, num_stocks)``.
    Output: position weights, ``(num_stocks,)``.

    NumPy/SciPy only.  ``mode`` must be ``Mode.MIN_MEAN_VARIANCE`` (or its int
    value ``3``); ``bound`` is the variance-penalty coefficient ``delta``.
    """

    def __init__(
        self,
        *,
        mode: Mode | int,
        bound: float,
        long_only: bool = True,
        full_position: bool = True,
        verbose: bool = False,
        rho: float = 1.0,
        inner_rtol: float = 1e-4,
        outer_tol: float = 1e-4,
        max_outer: int = 2000,
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        mode = Mode(mode)
        if mode != Mode.MIN_MEAN_VARIANCE:
            raise ValueError(
                f"MarkowitzADMM currently supports only Mode.MIN_MEAN_VARIANCE, "
                f"got {mode.name}.  Use Markowitz for the other modes."
            )
        super().__init__(
            positions_fn=lambda state, mu, sigma, x_bm: _solve(
                mu,
                sigma,
                delta=float(bound),
                long_only=long_only,
                full_position=full_position,
                verbose=verbose,
                rho=rho,
                inner_rtol=inner_rtol,
                outer_tol=outer_tol,
                max_outer=max_outer,
            ),
            num_stocks=num_stocks,
            logarithmic=logarithmic,
        )


def _solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    delta: float,
    long_only: bool,
    full_position: bool,
    verbose: bool,
    rho: float,
    inner_rtol: float,
    outer_tol: float,
    max_outer: int,
) -> np.ndarray:
    r"""Build the augmented-saddle problem and dispatch to `solve_admm`."""
    N = sigma.shape[0]

    if verbose:
        print(f"  markowitz_admm: mu shape {mu.shape} range [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  markowitz_admm: sigma shape {sigma.shape} range [{sigma.min():.4f}, {sigma.max():.4f}]")

    if delta <= 0.0:
        print(f"  markowitz_admm: invalid delta={delta}, using equal weights")
        return np.full(N, 1.0 / N)

    # LDL decomposition: sigma = L_ldl[perm, :] @ D @ L_ldl[perm, :].T,
    # then clamp negative D entries to zero (matching Markowitz).
    L_ldl, D, _perm = spla.ldl(sigma)
    L = L_ldl * np.sqrt(np.maximum(np.diag(D), 0.0)).reshape(1, N)

    if verbose:
        error = float(np.max(np.abs(sigma - L @ L.T)))
        print(f"  markowitz_admm: LDL max error {error:.4} (non-zero may indicate non-positive-semidefinite sigma)")

    n_slack_cash = 0 if full_position else 1
    n_total = N + n_slack_cash

    sqrt_2d = float(np.sqrt(2.0 * delta))

    # b for the residual block solves A^T b_inner = mu (with
    # A = sqrt(2*delta) L^T), equivalently L b_inner = mu / sqrt_2d.
    b_inner, *_ = np.linalg.lstsq(L, mu / sqrt_2d, rcond=None)

    bud_row = np.zeros(n_total)
    bud_row[:N] = 1.0
    if n_slack_cash:
        bud_row[N] = 1.0
    C = bud_row.reshape(1, n_total)
    d_vec = np.array([1.0])
    m = 1

    def K_times(w: np.ndarray) -> np.ndarray:
        r = w[:N]
        x_aug = w[N : N + n_total]
        lam = w[N + n_total :]
        out = np.empty(N + n_total + m)
        out[:N] = r + sqrt_2d * (L.T @ x_aug[:N])
        ATr = np.zeros(n_total)
        ATr[:N] = sqrt_2d * (L @ r)
        out[N : N + n_total] = ATr - rho * x_aug - C.T @ lam
        out[N + n_total :] = -(C @ x_aug)
        return out

    def rhs_builder(v: np.ndarray) -> np.ndarray:
        out = np.empty(N + n_total + m)
        out[:N] = b_inner
        out[N : N + n_total] = -rho * v
        out[N + n_total :] = -d_vec
        return out

    def project_fn(z: np.ndarray) -> np.ndarray:
        out = z.copy()
        if long_only:
            out[:N] = np.maximum(out[:N], 0.0)
        if n_slack_cash:
            out[N:] = np.maximum(out[N:], 0.0)
        return out

    result = solve_admm(
        K_times,
        rhs_builder,
        n_residual_block=N,
        n_x_block=n_total,
        n_lambda_block=m,
        rho=rho,
        project_fn=project_fn,
        inner_rtol=inner_rtol,
        outer_tol=outer_tol,
        max_outer=max_outer,
    )

    weights = result.x[:N].copy()
    if long_only:
        weights = np.maximum(weights, 0.0)

    if verbose:
        n_nonzero = int((np.abs(weights) > 1e-6).sum())
        s = float(weights.sum())
        exp_ret = float(mu @ weights)
        exp_vol = float(np.sqrt(max(weights @ sigma @ weights, 0.0)))
        print(
            f"  markowitz_admm: {n_nonzero}/{N} stocks, {s:.4f} invested, "
            f"E[r]={exp_ret:.4f}, vol={exp_vol:.4f} "
            f"(outer={result.outer_iters}, inner={result.inner_iters_total}, mnr={result.mnr_fires})"
        )

    return weights


def build(**kwargs) -> MarkowitzADMM:
    return MarkowitzADMM(**kwargs)
