"""Markowitz mean-variance portfolio optimization via augmented-saddle ADMM."""

from __future__ import annotations

import numpy as np
import scipy.linalg as spla

from ..mean_variance_portfolio import MeanVariancePortfolio
from .augmented_saddle_admm import solve_admm
from .markowitz import Mode


class MarkowitzADMM(MeanVariancePortfolio):
    r"""Markowitz mean-variance optimization via augmented-saddle ADMM.

    A drop-in alternative to
    [`Markowitz`][tradingflow.operators.portfolios.mean_variance.markowitz.Markowitz]
    that solves the underlying QP using an ADMM splitting on
    non-negativity together with an *augmented saddle* MINRES solver
    for the equality-constrained inner problem.  Wraps the reference
    [`solve_admm`][tradingflow.operators.portfolios.mean_variance.augmented_saddle_admm.solve_admm];
    the operator only builds the problem-specific `K_times`,
    `rhs_builder`, constraint matrix `C`, and `project_fn` (the
    block-wise non-negativity projector that distinguishes primary
    weights from the cash slack).  Compared with the CVXPY-backed
    variant, this solver:

    - depends only on NumPy and SciPy (no CVXPY at runtime);
    - is robust to rank-deficient equality constraint matrices: the
      inner KKT system is symmetric and solved by MINRES, with a
      rank-one minimum-norm refinement step that handles the
      null-space contribution to the dual when present;
    - exposes the equality dual `lam` if a future extension needs it.

    Currently restricted to the `MIN_MEAN_VARIANCE` mode
    (\(\delta x^T \Sigma x - \mu^T x\)).  Other
    [`Mode`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode]
    values require either an inequality constraint or an SOC
    constraint that are out of scope for this variant; use
    [`Markowitz`][tradingflow.operators.portfolios.mean_variance.markowitz.Markowitz]
    for those.

    `full_position=False` is supported via a non-negative cash slack
    \(s_c\) with \(\mathbf{1}^T x + s_c = 1\).  The operator's
    `project_fn` projects the cash slack to \(\geq 0\) unconditionally
    and projects the primary weights to \(\geq 0\) only when
    `long_only=True`.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
    predicted_returns
        Handle to predicted log-returns, shape `(num_stocks,)`.
    covariance
        Handle to predicted log-return covariance matrix, shape
        `(num_stocks, num_stocks)`.  Both inputs are converted to
        linear-return moments by
        [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio]
        before the QP is set up.  `sigma` is decomposed via LDL with
        negative-eigenvalue clamping (matching
        [`Markowitz`][tradingflow.operators.portfolios.mean_variance.markowitz.Markowitz]),
        so this operator accepts any symmetric `sigma` rather than
        requiring strict positive definiteness.
    mode
        Optimization mode; must be `Mode.MIN_MEAN_VARIANCE`.
        Other modes raise `ValueError`.
    bound
        Variance penalty coefficient \(\delta\).  Must be positive;
        non-positive values trigger a fallback to equal weights.
    long_only
        If `True` (default), enforce \(x \geq 0\) on the primary
        weights.  The cash slack is always \(\geq 0\) regardless.
    full_position
        If `True` (default), require full investment \(\mathbf{1}^T x = 1\).
        If `False`, allow underinvestment via a non-negative cash slack.
    verbose
        If `True`, print optimization diagnostics to stdout.
    rho
        ADMM penalty parameter.  Default `1.0`.  If `sigma` has entries
        that are orders of magnitude away from unity, rescaling `rho`
        accordingly can improve outer-loop convergence speed.
    inner_rtol
        MINRES inner-stopping tolerance \(\|K r\|/\|K\,\mathrm{rhs}\| \leq r_{tol}\).
    outer_tol
        ADMM primal/dual residual tolerance.
    max_outer
        Outer iteration cap.
    **kwargs
        Forwarded to [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio].
    """

    def __init__(
        self,
        universe,
        predicted_returns,
        covariance,
        *,
        mode: Mode,
        bound: float,
        long_only: bool = True,
        full_position: bool = True,
        verbose: bool = False,
        rho: float = 1.0,
        inner_rtol: float = 1e-4,
        outer_tol: float = 1e-4,
        max_outer: int = 2000,
        **kwargs,
    ) -> None:
        if mode != Mode.MIN_MEAN_VARIANCE:
            raise ValueError(
                f"MarkowitzADMM currently supports only Mode.MIN_MEAN_VARIANCE, "
                f"got {mode.name}.  Use Markowitz for the other modes."
            )
        super().__init__(
            universe,
            predicted_returns,
            covariance,
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
            **kwargs,
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
    r"""Build the augmented-saddle problem and dispatch to `solve_admm`.

    Minimises \(\delta\,x^T \Sigma x - \mu^T x\) subject to a budget
    constraint and the non-negativity bound, with the cash slack
    \(s_c\) (when `full_position=False`) carried as an extra component
    of the augmented `x` vector `x_aug = [x_primary (N); s_cash? (0/1)]`.
    """
    N = sigma.shape[0]

    if verbose:
        print(f"  markowitz_admm: mu shape {mu.shape} range [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  markowitz_admm: sigma shape {sigma.shape} range [{sigma.min():.4f}, {sigma.max():.4f}]")

    if delta <= 0.0:
        print(f"  markowitz_admm: invalid delta={delta}, using equal weights")
        return np.full(N, 1.0 / N)

    # LDL decomposition: sigma = L_ldl[perm, :] @ D @ L_ldl[perm, :].T,
    # then clamp negative D entries to zero (matching Markowitz).  After
    # clamping, L @ L.T is the PSD projection of sigma, which is what
    # the QP effectively minimises.  L itself is not lower-triangular
    # (only L[perm, :] is), so the b_inner solve below uses lstsq.
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
    # lstsq returns the minimum-norm solution when L has been
    # rank-deficient after eigenvalue clamping; for full-rank L it
    # reduces to a unique solve.
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
