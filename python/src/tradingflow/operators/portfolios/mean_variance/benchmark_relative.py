"""Benchmark-relative (tracking-error-budgeted) mean-variance portfolio optimization."""

from typing import Any

import numpy as np
import scipy as sp
import cvxpy as cp

from ..mean_variance_portfolio import MeanVariancePortfolio


class BenchmarkRelative(MeanVariancePortfolio):
    r"""Benchmark-relative Markowitz: maximize return subject to a tracking-error budget.

    Solves the single-mode QP

    \[
        \max_{x}\;\mu^\top x
        \quad\text{s.t.}\quad
            (x-x_{\mathrm{bm}})^\top \Sigma (x-x_{\mathrm{bm}}) \le \gamma_{TE}^2,\;\;
            \mathbf{1}^\top x = 1\;[\,\le 1\,],\;\;
            x \ge 0,
    \]

    where \(x_{\mathrm{bm}}\) is the benchmark weight vector (the
    `universe` input, restricted to the in-mask subset and renormalised
    to sum to 1 by
    [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio])
    and \(\gamma_{TE}\) is a *daily* tracking-error standard-deviation
    budget (in the same return units as the predicted moments).  The
    constraint is encoded as the SOC \(\|L^\top (x-x_{\mathrm{bm}})\|_2 \le \gamma_{TE}\)
    with \(\Sigma = L L^\top\) from an LDL decomposition (matching
    [`Markowitz`][tradingflow.operators.portfolios.mean_variance.markowitz.Markowitz]).

    The benchmark-relative formulation is the natural building block
    for index-enhancement strategies: at \(\gamma_{TE}\to 0\) the
    optimizer is forced to hold the benchmark; at \(\gamma_{TE}\to\infty\)
    it recovers the unconstrained maximum-return portfolio.  Unlike
    absolute Markowitz, the worst-case loss is bounded with respect to
    the benchmark, so even broken \(\mu\) predictions cannot drive the
    portfolio arbitrarily far from \(x_{\mathrm{bm}}\).

    ## Tracking-error units

    `bound` is interpreted in the same time-scale as the upstream
    \(\mu\) / \(\Sigma\) predictions (typically *daily* in TradingFlow,
    since the standard
    [`Ridge`][tradingflow.operators.predictors.mean.ridge.Ridge] /
    [`Shrinkage`][tradingflow.operators.predictors.variance.shrinkage.Shrinkage]
    pair is trained on `target_offset=1` daily log returns).  To target
    an *annualised* tracking-error of \(\bar\gamma_{TE}\) (e.g.\ a 5\%
    enhanced-index budget), pass `bound=` \(\bar\gamma_{TE} / \sqrt{252}\).

    ## Feasibility

    Holding \(x = x_{\mathrm{bm}}\) trivially satisfies every constraint
    (TE = 0, budget = 1, long-only since \(x_{\mathrm{bm}}\ge 0\)), so
    the problem is always feasible; on solver failure or numerical
    issues the operator falls back to \(x_{\mathrm{bm}}\).

    Parameters
    ----------
    universe
        Handle to universe / benchmark weights, shape `(num_stocks,)`.
        Plays a dual role: the active set (non-zero entries trigger
        rebalances) and the benchmark vector \(x_{\mathrm{bm}}\) used
        in the TE constraint.
    predicted_returns
        Handle to predicted log-returns, shape `(num_stocks,)`.
    covariance
        Handle to predicted log-return covariance matrix, shape
        `(num_stocks, num_stocks)`.  Both inputs are converted to
        linear-return moments by
        [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio]
        before the QP is set up.
    bound
        Tracking-error budget \(\gamma_{TE}\) in the same time-scale
        as the predicted moments (daily by default).  Must be positive;
        non-positive values trigger a fallback to the benchmark.
    long_only
        If `True` (default), enforce \(x \geq 0\).
    full_position
        If `True` (default), require full investment \(\mathbf{1}^\top x = 1\).
        If `False`, allow underinvestment \(\mathbf{1}^\top x \le 1\).
    verbose
        If `True`, print optimization diagnostics to stdout.
    **kwargs
        Forwarded to [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio].
    """

    def __init__(
        self,
        universe,
        predicted_returns,
        covariance,
        *,
        bound: float,
        long_only: bool = True,
        full_position: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            predicted_returns,
            covariance,
            positions_fn=lambda state, mu, sigma, x_bm: _solve(
                mu, sigma, x_bm, bound, long_only, full_position, verbose
            ),
            **kwargs,
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
    """Solve max mu^T x s.t. TE(x, x_bm) <= gamma_te, x ∈ feasible set."""
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
    # sigma = L L^T.  Same convention as `Markowitz._solve`; clamps any
    # numerically-negative D entries to zero.
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
