"""Markowitz mean-variance portfolio optimization."""

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scs

from ..mean_variance_portfolio import MeanVariancePortfolio


class MarkowitzSCS(MeanVariancePortfolio):
    r"""Markowitz mean-variance optimization.

    Solves the conic reformulation of:

    \[
    \text{maximize} \quad \mu^T x - \delta \sqrt{x^T \Sigma x}
    \]
    \[
    \text{subject to} \quad \mathbf{1}^T x = 1, \quad x \geq 0 \text{ (if long-only)}
    \]

    by factoring \(\Sigma = L L^T\) (Cholesky) and introducing an auxiliary
    variable \(t \geq \|L^T x\|\), yielding the second-order cone program:

    \[
    \text{minimize} \quad -\mu^T x + \delta t
    \]
    \[
    \text{subject to} \quad (t, L^T x) \in Q^{N+1}, \quad \mathbf{1}^T x = 1, \quad x \geq 0 \text{ (if long-only)}
    \]

    solved directly via SCS.

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
        before the SOCP is set up.
    risk_aversion
        Risk-aversion coefficient \(\delta\).
    long_only
        If `True` (default), enforce \(x \geq 0\).
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
        risk_aversion: float = 1.0,
        long_only: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            predicted_returns,
            covariance,
            positions_fn=lambda state, mu, sigma: _solve(mu, sigma, risk_aversion, long_only, verbose),
            **kwargs,
        )


def _solve(mu: np.ndarray, sigma: np.ndarray, delta: float, long_only: bool, verbose: bool) -> np.ndarray:
    """Solve the Markowitz SOCP directly via SCS."""
    N = len(mu)

    if verbose:
        print(f"  markowitz_scs: mu has shape {mu.shape} and range [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  markowitz_scs: sigma has shape {sigma.shape} and range [{sigma.min():.4f}, {sigma.max():.4f}]")

    # LDL decomposition: sigma = L @ D @ L.T, where D diagonal and L[perm, :] lower-triangular.
    L, D, perm = sp.linalg.ldl(sigma)
    L = L * np.sqrt(np.maximum(np.diag(D), 0.0)).reshape(1, N)

    if verbose:
        error = np.max(np.abs(sigma - L @ L.T))
        print(f"  markowitz_scs: L has shape {L.shape} and range [{L.min():.4f}, {L.max():.4f}]")
        print(f"  markowitz_scs: LDL max error {error:.4} (non-zero may indicate non-positive-semidefinite sigma)")

    # Decision variables: [x, t].
    n_vars = N + 1

    # Objective: minimize -mu'x + delta * t.
    c = np.empty(n_vars)
    c[:N] = -mu
    c[N] = delta

    # Constraint matrix A (block-sparse CSC).
    #   Row layout (must match cone order):
    #   [0]         zero:    [1 ... 1  0]
    #   [1..N]      nonneg:  [-I_N     0]        (if long_only)
    #   [r]         SOC[0]:  [0 ... 0 -1]
    #   [r+1..r+N]  SOC[1:]: [-L'      0]
    a = []

    # RHS vector b.
    b = []

    # Cone specification.
    cone = {}

    # Zero cone: A = [1' 0], b = 1
    cone["z"] = 1
    a.append(sparse.hstack([sparse.csc_matrix(np.ones((1, N))), sparse.csc_matrix((1, 1))]))
    b.append(1.0)

    # Nonneg cone: A = [-I 0], b = 0
    if long_only:
        cone["l"] = N
        a.append(sparse.hstack([-sparse.eye(N, format="csc"), sparse.csc_matrix((N, 1))]))
        b.extend([0.0] * N)

    # SOC: A = [0' -1; -L' 0], b = 0
    cone["q"] = [N + 1]
    a.append(sparse.hstack([sparse.csc_matrix((1, N)), sparse.csc_matrix([[-1]])]))
    a.append(sparse.hstack([sparse.csc_matrix(-L.T), sparse.csc_matrix((N, 1))]))
    b.extend([0.0] * (N + 1))

    # Solve the problem.
    solver = scs.SCS({"c": c, "A": sparse.vstack(a), "b": np.array(b)}, cone)
    sol = solver.solve()

    if sol["info"]["status"] not in ("solved", "solved_inaccurate"):
        print(f"  markowitz_scs: solver failed, using equal weights")
        return np.full(N, 1.0 / N)

    weights = sol["x"][:N].copy()

    if long_only:
        weights = np.maximum(weights, 0.0)

    if verbose:
        n_nonzero = (np.abs(weights) > 1e-6).sum()
        s = weights.sum()
        exp_ret = float(mu @ weights)
        exp_vol = float(np.sqrt(weights @ sigma @ weights))
        print(f"  markowitz_scs: problem status: {sol['info']['status']}")
        print(f"  markowitz_scs: {n_nonzero}/{N} stocks, {s:.4f} invested, E[r]={exp_ret:.4f}, vol={exp_vol:.4f}")

    return weights
