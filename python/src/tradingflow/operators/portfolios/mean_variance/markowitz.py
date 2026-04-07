"""Markowitz mean-variance portfolio optimization."""

import numpy as np
import scipy.sparse as sp
import scs

from ..mean_variance_portfolio import MeanVariancePortfolio


class Markowitz(MeanVariancePortfolio):
    """Markowitz mean-variance optimization (conic formulation 2.9).

    Solves the conic reformulation of:

        maximize  mu' x  -  delta * sqrt(x' Sigma x)
        subject to  1' x = 1
                    x >= 0   (if long_only)

    by factoring Sigma = L L' (Cholesky) and introducing an auxiliary
    variable t >= ||L' x||, yielding the second-order cone program:

        minimize  -mu' x + delta * t
        subject to  (t, L' x) in Q^{N+1}
                    1' x = 1
                    x >= 0   (if long_only)

    solved directly via SCS.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
    predicted_returns
        Handle to predicted returns, shape `(num_stocks,)`.
    covariance
        Handle to covariance matrix, shape `(num_stocks, num_stocks)`.
    risk_aversion
        Risk-aversion coefficient `delta`.
    long_only
        If `True` (default), enforce `x >= 0`.
    verbose
        If `True`, print optimization diagnostics to stdout.
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
    ) -> None:
        super().__init__(
            universe,
            predicted_returns,
            covariance,
            positions_fn=lambda state, mu, sigma: _solve(mu, sigma, risk_aversion, long_only, verbose),
        )


def _solve(mu: np.ndarray, sigma: np.ndarray, delta: float, long_only: bool, verbose: bool) -> np.ndarray:
    """Solve the Markowitz SOCP directly via SCS.

    Builds the second-order cone program in SCS standard form
    (minimize c'y  s.t.  A y + s = b,  s in K)  where
    y = [x; t] and K = {0}^1 x R+^N x Q^{N+1}.
    """
    N = len(mu)

    if verbose:
        print(f"  markowitz: mu contains data in range [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  markowitz: sigma contains data in range [{sigma.min():.4f}, {sigma.max():.4f}]")

    # Cholesky: Sigma = L @ L.T
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"  markowitz: Cholesky failed ({e}), using equal weights")
        return np.full(N, 1.0 / N)

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
    a.append(sp.hstack([sp.csc_matrix(np.ones((1, N))), sp.csc_matrix((1, 1))]))
    b.append(1.0)

    # Nonneg cone: A = [-I 0], b = 0
    if long_only:
        cone["l"] = N
        a.append(sp.hstack([-sp.eye(N, format="csc"), sp.csc_matrix((N, 1))]))
        b.extend([0.0] * N)

    # SOC: A = [0' -1; -L' 0], b = 0
    cone["q"] = [N + 1]
    a.append(sp.hstack([sp.csc_matrix((1, N)), sp.csc_matrix([[-1]])]))
    a.append(sp.hstack([sp.csc_matrix(-L.T), sp.csc_matrix((N, 1))]))
    b.extend([0.0] * (N + 1))

    # -- Solve ----------------------------------------------------------------
    solver = scs.SCS({"c": c, "A": sp.vstack(a), "b": np.array(b)}, cone)
    sol = solver.solve()

    if sol["info"]["status"] not in ("solved", "solved_inaccurate"):
        if verbose:
            print(f"  markowitz: solver status={sol['info']['status']}, using equal weights")
        return np.full(N, 1.0 / N)

    weights = sol["x"][:N].copy()

    if long_only:
        weights = np.maximum(weights, 0.0)

    if verbose:
        n_nonzero = (np.abs(weights) > 1e-6).sum()
        s = weights.sum()
        exp_ret = float(mu @ weights)
        exp_vol = float(np.sqrt(weights @ sigma @ weights))
        print(f"  markowitz: {n_nonzero}/{N} stocks, {s} invested, E[r]={exp_ret:.4f}, vol={exp_vol:.4f}")

    return weights
