"""Portfolios trading predicted return against predicted risk.

`markowitz` and `benchmark_relative` go through CVXPY, which needs the
covariance as a solver parameter. A full `(N, N)` matrix cannot be one at
universe scale — the DPP canonicalization map for an `N x N` parameter has
shape `N x N^2` — so both approximate it as `B Bᵀ + diag(d²)` from its top
eigenpairs, which makes the risk term DPP and the problem warm-startable
across rebalances. See `tradingflow.portfolio._factor`.

`admm_mnr` sidesteps CVXPY entirely, and with it the whole canonicalization
question: it never forms the covariance at all, only multiplies by it.
"""
