"""Ported portfolio operators for the flowops host.

Subpackages:

- ``mean`` - mean-only portfolios (proportional, rank_bucket, rank_equal,
  rank_linear, softmax).  NumPy only.
- ``variance`` - minimum_variance (cvxpy; blocker on the free-threaded venv).
- ``mean_variance`` - markowitz / benchmark_relative (cvxpy; blockers),
  markowitz_admm (NumPy/SciPy only), plus the augmented_saddle_admm and
  admm_mnr solver libraries.

Each leaf module exposes ``build(**kwargs) -> op`` implementing the host contract.
The shared bases live in ``flowops.portfolios._base``.  Submodules are not
imported eagerly here so that importing the package does not pull in cvxpy.
"""
