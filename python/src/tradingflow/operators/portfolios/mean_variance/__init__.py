"""Concrete mean-variance portfolio implementations.

- [`Markowitz`][tradingflow.operators.portfolios.mean_variance.markowitz.Markowitz]
  -Markowitz mean-variance optimization via CVXPY, with a pluggable
  [`Mode`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode] selecting
  among four equivalent formulations (min-variance-given-return,
  max-return-given-variance, variance-penalized, std-dev-penalized).
- [`MarkowitzADMM`][tradingflow.operators.portfolios.mean_variance.markowitz_admm.MarkowitzADMM]
  -Markowitz mean-variance optimization via augmented-saddle ADMM
  (NumPy/SciPy only, no CVXPY at runtime; robust to rank-deficient
  equality constraints).  Restricted to the
  [`MIN_MEAN_VARIANCE`][tradingflow.operators.portfolios.mean_variance.markowitz.Mode]
  mode.
- [`BenchmarkRelative`][tradingflow.operators.portfolios.mean_variance.benchmark_relative.BenchmarkRelative]
  -Benchmark-relative ("enhanced index") Markowitz: maximize predicted
  return subject to a hard tracking-error budget against the universe
  weights treated as the benchmark.  Single mode (max-return-given-TE),
  CVXPY backend.
- [`admm_mnr`] - unified matrix-free / dense ADMM-MNR for factor-model portfolios.
  One solver, three operators (DenseLSOperator / FactorLSOperator /
  BlockDiagLSOperator).  Sigma = F F^T + D is never formed in the matrix-free
  path (O(nk) per matvec); single-account is BlockDiag with K=1; box and
  rank-deficient equality constraints are handled natively; active-set
  completion yields a certified KKT point.  Entry points: solve_portfolio,
  solve_multi_account, admm_mnr, mnr_solve, and the operator classes.
"""

from .benchmark_relative import BenchmarkRelative
from .markowitz import Markowitz, Mode
from .markowitz_admm import MarkowitzADMM
from .admm_mnr import (
    LSOperator,
    DenseLSOperator,
    FactorLSOperator,
    BlockDiagLSOperator,
    factor_operator,
    dense_operator,
    dense_from_covariance,
    block_diag,
    admm_mnr,
    mnr_solve,
    minres_mnr,
    active_set_completion,
    completion_certificate,
    row_project_rhs,
    solve_portfolio,
    solve_multi_account,
    AdmmResult,
    MnrResult,
    MinresResult,
    CompletionInfo,
)

__all__ = [
    "BenchmarkRelative",
    "Markowitz",
    "MarkowitzADMM",
    "Mode",
    "LSOperator",
    "DenseLSOperator",
    "FactorLSOperator",
    "BlockDiagLSOperator",
    "factor_operator",
    "dense_operator",
    "dense_from_covariance",
    "block_diag",
    "admm_mnr",
    "mnr_solve",
    "minres_mnr",
    "active_set_completion",
    "completion_certificate",
    "row_project_rhs",
    "solve_portfolio",
    "solve_multi_account",
    "AdmmResult",
    "MnrResult",
    "MinresResult",
    "CompletionInfo",
]
