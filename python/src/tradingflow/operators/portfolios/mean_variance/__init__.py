"""Concrete mean-variance portfolio implementations.

- Markowitz - CVXPY mean-variance with pluggable Mode.
- MarkowitzADMM - augmented-saddle ADMM (NumPy/SciPy only), MIN_MEAN_VARIANCE.
- admm_mnr - unified matrix-free / dense ADMM-MNR for factor-model portfolios.
  One solver, three operators (DenseLSOperator / FactorLSOperator /
  BlockDiagLSOperator).  Sigma = F F^T + D is never formed in the matrix-free
  path (O(nk) per matvec); single-account is BlockDiag with K=1; box and
  rank-deficient equality constraints are handled natively; active-set
  completion yields a certified KKT point.  Entry points: solve_portfolio,
  solve_multi_account, admm_mnr, mnr_solve, and the operator classes.
"""

from .markowitz import Markowitz, Mode
from .markowitz_admm import MarkowitzADMM
from .admm_mnr import (
    LSOperator, DenseLSOperator, FactorLSOperator, BlockDiagLSOperator,
    factor_operator, dense_operator, dense_from_covariance, block_diag,
    admm_mnr, mnr_solve, minres_mnr, active_set_completion,
    completion_certificate, row_project_rhs,
    solve_portfolio, solve_multi_account,
    AdmmResult, MnrResult, MinresResult, CompletionInfo,
)

__all__ = [
    "Markowitz", "MarkowitzADMM", "Mode",
    "LSOperator", "DenseLSOperator", "FactorLSOperator", "BlockDiagLSOperator",
    "factor_operator", "dense_operator", "dense_from_covariance", "block_diag",
    "admm_mnr", "mnr_solve", "minres_mnr", "active_set_completion",
    "completion_certificate", "row_project_rhs",
    "solve_portfolio", "solve_multi_account",
    "AdmmResult", "MnrResult", "MinresResult", "CompletionInfo",
]
