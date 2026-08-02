use pyo3::types::PyDictMethods;

use super::MeanVariancePortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_operator_module;

/// Which way [`markowitz`] states the return-versus-risk trade-off.
///
/// All four trace the same efficient frontier; they differ in what `bound`
/// means and therefore in what is easy to specify. Every mode carries the
/// budget constraint, and optionally long-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Minimize `xᵀΣx` subject to `μᵀx >= bound`. `bound` is a required
    /// return — the natural statement when a mandate names a target.
    MinVarianceGivenReturn = 1,
    /// Maximize `μᵀx` subject to `sqrt(xᵀΣx) <= bound`. `bound` is a
    /// volatility budget, in the same units as the returns.
    MaxReturnGivenStdDev = 2,
    /// Maximize `μᵀx - bound · xᵀΣx`. `bound` is a risk-aversion coefficient
    /// with no direct interpretation, but the objective is unconstrained
    /// beyond the budget and so the easiest to solve.
    MinMeanVariance = 3,
    /// Maximize `μᵀx - bound · sqrt(xᵀΣx)`. Penalizes volatility rather than
    /// variance, so `bound` is a Sharpe-like price of risk.
    MinMeanStdDev = 4,
}

/// Classical Markowitz mean-variance optimization.
///
/// Maximizes predicted return against predicted risk in whichever form
/// [`Mode`] states, subject to a budget and optionally to long-only. Be aware
/// of what it does with a bad forecast: the optimizer will find and exploit
/// every apparent arbitrage in `μ` and `Σ`, so estimation error concentrates
/// the book rather than diversifying it. That is the standard reason to pair
/// it with a shrunk covariance and to prefer
/// [`benchmark_relative`](super::benchmark_relative) under an index mandate.
///
/// `full_position` requires the weights to sum to exactly one rather than at
/// most one, so the book is always fully invested. `factor_rank` is the rank
/// of the covariance approximation; see the [module docs](super).
///
/// Requires CVXPY in the embedded interpreter.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn markowitz(
    config: Config,
    mode: Mode,
    bound: f64,
    long_only: bool,
    full_position: bool,
    factor_rank: usize,
) -> impl MeanVariancePortfolio {
    py_operator_module(
        "tradingflow.portfolio.mean_variance.markowitz",
        config.params(|d| {
            d.set_item("mode", mode as u8)?;
            d.set_item("bound", bound)?;
            d.set_item("long_only", long_only)?;
            d.set_item("full_position", full_position)?;
            d.set_item("factor_rank", factor_rank)
        }),
    )
}
