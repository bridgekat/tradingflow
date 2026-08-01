use pyo3::types::PyDictMethods;

use super::MeanVariancePortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_segment_module;

/// Maximizes predicted return subject to a tracking-error budget against the
/// universe.
///
/// The enhanced-indexing formulation: the universe weights are read as the
/// benchmark, and the book may deviate from them only as far as
/// `sqrt((x - w)ᵀΣ(x - w)) <= bound` allows. This is what an index-relative
/// mandate is actually measured on, and it is far better behaved than absolute
/// [`markowitz`](super::markowitz) under estimation error — errors in `μ` can
/// only move the book within the tracking-error ball, so a bad forecast costs
/// tracking error rather than the whole portfolio.
///
/// `bound` is the tracking-error budget in the same units and horizon as the
/// covariance, so a daily `Σ` makes it a daily figure. A non-positive budget
/// leaves no room to deviate and returns the benchmark itself.
///
/// Requires CVXPY in the embedded interpreter.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn benchmark_relative(
    config: Config,
    bound: f64,
    long_only: bool,
    full_position: bool,
    factor_rank: usize,
) -> impl MeanVariancePortfolio {
    py_segment_module(
        "tradingflow.portfolio.mean_variance.benchmark_relative",
        config.params(|d| {
            d.set_item("bound", bound)?;
            d.set_item("long_only", long_only)?;
            d.set_item("full_position", full_position)?;
            d.set_item("factor_rank", factor_rank)
        }),
    )
}
