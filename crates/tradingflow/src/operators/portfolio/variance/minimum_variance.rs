use pyo3::types::PyDictMethods;

use super::VariancePortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_operator_module;

/// Global minimum variance: the book with the least predicted risk, subject to
/// being fully invested.
///
/// Forecasts no returns at all, which is the point — expected returns are much
/// harder to estimate than covariances, and a mean-variance optimizer amplifies
/// errors in them. What it is sensitive to instead is the conditioning of the
/// covariance: a near-singular estimate offers apparently riskless directions
/// and this will lever straight into them, so pair it with a structured
/// predictor rather than a raw
/// [`sample`](crate::operators::predictor::variance::sample) covariance.
///
/// `long_only` additionally forbids short positions, which acts as an implicit
/// regularizer on exactly that failure.
///
/// `factor_rank` is the rank of the covariance approximation the solver
/// optimizes against; see the [module docs](super::super::mean_variance) for
/// why the full matrix cannot be used directly.
///
/// Requires CVXPY in the embedded interpreter.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn minimum_variance(
    config: Config,
    long_only: bool,
    factor_rank: usize,
) -> impl VariancePortfolio {
    py_operator_module(
        "tradingflow.portfolio.variance.minimum_variance",
        config.params(|d| {
            d.set_item("long_only", long_only)?;
            d.set_item("factor_rank", factor_rank)
        }),
    )
}
