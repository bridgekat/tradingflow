use super::MeanPortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_operator_module;

/// Weights each stock in proportion to its predicted return.
///
/// Stocks predicted to fall get nothing; the rest split the book by how much
/// they are expected to return. The most direct translation of a prediction
/// into a position, and the one most exposed to a bad one — with no risk model
/// it will concentrate the whole book into a single volatile name that happens
/// to score highest.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn proportional(config: Config) -> impl MeanPortfolio {
    py_operator_module(
        "tradingflow.portfolio.mean.proportional",
        config.params(|_| Ok(())),
    )
}
