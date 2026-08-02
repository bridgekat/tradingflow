use pyo3::types::PyDictMethods;

use super::VariancePredictor;
use crate::operators::predictor::Config;
use crate::python::py_operator_module;

/// What [`shrinkage`] pulls the sample covariance toward — the three targets
/// surveyed in Pantaleo et al. (2010), Section III.D.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Every variance the average sample variance, every covariance the
    /// average sample covariance. The most aggressive of the three: it keeps
    /// only the overall level of risk and comovement.
    CommonCovariance = 1,
    /// Sample variances kept, every correlation the average off-diagonal
    /// correlation. Retains the cross-sectional spread of volatility, which
    /// the common-covariance target discards.
    ConstantCorrelation = 2,
    /// The [`single_index`](super::single_index) factor covariance, so the
    /// structure imposed is a market factor rather than a flat constant.
    SingleIndex = 3,
}

/// Linear shrinkage of the sample covariance toward a structured target:
/// `Σ = αF + (1 - α)S`.
///
/// The sample covariance `S` is noisy but unbiased; the target `F` is heavily
/// biased but barely noisy. Mixing them beats either, and the Schäfer-Strimmer
/// (2005) estimator picks the intensity `α` analytically — no cross-validation
/// and no tuning parameter — by comparing the estimated variance of each
/// sample covariance entry against how far it sits from the target. A short
/// window shrinks hard, a long one barely at all.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn shrinkage(config: Config, target: Target) -> impl VariancePredictor {
    py_operator_module(
        "tradingflow.predictor.variance.shrinkage",
        config.params(|d| d.set_item("target", target as u8)),
    )
}
