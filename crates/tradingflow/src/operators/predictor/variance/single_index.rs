use super::VariancePredictor;
use crate::operators::predictor::Config;
use crate::python::py_segment_module;

/// Single-index factor model: one market factor plus idiosyncratic variance.
///
/// Regresses each stock on the equal-weighted cross-sectional mean return as a
/// market proxy and returns `Σ = σ_f² ββᵀ + diag(σ_ε²)` — the *SI* estimator
/// of Pantaleo et al. (2010). Estimating `N` betas and `N` residual variances
/// instead of `N(N+1)/2` covariances makes it well conditioned at any window
/// length, at the cost of assuming every correlation between two stocks runs
/// through the single factor.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn single_index(config: Config) -> impl VariancePredictor {
    py_segment_module(
        "tradingflow.predictor.variance.single_index",
        config.params(|_| Ok(())),
    )
}
