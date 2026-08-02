use crate::data::Instant;
use crate::graph::Operator;
use crate::ports::{ArrayPort, SignalPort};
use crate::python::py_operator_module;

/// Gaussian negative log-likelihood of a covariance predictor against realized
/// targets, averaged over the evaluation period.
///
/// See [module-level docs](super::super) for inputs and outputs. This metric
/// takes `[N, N]` covariance matrix prediction and `[N]` realized target
/// vector, and outputs scalar log-likelihoods.
///
/// The output is `log |Σ| + (1/T) Σ_t rᵀ Σ⁺ r`, the multivariate-normal
/// log-density with its `N log(2π)` constant and `1/2` prefactor dropped, so
/// lower is better.
///
/// `Σ⁺` is the pseudo-inverse, which keeps a rank-deficient prediction
/// usable by restricting the likelihood to its PSD subspace; stocks whose
/// predicted variance is non-finite are excluded and contribute nothing.
pub fn log_likelihood() -> impl Operator<
    Inputs = (
        SignalPort<0>,
        ArrayPort<f64, 2>,
        SignalPort<0>,
        ArrayPort<f64, 1>,
    ),
    Outputs = (SignalPort<0>, ArrayPort<f64, 0>),
    Context = Instant,
> {
    py_operator_module("tradingflow.metric.predictor.variance.log_likelihood", None)
}
