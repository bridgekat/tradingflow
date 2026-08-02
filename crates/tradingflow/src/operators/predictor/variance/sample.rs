use super::VariancePredictor;
use crate::operators::predictor::Config;
use crate::python::py_operator_module;

/// Sample covariance of the target over the window, computed pairwise so a
/// stock missing on some periods still contributes on the rest.
///
/// The *Markowitz* direct estimator of Pantaleo et al. (2010), and the
/// baseline the structured estimators are measured against. Be aware of what
/// it does when the window is short relative to the cross-section: the matrix
/// is singular past rank `T`, and its smallest eigenvalues are almost entirely
/// estimation noise, which a mean-variance optimizer will read as riskless
/// directions and lever into. Prefer a structured estimator unless the window
/// is genuinely long.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn sample(config: Config) -> impl VariancePredictor {
    py_operator_module(
        "tradingflow.predictor.variance.sample",
        config.params(|_| Ok(())),
    )
}
