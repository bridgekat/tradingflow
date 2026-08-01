use pyo3::types::PyDictMethods;

use super::MeanPredictor;
use crate::operators::predictor::Config;
use crate::python::py_segment_module;

/// Pooled L1-penalized regression on the feature panel, refit from the whole
/// window each time.
///
/// Solves `min (1/m)‖y - X̃β - ȳ‖² + α‖β‖₁`. Standardization and the
/// dimensionless `alpha` match [`ridge`](super::ridge); the difference is that
/// the L1 penalty drives coefficients exactly to zero, so this both regularizes
/// and *selects* — useful when most of a wide factor panel is expected to be
/// noise. The price is that L1 has no closed form, so the fit goes through a
/// CVXPY/SCS solve rather than a QR, and each refit is much more expensive.
///
/// `max_samples` and `subsample_seed` cap and seed the subsample of pooled
/// rows, and matter more here than for the closed-form predictors.
///
/// Requires CVXPY in the embedded interpreter; the import is deferred to the
/// first fit, so a graph that never rebalances will not notice its absence.
///
/// See [module-level docs](super::super) for inputs and outputs.
///
/// # Panics
///
/// If `alpha` is negative.
pub fn lasso(
    config: Config,
    alpha: f64,
    max_samples: Option<usize>,
    subsample_seed: u64,
) -> impl MeanPredictor {
    assert!(
        alpha >= 0.0,
        "lasso alpha must be non-negative, got {alpha}"
    );
    py_segment_module(
        "tradingflow.predictor.mean.lasso",
        config.params(|d| {
            d.set_item("alpha", alpha)?;
            d.set_item("max_samples", max_samples)?;
            d.set_item("subsample_seed", subsample_seed)
        }),
    )
}
