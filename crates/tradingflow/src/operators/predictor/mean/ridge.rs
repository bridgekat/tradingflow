use pyo3::types::PyDictMethods;

use super::MeanPredictor;
use crate::operators::predictor::Config;
use crate::python::py_operator_module;

/// Pooled L2-penalized regression on the feature panel, refit from the whole
/// window each time.
///
/// Solves `min (1/m)‖y - X̃β - ȳ‖² + α‖β‖²` by augmented QR. The `1/m`
/// prefactor together with pool-standardization makes `alpha` dimensionless:
/// the same value means the same amount of regularization at any sample size
/// and any target scale, so it transfers between panels. Unlike
/// [`linear_regression`](super::linear_regression) the penalty makes the
/// system solvable even when the design is rank-deficient, which is why a
/// wide, collinear factor panel usually wants this one.
///
/// `max_samples` and `subsample_seed` cap and seed the subsample of pooled
/// rows fed to the factorization, as in
/// [`linear_regression`](super::linear_regression).
///
/// See [module-level docs](super::super) for inputs and outputs.
///
/// # Panics
///
/// If `alpha` is negative.
pub fn ridge(
    config: Config,
    alpha: f64,
    max_samples: Option<usize>,
    subsample_seed: u64,
) -> impl MeanPredictor {
    assert!(
        alpha >= 0.0,
        "ridge alpha must be non-negative, got {alpha}"
    );
    py_operator_module(
        "tradingflow.predictor.mean.ridge",
        config.params(|d| {
            d.set_item("alpha", alpha)?;
            d.set_item("max_samples", max_samples)?;
            d.set_item("subsample_seed", subsample_seed)
        }),
    )
}
