use pyo3::types::PyDictMethods;

use super::MeanPredictor;
use crate::operators::predictor::Config;
use crate::python::py_segment_module;

/// Pooled L2-penalized regression fit from an incrementally maintained Gram.
///
/// Fits the same model as [`ridge`](super::ridge) — the dimensionless `alpha`
/// means exactly what it does there — but from sufficient statistics folded
/// one sampling period at a time. Each refit costs `O(F³)` regardless of
/// history length, and the node retains only `target_offset + 1`
/// cross-sections. This is usually the right default for a wide factor panel.
///
/// [`Config::max_periods`] still bounds the training set, by down-dating pairs
/// out of the pool as they age; `None` fits on an expanding window, which here
/// costs no extra memory.
///
/// See [module-level docs](super::super) for inputs and outputs.
///
/// # Panics
///
/// If `alpha` is negative.
pub fn ridge_incr(config: Config, alpha: f64) -> impl MeanPredictor {
    assert!(
        alpha >= 0.0,
        "ridge alpha must be non-negative, got {alpha}"
    );
    py_segment_module(
        "tradingflow.predictor.mean.ridge_incr",
        config.params(|d| d.set_item("alpha", alpha)),
    )
}
