use pyo3::types::PyDictMethods;

use super::MeanPortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_segment_module;

/// Weights each stock by the softmax of its predicted return.
///
/// `temperature` sets how sharply conviction becomes weight: near zero the
/// book collapses onto the single best-predicted stock, and as it grows the
/// weights flatten toward equal. Unlike [`proportional`](super::proportional)
/// every eligible stock keeps a positive weight, so the book stays diversified
/// even when one prediction runs away from the rest — the exponential
/// concentrates smoothly rather than winner-take-all.
///
/// See [module-level docs](super::super) for inputs and outputs.
///
/// # Panics
///
/// If `temperature` is not positive.
pub fn softmax(config: Config, temperature: f64) -> impl MeanPortfolio {
    assert!(
        temperature > 0.0,
        "softmax temperature must be positive, got {temperature}"
    );
    py_segment_module(
        "tradingflow.portfolio.mean.softmax",
        config.params(|d| d.set_item("temperature", temperature)),
    )
}
