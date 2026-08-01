use pyo3::types::PyDictMethods;

use super::MeanPortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_segment_module;

/// Equally weights the top `top_fraction` of positively-predicted stocks.
///
/// Uses the predictions only to rank, then discards their magnitudes
/// entirely. That is deliberate: a predictor usually orders the cross-section
/// far more reliably than it estimates returns, and equal weighting is immune
/// to any single prediction being wildly wrong. The cost is that genuine
/// conviction is thrown away along with the noise — every selected stock gets
/// the same weight whether it was ranked first or last.
///
/// See [module-level docs](super::super) for inputs and outputs.
///
/// # Panics
///
/// If `top_fraction` is not in `(0, 1]`.
pub fn rank_equal(config: Config, top_fraction: f64) -> impl MeanPortfolio {
    assert!(
        top_fraction > 0.0 && top_fraction <= 1.0,
        "top_fraction must be in (0, 1], got {top_fraction}"
    );
    py_segment_module(
        "tradingflow.portfolio.mean.rank_equal",
        config.params(|d| d.set_item("top_fraction", top_fraction)),
    )
}
