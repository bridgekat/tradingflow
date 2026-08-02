use pyo3::types::PyDictMethods;

use super::MeanPortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_operator_module;

/// Weights the top `top_fraction` of stocks linearly by rank.
///
/// Like [`rank_equal`](super::rank_equal) it reads the predictions only as an
/// ordering, but it tilts toward the top of that ordering rather than
/// spreading evenly: with `k` stocks selected the best-ranked gets `k` units
/// against the last one's `1`. The middle ground between trusting the
/// ordering and trusting the magnitudes.
///
/// See [module-level docs](super::super) for inputs and outputs.
///
/// # Panics
///
/// If `top_fraction` is not in `(0, 1]`.
pub fn rank_linear(config: Config, top_fraction: f64) -> impl MeanPortfolio {
    assert!(
        top_fraction > 0.0 && top_fraction <= 1.0,
        "top_fraction must be in (0, 1], got {top_fraction}"
    );
    py_operator_module(
        "tradingflow.portfolio.mean.rank_linear",
        config.params(|d| d.set_item("top_fraction", top_fraction)),
    )
}
