use pyo3::types::PyDictMethods;

use super::MeanPortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_segment_module;

/// Equally weights the stocks whose percentile rank falls in `[low, high)`.
///
/// The instrument for *measuring* a feature rather than trading a prediction.
/// Run one per decile — `low = d/10`, `high = (d+1)/10` — and the resulting
/// return series say whether the feature separates winners from losers at all,
/// and whether it does so monotonically or only at the extremes. The spread
/// between the top and bottom buckets is the feature's raw return.
///
/// The input is read as a feature rather than a return, so
/// [`Config::logarithmic`] should be `false`: only the ordering matters, and
/// no monotone transform can change it.
///
/// See [module-level docs](super::super) for inputs and outputs.
///
/// # Panics
///
/// If the bucket is not a non-empty sub-range of `[0, 1]`.
pub fn rank_bucket(config: Config, low: f64, high: f64) -> impl MeanPortfolio {
    assert!(
        (0.0..high).contains(&low) && high <= 1.0,
        "expected 0 <= low < high <= 1, got {low} and {high}"
    );
    py_segment_module(
        "tradingflow.portfolio.mean.rank_bucket",
        config.params(|d| {
            d.set_item("low", low)?;
            d.set_item("high", high)
        }),
    )
}
