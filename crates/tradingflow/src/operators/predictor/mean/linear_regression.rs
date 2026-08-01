use pyo3::types::PyDictMethods;

use super::MeanPredictor;
use crate::operators::predictor::Config;
use crate::python::py_segment_module;

/// Pooled ordinary least squares on the feature panel, refit from the whole
/// window each time.
///
/// Features and target are pool-standardized before the QR solve, so the fit
/// is invariant to the scale of either. A rank-deficient design — collinear
/// features, or fewer samples than features — has no unique solution, and
/// rather than silently take the minimum-norm one the fit falls back to zero
/// coefficients, predicting the target mean for every stock. Reach for
/// [`ridge`](super::ridge) when that happens rather than dropping features.
///
/// `max_samples` caps the pooled rows fed to the factorization by taking a
/// uniform random subset, keeping the fit unbiased while bounding the `O(m F²)`
/// cost; `None` uses every valid row. `subsample_seed` makes that draw
/// reproducible.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn linear_regression(
    config: Config,
    max_samples: Option<usize>,
    subsample_seed: u64,
) -> impl MeanPredictor {
    py_segment_module(
        "tradingflow.predictor.mean.linear_regression",
        config.params(|d| {
            d.set_item("max_samples", max_samples)?;
            d.set_item("subsample_seed", subsample_seed)
        }),
    )
}
