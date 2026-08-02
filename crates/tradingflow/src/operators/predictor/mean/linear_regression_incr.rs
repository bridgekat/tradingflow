use super::MeanPredictor;
use crate::operators::predictor::Config;
use crate::python::py_operator_module;

/// Pooled ordinary least squares fit from an incrementally maintained Gram.
///
/// Fits the same model as [`linear_regression`](super::linear_regression) —
/// pooled, pool-standardized OLS — but reaches it from sufficient statistics
/// folded one sampling period at a time. Each refit then costs `O(F³)`
/// regardless of how far back history reaches, and the node retains only
/// `target_offset + 1` cross-sections rather than the whole window.
///
/// [`Config::max_periods`] still bounds the training set, by down-dating pairs
/// out of the pool as they age rather than by slicing a window; `None` fits on
/// an expanding window over everything seen, which here costs no extra memory.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn linear_regression_incr(config: Config) -> impl MeanPredictor {
    py_operator_module(
        "tradingflow.predictor.mean.linear_regression_incr",
        config.params(|_| Ok(())),
    )
}
