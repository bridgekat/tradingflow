use super::MeanPredictor;
use crate::operators::predictor::Config;
use crate::python::py_segment_module;

/// Predicts each stock's return as its own sample mean over the window,
/// ignoring features entirely.
///
/// The baseline every feature-driven predictor has to beat: it captures
/// whatever persistent cross-sectional drift exists and nothing else, so a
/// model that cannot outperform it is not extracting signal from its features.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn sample(config: Config) -> impl MeanPredictor {
    py_segment_module(
        "tradingflow.predictor.mean.sample",
        config.params(|_| Ok(())),
    )
}
