use crate::data::Instant;
use crate::graph::Operator;
use crate::ports::{ArrayPort, SignalPort};
use crate::python::py_operator_module;

/// Cross-sectional information coefficient of a predictor against a realized
/// target, averaged over the evaluation period.
///
/// See [module-level docs](super::super) for inputs and outputs. This metric
/// takes `[N]` expected value prediction and `[N]` realized target vector,
/// and outputs the information coefficient (Pearson correlation) between them.
///
/// To obtain rank information coefficient (Spearman correlation) instead,
/// rank both inputs before passing them to this metric.
pub fn information_coefficient() -> impl Operator<
    Inputs = (
        SignalPort<0>,
        ArrayPort<f64, 1>,
        SignalPort<0>,
        ArrayPort<f64, 1>,
    ),
    Outputs = (SignalPort<0>, ArrayPort<f64, 0>),
    Context = Instant,
> {
    py_operator_module(
        "tradingflow.metric.predictor.mean.information_coefficient",
        None,
    )
}
