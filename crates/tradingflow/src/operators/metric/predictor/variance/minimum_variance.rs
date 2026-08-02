use crate::data::Instant;
use crate::graph::Operator;
use crate::ports::{ArrayPort, SignalPort};
use crate::python::py_operator_module;

/// Realized variance of the global minimum variance portfolio implied by a
/// covariance predictor, over the evaluation period.
///
/// See [module-level docs](super::super) for inputs and outputs. This metric
/// takes `[N, N]` covariance matrix prediction and `[N]` realized target
/// vector, and outputs scalar GMV variances.
///
/// Each input prediction is solved for GMV weights `w = Σ⁺1 / (1ᵀΣ⁺1)`,
/// which then accumulate one portfolio return per sampling period; the period
/// closes on the realized variance of those returns. Lower is better, and
/// unlike [`log_likelihood`](super::log_likelihood) this scores a prediction
/// by the portfolio it produces rather than by its density.
///
/// Both the prediction and the target are in **log-return** units: the
/// prediction is converted to linear-return covariance before solving GMV,
/// and the target vector is mapped elementwise, so the reported variance is
/// in the units the GMV objective actually minimizes. Stocks whose predicted
/// variance is non-finite are excluded and hold a zero weight.
pub fn minimum_variance() -> impl Operator<
    Inputs = (
        SignalPort<0>,
        ArrayPort<f64, 2>,
        SignalPort<0>,
        ArrayPort<f64, 1>,
    ),
    Outputs = (SignalPort<0>, ArrayPort<f64, 0>),
    Context = Instant,
> {
    py_operator_module(
        "tradingflow.metric.predictor.variance.minimum_variance",
        None,
    )
}
