//! Cross-sectional mean (first-moment) return predictors.
//!
//! Every predictor here emits an `[N]` vector of expected returns. They fall
//! into three groups: [`sample`] ignores features entirely and is the baseline
//! the others must beat; [`linear_regression`], [`ridge`] and [`lasso`] refit
//! a pooled panel regression from the whole window each time; and
//! [`linear_regression_incr`] and [`ridge_incr`] fit the same
//! models from an incrementally maintained Gram, at a per-rebalance cost
//! independent of history length. Prefer the incremental pair unless you need
//! the L1 penalty or a subsample cap.
//!
//! See [module-level docs](super) for inputs, outputs and windowing.

mod lasso;
mod linear_regression;
mod linear_regression_incr;
mod ridge;
mod ridge_incr;
mod sample;

pub use lasso::lasso;
pub use linear_regression::linear_regression;
pub use linear_regression_incr::linear_regression_incr;
pub use ridge::ridge;
pub use ridge_incr::ridge_incr;
pub use sample::sample;

use crate::data::Instant;
use crate::ports::{ArrayPort, SignalPort};

/// The wiring every mean predictor shares:
/// `(sample_signal, features, target, rebalance_signal, universe)`.
pub type Inputs = (
    SignalPort<0>,
    ArrayPort<f64, 2>,
    ArrayPort<f64, 1>,
    SignalPort<0>,
    ArrayPort<f64, 1>,
);

/// A rebalance pulse and the `[N]` vector of predicted returns.
pub type Outputs = (SignalPort<0>, ArrayPort<f64, 1>);

/// What every constructor in this module returns.
pub trait MeanPredictor:
    crate::graph::Segment<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}

impl<S> MeanPredictor for S where
    S: crate::graph::Segment<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}
