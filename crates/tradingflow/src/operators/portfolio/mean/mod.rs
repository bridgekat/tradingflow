//! Portfolios built from predicted returns alone, with no risk model.
//!
//! Cheap, closed-form and solver-free. They divide by what they trust about
//! the prediction: [`proportional`] and [`softmax`] trust the magnitudes,
//! while [`rank_equal`], [`rank_linear`] and [`rank_bucket`] use them only to
//! order the cross-section. Ranking is the safer default — a predictor
//! typically orders stocks far better than it estimates how much each will
//! return, and a single wild prediction cannot dominate a book that only
//! looked at its position in the sort.
//!
//! [`rank_bucket`] is the odd one out: it exists to *measure* a feature rather
//! than to trade a prediction.
//!
//! See [module-level docs](super) for inputs and outputs.

mod proportional;
mod rank_bucket;
mod rank_equal;
mod rank_linear;
mod softmax;

pub use proportional::proportional;
pub use rank_bucket::rank_bucket;
pub use rank_equal::rank_equal;
pub use rank_linear::rank_linear;
pub use softmax::softmax;

use crate::data::Instant;
use crate::ports::{ArrayPort, SignalPort};

/// The wiring every mean portfolio shares:
/// `(rebalance_signal, universe, predicted_returns)`.
pub type Inputs = (SignalPort<0>, ArrayPort<f64, 1>, ArrayPort<f64, 1>);

/// A rebalance pulse and the `[N]` book.
pub type Outputs = (SignalPort<0>, ArrayPort<f64, 1>);

/// What every constructor in this module returns.
pub trait MeanPortfolio:
    crate::graph::Operator<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}

impl<S> MeanPortfolio for S where
    S: crate::graph::Operator<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}
