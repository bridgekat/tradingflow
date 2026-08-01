//! Portfolios built from a covariance matrix alone, with no view on return.
//!
//! Only [`minimum_variance`] lives here. The premise is that expected returns
//! are far harder to predict than covariances, so a book that optimizes risk
//! and declines to forecast return at all is often the more honest one.
//!
//! See [module-level docs](super) for inputs and outputs.

mod minimum_variance;

pub use minimum_variance::minimum_variance;

use crate::data::Instant;
use crate::ports::{ArrayPort, SignalPort};

/// The wiring every variance portfolio shares:
/// `(rebalance_signal, universe, covariance)`.
pub type Inputs = (SignalPort<0>, ArrayPort<f64, 1>, ArrayPort<f64, 2>);

/// A rebalance pulse and the `[N]` book.
pub type Outputs = (SignalPort<0>, ArrayPort<f64, 1>);

/// What every constructor in this module returns.
pub trait VariancePortfolio:
    crate::graph::Segment<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}

impl<S> VariancePortfolio for S where
    S: crate::graph::Segment<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}
