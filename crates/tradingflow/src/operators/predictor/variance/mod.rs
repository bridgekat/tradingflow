//! Cross-sectional variance (covariance matrix) predictors.
//!
//! Every predictor here emits an `[N, N]` covariance of the target, and every
//! one ignores the feature panel — they estimate second moments from the
//! target cross-sections alone. They exist because [`sample`] covariance is
//! badly conditioned whenever the window is not much longer than the
//! cross-section is wide, which for a realistic universe it never is: with
//! `N` stocks and `T` periods it has at most `T` nonzero eigenvalues, and the
//! smallest ones are noise that a portfolio optimizer will happily lever into.
//!
//! The rest impose structure to fix that. [`single_index`] reduces the whole
//! matrix to one factor plus idiosyncratic variance; [`shrinkage`] pulls the
//! sample estimate toward such a structured target by an analytically optimal
//! amount; [`rmt`] keeps only the eigenvalues that random matrix theory says
//! are not noise; and [`hierarchical`] replaces the correlation with the
//! cophenetic similarity of a dendrogram fit to it.
//!
//! See [module-level docs](super) for inputs, outputs and windowing.

mod hierarchical;
mod rmt;
mod sample;
mod shrinkage;
mod single_index;

pub use hierarchical::{Linkage, hierarchical};
pub use rmt::{Replacement, rmt};
pub use sample::sample;
pub use shrinkage::{Target, shrinkage};
pub use single_index::single_index;

use crate::data::Instant;
use crate::ports::{ArrayPort, SignalPort};

/// The wiring every variance predictor shares:
/// `(sample_signal, features, target, rebalance_signal, universe)`.
pub type Inputs = (
    SignalPort<0>,
    ArrayPort<f64, 2>,
    ArrayPort<f64, 1>,
    SignalPort<0>,
    ArrayPort<f64, 1>,
);

/// A rebalance pulse and the `[N, N]` predicted covariance.
pub type Outputs = (SignalPort<0>, ArrayPort<f64, 2>);

/// What every constructor in this module returns.
pub trait VariancePredictor:
    crate::graph::Segment<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}

impl<S> VariancePredictor for S where
    S: crate::graph::Segment<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}
