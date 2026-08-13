//! Characteristics of portfolio weight arrays.
//!
//! # Inputs
//!
//! - `signal`: a scalar signal indicating a single period of interest.
//! - `weights`: a one-dimensional array of portfolio weights, all finite.
//!
//! # Outputs
//!
//! - `metric`: the performance metric computed from the input weights,
//!   updated each period at `signal == true`.

mod breadth;
mod exposure;
mod period_return;
mod turnover;

pub use breadth::breadth;
pub use exposure::{gross_exposure, long_exposure, net_exposure, short_exposure};
pub use period_return::period_return;
pub use turnover::turnover;
