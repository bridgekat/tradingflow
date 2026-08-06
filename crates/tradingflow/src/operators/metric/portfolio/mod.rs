//! Characteristics of portfolio weight arrays.
//!
//! # Inputs
//!
//! - `signal`: a scalar signal indicating a single period of interest.
//! - `weights`: a one-dimensional array of portfolio weights.
//!
//! # Outputs
//!
//! - `metric`: the performance metric computed from the input weights,
//!   updated each period at `signal == true`.

mod turnover;

pub use turnover::turnover;
