//! Quality metrics for cross-sectional feature arrays.
//!
//! # Inputs
//!
//! - `signal`: a scalar signal indicating a single period of interest.
//! - `features`: a one-dimensional array of cross-sectional feature values.
//! - `target`: a one-dimensional array of cross-sectional target values.
//!
//! # Outputs
//!
//! - `metric`: the performance metric computed from the input features
//!   and target, updated each period at `signal == true`.

mod information_coefficient;

pub use information_coefficient::information_coefficient;
