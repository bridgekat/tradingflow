//! Quality of cross-sectional predictions against realized targets.
//!
//! # Inputs
//!
//! - `predict_signal`: a scalar signal indicating an evaluation period.
//! - `predict`: an array of cross-sectional predictions for that period.
//! - `target_signal`: a scalar signal indicating a sampling period.
//! - `target`: an array of realized targets over that period.
//!
//! # Outputs
//!
//! - `metric`: the performance metric computed from the input predictions
//!   and targets, updated at the end of each evaluation period.
//!
//! When `predict_signal == true`, it must be the case that
//! `target_signal == true`. The current `target` is then counted into the
//! *previous* evaluation period, and outputs are recomputed for the
//! previous evaluation period. The next `target` will be the first sample in
//! the current evaluation period.
//!
//! Each evaluation period can span one or more multiple sampling periods.
//! Feature ICs are typically calculated each trading day, using the
//! predictions from yesterday and the realized targets from today,
//! in which case both periods equal one trading day. However, GMV realized
//! variances must be calculated using multiple samples per GMV, in which case
//! the sampling period is still one trading day, but the evaluation period is
//! much less frequent, e.g. one month.

pub mod mean;
pub mod variance;
