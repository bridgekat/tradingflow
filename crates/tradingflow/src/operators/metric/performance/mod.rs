//! Performance metrics of net-asset-value scalars.
//!
//! # Inputs
//!
//! - `signal`: a scalar signal indicating a single period of interest.
//! - `value`: a scalar net asset value at the end of each period.
//!
//! # Outputs
//!
//! - `metric`: the performance metric computed from the input signal
//!   and value, updated each period at `signal == true`.

mod drawdown;
mod r#return;

pub use drawdown::drawdown;
pub use r#return::{
    comp_return, log_return_mean, log_return_sharpe, log_return_vol, return_mean, return_sharpe,
    return_vol,
};
