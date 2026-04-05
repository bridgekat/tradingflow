//! Clock-driven financial metrics operators.
//!
//! Each operator takes a scalar [`Array<T>`](crate::Array) input and
//! produces a scalar [`Array<T>`](crate::Array) output.  Intended to be
//! triggered by a clock source so that each tick represents one period
//! (e.g. monthly).  All metrics are since-inception (not rolling).
//!
//! # Metrics
//!
//! - [`CompoundReturn`] — `(current / first)^(1/n) - 1`
//! - [`AverageReturn`] — mean of period returns
//! - [`Volatility`] — population standard deviation of period returns
//! - [`SharpeRatio`] — mean / std of period returns
//! - [`Drawdown`] — `(current - running_max) / running_max`

pub mod average_return;
pub mod compound_return;
pub mod drawdown;
pub mod sharpe_ratio;
pub mod volatility;

pub use average_return::AverageReturn;
pub use compound_return::CompoundReturn;
pub use drawdown::Drawdown;
pub use sharpe_ratio::SharpeRatio;
pub use volatility::Volatility;
