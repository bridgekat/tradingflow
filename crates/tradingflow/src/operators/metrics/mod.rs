//! Clock-driven since-inception financial metrics over the
//! [`ArrayView`](crate::data::ArrayView) currency. The first four take
//! `(clock, data)` and gate on the clock's notify bit (emitting `notify = false`
//! off-tick); `Drawdown`/`Turnover` are single-input. The data input is a
//! rank-`N` view (the leading element is read for the scalar metrics;
//! `Turnover` reads the whole weight vector); every output is a rank-0 scalar
//! view.
//!
//! The clock is the **leading** port, matching every other clock-gated operator
//! in the library ([`Clocked`](super::structural::Clocked),
//! [`ResampleClocked`](super::structural::ResampleClocked)), so the gated shapes stay
//! interchangeable.

mod average_return;
mod common;
mod compound_return;
mod drawdown;
mod sharpe_ratio;
mod turnover;
mod volatility;

pub use average_return::*;
pub use compound_return::*;
pub use drawdown::*;
pub use sharpe_ratio::*;
pub use turnover::*;
pub use volatility::*;
