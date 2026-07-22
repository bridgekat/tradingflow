//! Rolling (windowed) operators, implemented directly on
//! [`Operator`](crate::graph::typed::Operator). The [`Accumulator`] /
//! [`Window`] / [`Rolling`] framework and the four accumulators + [`Ema`] keep
//! the output buffer in the operator state (sized on the `init` build call);
//! the `Accumulator: Send + Sync` bound holds because the accumulator lives in
//! the operator State, which is a `Send + Sync` cell.
//!
//! Note: rolling reads event time from the series window (`timestamp`), NOT
//! the threaded `Instant`, so the time-delta window needs no clock wiring. A
//! [`SeriesView`](crate::data::SeriesView) window is view-local, so all
//! addressing is relative to the window's newest row (the accumulated span is
//! always the window's tail); a retention-bounded record works as long as the
//! bound covers the rolling window — the rows the accumulator still holds are
//! never trimmed away.

mod accumulator;
mod covariance;
mod ema;
mod mean;
mod operator;
mod sum;
mod variance;
mod window;

pub use accumulator::*;
pub use covariance::*;
pub use ema::*;
pub use mean::*;
pub use operator::*;
pub use sum::*;
pub use variance::*;
pub use window::*;
