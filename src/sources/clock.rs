//! Clock source — emits `()` events at supplied timestamps.
//!
//! Calendar-aligned schedules (daily / monthly in a given timezone) are
//! generated on the Python side via `zoneinfo` and passed to [`clock`] as a
//! pre-computed list.  Keeping calendar/timezone logic in Python lets the
//! Rust core stay free of `chrono` / `chrono-tz`.
//!
//! A clock source can be used two ways:
//!
//! 1. As the sole trigger of a time-series-semantics operator, via the
//!    `trigger` argument of
//!    [`Scenario::add_operator`](crate::Scenario::add_operator).
//! 2. To drive a regular Array-valued input on a message-passing operator
//!    (e.g. a predictor that distinguishes rebalance ticks from data
//!    ticks), route the clock through a trivial
//!    [`Const`](crate::operators::Const) operator clocked by it; the
//!    `Const`'s output is an `Array<f64>` handle that can be wired as a
//!    normal input.

use crate::time::Instant;

use super::iter_source::IterSource;

/// Create a clock source from explicit timestamps.
///
/// The output node holds `()` (zero-sized, purely a trigger).
pub fn clock(timestamps: Vec<Instant>) -> IterSource<()> {
    let count = timestamps.len();
    IterSource::new(timestamps.into_iter().map(|ts| (ts, ())), ()).with_estimated_count(count)
}
