//! Pulse source — emits bare `()` triggers at supplied timestamps.
//!
//! The output node holds `()` (zero-sized, purely a trigger signal); pulse
//! handles carry no data. Clock-gated operators ([`Clocked`],
//! [`ResampleClocked`], and the since-inception metrics) take such a pulse as
//! their leading input port and fire only on its notify bit.
//!
//! Not to be confused with [`WallClock`](crate::WallClock), which drives the
//! event loop, or with the ambient event time (the graph-level context every
//! operator's `compute` is handed).
//!
//! Calendar-aligned schedules (daily / monthly in a given timezone) are
//! generated on the Python side via `zoneinfo` and passed to [`pulse`] as a
//! pre-computed list.  Keeping calendar/timezone logic in Python lets the
//! Rust core stay free of `chrono` / `chrono-tz`.
//!
//! [`Clocked`]: crate::operators::Clocked
//! [`ResampleClocked`]: crate::operators::ResampleClocked

use crate::data::Instant;

use super::iter_source::IterSource;

/// Create a pulse source from explicit timestamps.
///
/// The output node holds `()` (zero-sized, purely a trigger).  The
/// resulting [`IterSource`] is `Clone` and reusable across multiple
/// scenario sessions.
pub fn pulse(timestamps: Vec<Instant>) -> IterSource<()> {
    IterSource::from_vec_with_default(timestamps.into_iter().map(|ts| (ts, ())).collect(), ())
}
