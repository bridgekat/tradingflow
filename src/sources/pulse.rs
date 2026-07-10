//! Pulse source — emits bare `()` triggers at supplied timestamps.
//!
//! The output node holds `()` (zero-sized, purely a trigger signal); pulse
//! handles carry no data. Clock-gated operators ([`Clocked`], [`Resample`],
//! and the since-inception metrics) take such a pulse as their leading input
//! port and fire only on its notify bit.
//!
//! Not to be confused with [`WallClock`](crate::WallClock), which drives the
//! event loop, or [`EventTime`](crate::operators::EventTime), the out-of-band
//! cell holding the current batch's event time.
//!
//! Calendar-aligned schedules (daily / monthly in a given timezone) are
//! generated on the Python side via `zoneinfo` and passed to [`pulse`] as a
//! pre-computed list.  Keeping calendar/timezone logic in Python lets the
//! Rust core stay free of `chrono` / `chrono-tz`.
//!
//! [`Clocked`]: crate::operators::Clocked
//! [`Resample`]: crate::operators::Resample

use crate::Instant;

use super::iter_source::IterSource;

/// Create a pulse source from explicit timestamps.
///
/// The output node holds `()` (zero-sized, purely a trigger).  The
/// resulting [`IterSource`] is `Clone` and reusable across multiple
/// scenario sessions.
pub fn pulse(timestamps: Vec<Instant>) -> IterSource<()> {
    IterSource::from_vec_with_default(timestamps.into_iter().map(|ts| (ts, ())).collect(), ())
}
