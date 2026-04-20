//! Clock source — emits `()` events at supplied timestamps.
//!
//! The output node holds `()` (zero-sized, purely a trigger signal).
//! On the Python side this corresponds to [`NodeKind.UNIT`], and clock
//! handles carry no data.
//!
//! Calendar-aligned schedules (daily / monthly in a given timezone) are
//! generated on the Python side via `zoneinfo` and passed to [`clock`] as a
//! pre-computed list.  Keeping calendar/timezone logic in Python lets the
//! Rust core stay free of `chrono` / `chrono-tz`.

use crate::Instant;

use super::iter_source::IterSource;

/// Create a clock source from explicit timestamps.
///
/// The output node holds `()` (zero-sized, purely a trigger).
pub fn clock(timestamps: Vec<Instant>) -> IterSource<()> {
    let count = timestamps.len();
    IterSource::new(timestamps.into_iter().map(|ts| (ts, ())), ()).with_estimated_count(count)
}
