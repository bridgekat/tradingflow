//! Wall-clock time sources gating the ingest event loop.

use std::future::Future;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::data::Instant;

/// A wall clock source: gates the release of future-dated and implicitly
/// (["now"](super::Stamp::Now)) stamped events in the ingest merge.
///
/// The [`Queue`](super::Queue) is generic over this so tests can substitute a
/// simulated clock whose `wait_until` jumps time forward deterministically;
/// everything else runs on the real [`WallClock`].
pub trait Clock {
    /// Returns current wall-clock reading. Must be non-decreasing.
    fn now(&self) -> Instant;

    /// Returns a future that resolves once the wall clock has moved strictly
    /// past `t` (so `now() > t` afterwards).
    fn wait_until(&mut self, t: Instant) -> impl Future<Output = ()>;
}

/// Current TAI time from the system clock (UTC → TAI through [`hifitime`]'s
/// leap-second table, see [`Instant::from_utc_nanos`]).
fn tai_now() -> Instant {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch");
    Instant::from_utc_nanos(unix.as_nanos() as i64)
}

/// The real (TAI) wall clock driving the event loop: `now()` reads system
/// time converted to TAI, and `wait_until` sleeps on the tokio timer until
/// the clock has moved strictly past the target.
///
/// Under the ingest merge rules this behaves uniformly for backtests and live
/// runs: historical timestamps sit below `now` and replay at full speed,
/// future-dated events are released only once their timestamp actually
/// arrives, and implicit `Stamp::Now` events are stamped with real TAI time.
///
/// Pass it to `Scenario::new(WallClock)` to drive a session on
/// real time — it is the default [`Scenario`](super::Scenario) clock.
pub struct WallClock;

impl Clock for WallClock {
    fn now(&self) -> Instant {
        tai_now()
    }

    async fn wait_until(&mut self, t: Instant) {
        // Sleep to just past the target, re-checking after each wake: the
        // contract is strict (`now() > t` on return), and the timer may round
        // or wake early.
        loop {
            let now = tai_now();
            if now > t {
                return;
            }
            let ns = (t.as_nanos() - now.as_nanos()).max(0) as u64 + 1;
            tokio::time::sleep(std::time::Duration::from_nanos(ns)).await;
        }
    }
}
