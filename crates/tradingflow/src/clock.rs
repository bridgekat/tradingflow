use std::time::{SystemTime, UNIX_EPOCH};

use crate::data::{Duration, Instant};
use crate::graph::Clock;

/// The system wall clock driving the event loop: `now()` reads system time as
/// UNIX nanoseconds (fixing the naive [`Instant`] epoch at 1970-01-01), and
/// `wait_until` sleeps on the tokio timer until the clock has moved strictly
/// past the target.
///
/// Under the ingest merge rules this behaves uniformly for backtests and live
/// runs: historical timestamps sit below `now` and replay at full speed,
/// future-dated events are released only once their timestamp actually
/// arrives, and implicit `Stamp::Now` events are stamped with the current
/// wall-clock reading.
pub struct WallClock;

impl Clock<Instant> for WallClock {
    fn min(&self) -> Instant {
        Instant::MIN
    }

    fn now(&self) -> Instant {
        let unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before the Unix epoch");
        Instant::from_offset(Duration::from_nanos(unix.as_nanos() as i64))
    }

    async fn wait_until(&mut self, t: Instant) {
        // Sleep to just past the target, re-checking after each wake: the
        // contract is strict (`now() > t` on return), and the timer may round
        // or wake early.
        loop {
            let now = self.now();
            if now > t {
                return;
            }
            let ns = (t - now).as_nanos().max(0) as u64 + 1;
            tokio::time::sleep(std::time::Duration::from_nanos(ns)).await;
        }
    }
}
