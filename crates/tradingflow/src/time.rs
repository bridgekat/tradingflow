//! Wall clock implementations.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::data::{Duration, Instant};
use crate::graph::Time;

/// A standard wall clock implementation which assumes and outputs [`Instant`]
/// as UNIX nanoseconds since `1970-01-01 00:00:00` UTC.
///
/// This clock is not leap-aware: during UTC leap seconds, a single "UNIX
/// second" spans two SI seconds, by definition.
pub struct UnixTime;

impl Time<Instant> for UnixTime {
    fn now(&self) -> Instant {
        let unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before the UNIX epoch");
        Instant::from_offset(Duration::from_nanos(unix.as_nanos() as i64))
    }

    async fn wait_until(&mut self, t: Instant) {
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
