use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use super::Duration;

/// A point in time, defined as the number of nanoseconds since a globally
/// fixed epoch. Stored as an [`i64`] with `repr(transparent)`.
///
/// This type is considered "naive" in that it does not assume a particular
/// time scale: whether it stands for TAI or UTC timestamp is up to the user.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant(i64);

impl Instant {
    /// The epoch has zero offset.
    pub const EPOCH: Instant = Instant(0);

    /// Minimum representable timestamp.
    pub const MIN: Instant = Instant(i64::MIN);

    /// Maximum representable timestamp.
    pub const MAX: Instant = Instant(i64::MAX);

    pub const fn from_offset(offset: Duration) -> Self {
        Self(offset.as_nanos())
    }

    pub const fn as_offset(&self) -> Duration {
        Duration::from_nanos(self.0)
    }

    /// Reinterprets a slice of `i64` nanoseconds as a slice of [`Instant`].
    pub fn from_nanos_slice(ns: &[i64]) -> &[Self] {
        // SAFETY: `Self` is `#[repr(transparent)]` over `i64`.
        unsafe { std::slice::from_raw_parts(ns.as_ptr() as *const Self, ns.len()) }
    }

    /// Reinterprets a slice of [`Instant`] as a slice of `i64` nanoseconds.
    pub fn as_nanos_slice(ts: &[Self]) -> &[i64] {
        // SAFETY: `Self` is `#[repr(transparent)]` over `i64`.
        unsafe { std::slice::from_raw_parts(ts.as_ptr() as *const i64, ts.len()) }
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;

    fn add(self, rhs: Duration) -> Instant {
        Instant(self.0 + rhs.as_nanos())
    }
}

impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.as_nanos();
    }
}

impl Sub<Duration> for Instant {
    type Output = Instant;

    fn sub(self, rhs: Duration) -> Instant {
        Instant(self.0 - rhs.as_nanos())
    }
}

impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs.as_nanos();
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;

    fn sub(self, rhs: Instant) -> Duration {
        Duration::from_nanos(self.0 - rhs.0)
    }
}

impl fmt::Display for Instant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instant({}ns)", self.0)
    }
}
