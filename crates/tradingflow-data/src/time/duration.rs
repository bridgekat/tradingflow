use std::fmt;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// An elapsed interval in nanoseconds, stored as an [`i64`] with
/// `repr(transparent)`.
///
/// This type is considered "naive" in that it does not assume a particular
/// time scale: whether it stands for SI or UTC nanoseconds is up to the user.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(i64);

impl Duration {
    /// Zero duration.
    pub const ZERO: Duration = Duration(0);

    /// Minimum representable duration.
    pub const MIN: Duration = Duration(i64::MIN);

    /// Maximum representable duration.
    pub const MAX: Duration = Duration(i64::MAX);

    pub const fn from_nanos(ns: i64) -> Self {
        Self(ns)
    }

    pub const fn from_micros(us: i64) -> Self {
        Self(us * 1_000)
    }

    pub const fn from_millis(ms: i64) -> Self {
        Self(ms * 1_000_000)
    }

    pub const fn from_seconds(s: i64) -> Self {
        Self(s * 1_000_000_000)
    }

    pub const fn from_minutes(m: i64) -> Self {
        Self(m * 60 * 1_000_000_000)
    }

    pub const fn from_hours(h: i64) -> Self {
        Self(h * 3600 * 1_000_000_000)
    }

    pub const fn from_days(d: i64) -> Self {
        Self(d * 86_400 * 1_000_000_000)
    }

    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    pub const fn as_micros(self) -> i64 {
        self.0 / 1_000
    }

    pub const fn as_millis(self) -> i64 {
        self.0 / 1_000_000
    }

    pub const fn as_seconds(self) -> i64 {
        self.0 / 1_000_000_000
    }

    pub const fn as_minutes(self) -> i64 {
        self.0 / (60 * 1_000_000_000)
    }

    pub const fn as_hours(self) -> i64 {
        self.0 / (3600 * 1_000_000_000)
    }

    pub const fn as_days(self) -> i64 {
        self.0 / (86_400 * 1_000_000_000)
    }

    /// Reinterprets a slice of `i64` nanoseconds as a slice of [`Duration`].
    pub fn from_nanos_slice(ns: &[i64]) -> &[Self] {
        // SAFETY: `Self` is `#[repr(transparent)]` over `i64`.
        unsafe { std::slice::from_raw_parts(ns.as_ptr() as *const Self, ns.len()) }
    }

    /// Reinterprets a slice of [`Duration`] as a slice of `i64` nanoseconds.
    pub fn as_nanos_slice(ts: &[Self]) -> &[i64] {
        // SAFETY: `Self` is `#[repr(transparent)]` over `i64`.
        unsafe { std::slice::from_raw_parts(ts.as_ptr() as *const i64, ts.len()) }
    }
}

impl Neg for Duration {
    type Output = Duration;

    fn neg(self) -> Duration {
        Duration(-self.0)
    }
}

impl Add<Duration> for Duration {
    type Output = Duration;

    fn add(self, rhs: Duration) -> Duration {
        Duration(self.0 + rhs.0)
    }
}

impl AddAssign<Duration> for Duration {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.0;
    }
}

impl Sub<Duration> for Duration {
    type Output = Duration;

    fn sub(self, rhs: Duration) -> Duration {
        Duration(self.0 - rhs.0)
    }
}

impl SubAssign<Duration> for Duration {
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs.0;
    }
}

impl Mul<i64> for Duration {
    type Output = Duration;

    fn mul(self, rhs: i64) -> Duration {
        Duration(self.0 * rhs)
    }
}

impl Mul<Duration> for i64 {
    type Output = Duration;

    fn mul(self, rhs: Duration) -> Duration {
        Duration(self * rhs.0)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Duration({}ns)", self.0)
    }
}
