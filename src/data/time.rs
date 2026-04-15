//! SI-second instants and durations.
//!
//! [`Instant`] represents a point in time as an SI-second nanosecond count
//! since the PTP epoch **1970-01-01 00:00:00 TAI** (IEEE 1588).  [`Duration`]
//! represents an elapsed interval in SI nanoseconds.
//!
//! Both types are `#[repr(transparent)]` newtypes around `i64`, so slices of
//! `Instant` and slices of `i64` share identical layout.  Arithmetic follows
//! the usual rules: `Instant − Instant = Duration`, `Instant ± Duration =
//! Instant`, `Duration ± Duration = Duration`, `Duration × i64 = Duration`.
//! Adding two `Instant`s is not defined.
//!
//! # Timeline convention
//!
//! This crate — and the Python wrapper above it — uses TAI throughout.  An
//! `Instant`'s numerical value is SI nanoseconds since 1970-01-01 00:00:00
//! TAI.  This matches NumPy / pandas naïve `datetime64[ns]` arithmetic:
//! every calendar day is exactly 86 400 SI seconds and subtraction yields
//! true elapsed SI time, both inside Rust and across the Python boundary.
//! No conversion happens at the FFI edge.
//!
//! A parsed `"2024-01-01"` therefore labels the instant 2024-01-01 00:00:00
//! TAI — which is 2023-12-31 23:59:23 UTC, 37 SI seconds earlier than the
//! same string would mean under a UTC interpretation.  For almost any
//! backtest this offset is absorbed uniformly and invisible.  When data
//! must be anchored to a wall-clock UTC reference (e.g. external systems,
//! sub-second plots), use the UTC ↔ TAI helpers:
//!
//! * [`Instant::from_utc_nanos`] / [`Instant::to_utc_nanos`] — method form.
//! * [`utc_to_tai`] / [`tai_to_utc`] — free-function form (exposed to Python).
//!
//! Both delegate to [`hifitime`]'s IERS leap-second table.  Pre-1972 dates
//! receive no offset (IERS Bulletin C starts in 1972); for any modern date
//! the offset is the current TAI−UTC value (37 s as of 2025).

use std::fmt;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use std::sync::LazyLock;

use hifitime::{Duration as HfDuration, Epoch, TimeScale, UNIX_REF_EPOCH};

// ===========================================================================
// Instant
// ===========================================================================

/// A point in time as SI nanoseconds since the PTP epoch
/// (1970-01-01 00:00:00 TAI).
///
/// `repr(transparent)` over `i64`.  Totally ordered.  Subtraction yields
/// [`Duration`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant(i64);

impl Instant {
    /// The PTP epoch (1970-01-01 00:00:00 TAI).  Numeric value: zero.
    pub const EPOCH: Instant = Instant(0);

    /// Minimum representable instant.
    pub const MIN: Instant = Instant(i64::MIN);

    /// Maximum representable instant.
    pub const MAX: Instant = Instant(i64::MAX);

    /// Create from SI nanoseconds since the PTP epoch.
    #[inline(always)]
    pub const fn from_nanos(ns: i64) -> Self {
        Self(ns)
    }

    /// Return SI nanoseconds since the PTP epoch.
    #[inline(always)]
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    /// Reinterpret a slice of `i64` nanoseconds as a slice of `Instant`.
    ///
    /// Zero-cost by `#[repr(transparent)]`.
    #[inline(always)]
    pub fn from_nanos_slice(ns: &[i64]) -> &[Instant] {
        // SAFETY: `Instant` is `#[repr(transparent)]` over `i64`.
        unsafe { std::slice::from_raw_parts(ns.as_ptr() as *const Instant, ns.len()) }
    }

    /// Reinterpret a slice of `Instant` as a slice of `i64` nanoseconds.
    ///
    /// Zero-cost by `#[repr(transparent)]`.
    #[inline(always)]
    pub fn as_nanos_slice(ts: &[Instant]) -> &[i64] {
        // SAFETY: `Instant` is `#[repr(transparent)]` over `i64`.
        unsafe { std::slice::from_raw_parts(ts.as_ptr() as *const i64, ts.len()) }
    }

    /// Convert from a [`hifitime::Epoch`].  Useful for date-string
    /// parsing without going through UNIX nanoseconds.
    #[inline]
    pub fn from_hifitime_epoch(epoch: Epoch) -> Self {
        // Both sides expressed as TAI duration since hifitime's prime epoch:
        // their difference is true elapsed SI time.
        let elapsed = epoch.to_tai_duration() - *PTP_EPOCH_TAI_DUR;
        Self(saturate_to_i64(elapsed.total_nanoseconds()))
    }

    /// Convert to a [`hifitime::Epoch`] (TAI scale).
    #[inline]
    pub fn to_hifitime_epoch(self) -> Epoch {
        let elapsed = HfDuration::from_truncated_nanoseconds(self.0);
        Epoch::from_tai_duration(*PTP_EPOCH_TAI_DUR + elapsed)
    }

    /// Convert from UTC nanoseconds — i.e. nanoseconds since
    /// 1970-01-01 00:00:00 UTC under the UNIX time convention (every day
    /// = 86 400 seconds, leap seconds skipped).  Use this when ingesting
    /// a timestamp that was generated under a leap-second-aware UTC
    /// clock, to convert it onto this crate's TAI timeline.
    ///
    /// Applies the TAI−UTC offset via [`hifitime`]'s IERS table.
    #[inline]
    pub fn from_utc_nanos(utc_ns: i64) -> Self {
        let utc_dur = HfDuration::from_truncated_nanoseconds(utc_ns);
        Self::from_hifitime_epoch(Epoch::from_unix_duration(utc_dur))
    }

    /// Return UTC nanoseconds — the UNIX-time representation of this
    /// instant.  Use this when emitting a timestamp to an external
    /// leap-second-aware system, or for plotting against UTC wall-clock
    /// axes.
    ///
    /// Subtracts the TAI−UTC offset via [`hifitime`]'s IERS table.  The
    /// inverse of [`from_utc_nanos`](Self::from_utc_nanos) is exact
    /// except at the instants of leap-second insertions, where UTC is
    /// not injective in UNIX time.
    #[inline]
    pub fn to_utc_nanos(self) -> i64 {
        let epoch = self.to_hifitime_epoch();
        // UNIX duration = elapsed UTC seconds since 1970-01-01 00:00:00 UTC.
        let unix_dur =
            epoch.to_duration_in_time_scale(TimeScale::UTC) - UNIX_REF_EPOCH.to_utc_duration();
        saturate_to_i64(unix_dur.total_nanoseconds())
    }
}

// ===========================================================================
// UTC ↔ TAI free functions (also exposed to Python via the bridge)
// ===========================================================================

/// Convert a UTC timestamp in nanoseconds (UNIX-time convention) to a TAI
/// timestamp in nanoseconds since the PTP epoch.
///
/// Shorthand for `Instant::from_utc_nanos(utc_ns).as_nanos()`.
#[inline]
pub fn utc_to_tai(utc_ns: i64) -> i64 {
    Instant::from_utc_nanos(utc_ns).as_nanos()
}

/// Convert a TAI timestamp in nanoseconds since the PTP epoch to a UTC
/// timestamp in nanoseconds (UNIX-time convention).
///
/// Shorthand for `Instant::from_nanos(tai_ns).to_utc_nanos()`.
#[inline]
pub fn tai_to_utc(tai_ns: i64) -> i64 {
    Instant::from_nanos(tai_ns).to_utc_nanos()
}

/// PTP epoch (1970-01-01 00:00:00 TAI) expressed as a TAI duration since
/// hifitime's prime epoch.  Computed once via [`hifitime`] to avoid
/// hand-deriving Gregorian calendar arithmetic.
static PTP_EPOCH_TAI_DUR: LazyLock<HfDuration> =
    LazyLock::new(|| Epoch::from_gregorian_tai_at_midnight(1970, 1, 1).to_tai_duration());

#[inline]
fn saturate_to_i64(ns: i128) -> i64 {
    ns.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

impl fmt::Display for Instant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instant({} ns TAI)", self.0)
    }
}

// ===========================================================================
// Duration
// ===========================================================================

/// An elapsed interval in SI nanoseconds.
///
/// `repr(transparent)` over `i64`.  Signed: may be negative.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(i64);

impl Duration {
    /// Zero duration.
    pub const ZERO: Duration = Duration(0);

    /// Minimum representable duration.
    pub const MIN: Duration = Duration(i64::MIN);

    /// Maximum representable duration.
    pub const MAX: Duration = Duration(i64::MAX);

    /// Create from SI nanoseconds.
    #[inline(always)]
    pub const fn from_nanos(ns: i64) -> Self {
        Self(ns)
    }

    /// Create from SI microseconds.
    #[inline(always)]
    pub const fn from_micros(us: i64) -> Self {
        Self(us * 1_000)
    }

    /// Create from SI milliseconds.
    #[inline(always)]
    pub const fn from_millis(ms: i64) -> Self {
        Self(ms * 1_000_000)
    }

    /// Create from SI seconds.
    #[inline(always)]
    pub const fn from_seconds(s: i64) -> Self {
        Self(s * 1_000_000_000)
    }

    /// Create from SI minutes (60 SI seconds).
    #[inline(always)]
    pub const fn from_minutes(m: i64) -> Self {
        Self(m * 60 * 1_000_000_000)
    }

    /// Create from SI hours (3600 SI seconds).
    #[inline(always)]
    pub const fn from_hours(h: i64) -> Self {
        Self(h * 3600 * 1_000_000_000)
    }

    /// Create from SI days (86 400 SI seconds — not calendar days, which vary
    /// around DST and leap seconds).
    #[inline(always)]
    pub const fn from_days(d: i64) -> Self {
        Self(d * 86_400 * 1_000_000_000)
    }

    /// Return SI nanoseconds.
    #[inline(always)]
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    /// Return SI seconds (floor toward zero).
    #[inline(always)]
    pub const fn as_seconds(self) -> i64 {
        self.0 / 1_000_000_000
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Duration({} ns)", self.0)
    }
}

// ===========================================================================
// Arithmetic
// ===========================================================================

impl Sub<Instant> for Instant {
    type Output = Duration;
    #[inline(always)]
    fn sub(self, rhs: Instant) -> Duration {
        Duration(self.0 - rhs.0)
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;
    #[inline(always)]
    fn add(self, rhs: Duration) -> Instant {
        Instant(self.0 + rhs.0)
    }
}

impl AddAssign<Duration> for Instant {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.0;
    }
}

impl Sub<Duration> for Instant {
    type Output = Instant;
    #[inline(always)]
    fn sub(self, rhs: Duration) -> Instant {
        Instant(self.0 - rhs.0)
    }
}

impl SubAssign<Duration> for Instant {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs.0;
    }
}

impl Add<Duration> for Duration {
    type Output = Duration;
    #[inline(always)]
    fn add(self, rhs: Duration) -> Duration {
        Duration(self.0 + rhs.0)
    }
}

impl AddAssign<Duration> for Duration {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.0;
    }
}

impl Sub<Duration> for Duration {
    type Output = Duration;
    #[inline(always)]
    fn sub(self, rhs: Duration) -> Duration {
        Duration(self.0 - rhs.0)
    }
}

impl SubAssign<Duration> for Duration {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs.0;
    }
}

impl Mul<i64> for Duration {
    type Output = Duration;
    #[inline(always)]
    fn mul(self, rhs: i64) -> Duration {
        Duration(self.0 * rhs)
    }
}

impl Mul<Duration> for i64 {
    type Output = Duration;
    #[inline(always)]
    fn mul(self, rhs: Duration) -> Duration {
        Duration(self * rhs.0)
    }
}

impl Neg for Duration {
    type Output = Duration;
    #[inline(always)]
    fn neg(self) -> Duration {
        Duration(-self.0)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_transparent_layout() {
        assert_eq!(std::mem::size_of::<Instant>(), std::mem::size_of::<i64>());
        assert_eq!(std::mem::align_of::<Instant>(), std::mem::align_of::<i64>());
        assert_eq!(std::mem::size_of::<Duration>(), std::mem::size_of::<i64>());
    }

    #[test]
    fn arithmetic() {
        let a = Instant::from_nanos(1_000);
        let b = Instant::from_nanos(2_500);
        let d = b - a;
        assert_eq!(d.as_nanos(), 1_500);

        let c = a + Duration::from_nanos(500);
        assert_eq!(c.as_nanos(), 1_500);

        let e = b - Duration::from_seconds(1);
        assert_eq!(e.as_nanos(), 2_500 - 1_000_000_000);

        let sum = Duration::from_millis(2) + Duration::from_micros(500);
        assert_eq!(sum.as_nanos(), 2_500_000);

        let scaled = Duration::from_seconds(3) * 4;
        assert_eq!(scaled.as_seconds(), 12);
    }

    #[test]
    fn ordering() {
        let a = Instant::from_nanos(1);
        let b = Instant::from_nanos(2);
        assert!(a < b);
        assert!(a != b);
        assert_eq!(a, Instant::from_nanos(1));
    }

    #[test]
    fn slice_reinterpret_roundtrip() {
        let ns: Vec<i64> = vec![1, 2, 3];
        let ts = Instant::from_nanos_slice(&ns);
        assert_eq!(ts.len(), 3);
        assert_eq!(ts[0].as_nanos(), 1);
        let back = Instant::as_nanos_slice(ts);
        assert_eq!(back, &ns[..]);
    }

    #[test]
    fn utc_conversion_modern() {
        // 2024-01-01 00:00:00 UTC → UNIX seconds = 1_704_067_200.
        // TAI−UTC = 37 s → TAI = 1_704_067_237 since PTP epoch (approx).
        let unix_ns = 1_704_067_200_i64 * 1_000_000_000;
        let inst = Instant::from_utc_nanos(unix_ns);
        assert_eq!(inst.as_nanos(), unix_ns + 37 * 1_000_000_000);
        assert_eq!(inst.to_utc_nanos(), unix_ns);
    }

    #[test]
    fn utc_conversion_boundary() {
        // 2017-01-01 00:00:00 UTC: offset becomes 37 s.
        let unix_ns = 1_483_228_800_i64 * 1_000_000_000;
        let inst = Instant::from_utc_nanos(unix_ns);
        assert_eq!(inst.as_nanos(), unix_ns + 37 * 1_000_000_000);

        // One second before: offset was still 36 s.
        let unix_ns = 1_483_228_799_i64 * 1_000_000_000;
        let inst = Instant::from_utc_nanos(unix_ns);
        assert_eq!(inst.as_nanos(), unix_ns + 36 * 1_000_000_000);
    }

    #[test]
    fn utc_conversion_pre_1972() {
        // hifitime's IERS table starts at 1972-01-01.  Pre-1972 timestamps
        // get zero offset (so PTP_time = unix_time before 1972), losing the
        // ~8 s pre-1972 rubber-second offset.  Round-trip is still exact.
        let unix_ns = 0_i64;
        let inst = Instant::from_utc_nanos(unix_ns);
        assert_eq!(inst.as_nanos(), 0);
        assert_eq!(inst.to_utc_nanos(), unix_ns);
    }
}
