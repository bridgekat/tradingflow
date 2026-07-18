//! Integration tests for `Instant` / `Duration` and the UTC ↔ TAI conversions.

use tradingflow_data::{Duration, Instant};

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
