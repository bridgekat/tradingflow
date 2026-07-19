use tradingflow_data::{Duration, Instant};

#[test]
fn repr_transparent_layout() {
    assert_eq!(std::mem::size_of::<Instant>(), std::mem::size_of::<i64>());
    assert_eq!(std::mem::align_of::<Instant>(), std::mem::align_of::<i64>());
    assert_eq!(std::mem::size_of::<Duration>(), std::mem::size_of::<i64>());
}

#[test]
fn arithmetic() {
    let a = Instant::from_offset(Duration::from_nanos(1_000));
    let b = Instant::from_offset(Duration::from_nanos(2_500));
    let d = b - a;
    assert_eq!(d.as_nanos(), 1_500);

    let c = a + Duration::from_nanos(500);
    assert_eq!(c.as_offset().as_nanos(), 1_500);

    let e = b - Duration::from_seconds(1);
    assert_eq!(e.as_offset().as_nanos(), 2_500 - 1_000_000_000);

    let sum = Duration::from_millis(2) + Duration::from_micros(500);
    assert_eq!(sum.as_nanos(), 2_500_000);

    let scaled = Duration::from_seconds(3) * 4;
    assert_eq!(scaled.as_seconds(), 12);
}

#[test]
fn ordering() {
    let a = Instant::from_offset(Duration::from_nanos(1));
    let b = Instant::from_offset(Duration::from_nanos(2));
    assert!(a < b);
    assert!(a != b);
    assert_eq!(a, Instant::from_offset(Duration::from_nanos(1)));
}

#[test]
fn slice_reinterpret_roundtrip() {
    let ns: Vec<i64> = vec![1, 2, 3];
    let ts = Instant::from_nanos_slice(&ns);
    assert_eq!(ts.len(), 3);
    assert_eq!(ts[0].as_offset().as_nanos(), 1);
    let back = Instant::as_nanos_slice(ts);
    assert_eq!(back, &ns[..]);
}
