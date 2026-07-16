//! Integration tests for the strategy engine end to end: TradingFlow sources
//! and operators driven by `tradingflow::ingest`'s [`Scenario`] / [`Session`]
//! — event merging, coalesced batches, stale-input gating, and the driver's
//! event counters. (The merge semantics themselves are covered in
//! `tests/ingest.rs`.)

use tradingflow::operators::{add, as_view, filter, record};
use tradingflow::sources::ArraySource;
use tradingflow::{Array, ArrayView, Instant, Scenario, Series, WallClock};

fn tss(xs: &[i64]) -> Vec<Instant> {
    xs.iter().copied().map(Instant::from_nanos).collect()
}

fn src(ts: &[i64], vals: &[f64]) -> ArraySource<f64, 0> {
    ArraySource::new(
        Series::from_vec([], tss(ts), vals.to_vec()),
        Array::scalar(0.0),
    )
}

/// Replay [10,20,30] @ [1,2,3] into a Record.
#[tokio::test]
async fn run_single_source_record() {
    let mut sc = Scenario::new(WallClock);
    let h = sc.add_source(src(&[1, 2, 3], &[10.0, 20.0, 30.0]));
    let hv = sc.push(as_view(), h);
    let hrec = sc.push(record(), hv);

    let mut session = sc.build();
    session.run(|_, _| {}).await;

    let s: &Series<f64, 0> = session.ref_view(hrec);
    assert_eq!(s.timestamps(), tss(&[1, 2, 3]).as_slice());
    assert_eq!(s.values(), &[10.0, 20.0, 30.0]);
}

/// Staggered timestamps; the un-fired input keeps its stale value.
/// ts1:10+0, ts2:10+20, ts3:30+40 → [10,30,70].
#[tokio::test]
async fn run_two_sources_add() {
    let mut sc = Scenario::new(WallClock);
    let ha = sc.add_source(src(&[1, 3], &[10.0, 30.0]));
    let hb = sc.add_source(src(&[2, 3], &[20.0, 40.0]));
    let (hav, hbv) = (sc.push(as_view(), ha), sc.push(as_view(), hb));
    let ho = sc.push(add(), (hav, hbv));
    let hrec = sc.push(record(), ho);

    let mut session = sc.build();
    session.run(|_, _| {}).await;

    let s: &Series<f64, 0> = session.ref_view(hrec);
    assert_eq!(s.timestamps(), tss(&[1, 2, 3]).as_slice());
    assert_eq!(s.values(), &[10.0, 30.0, 70.0]);
}

/// Two sources at the same timestamps → one coalesced batch each.
/// ts1:10+100, ts2:20+200 → [110,220].
#[tokio::test]
async fn run_coalescing() {
    let mut sc = Scenario::new(WallClock);
    let ha = sc.add_source(src(&[1, 2], &[10.0, 20.0]));
    let hb = sc.add_source(src(&[1, 2], &[100.0, 200.0]));
    let (hav, hbv) = (sc.push(as_view(), ha), sc.push(as_view(), hb));
    let ho = sc.push(add(), (hav, hbv));
    let hrec = sc.push(record(), ho);

    let mut session = sc.build();
    session.run(|_, _| {}).await;

    let s: &Series<f64, 0> = session.ref_view(hrec);
    assert_eq!(s.timestamps(), tss(&[1, 2]).as_slice());
    assert_eq!(s.values(), &[110.0, 220.0]);
}

/// The cutoff must survive the driver — dropped ticks produce no Record row.
/// [1,5,2,10] keep >3 → (2,5),(4,10).
#[tokio::test]
async fn run_filter_cutoff() {
    let mut sc = Scenario::new(WallClock);
    let h = sc.add_source(src(&[1, 2, 3, 4], &[1.0, 5.0, 2.0, 10.0]));
    let hv = sc.push(as_view(), h);
    let hf = sc.push(
        filter(|v: ArrayView<f64, 0>| v.to_contiguous()[0] > 3.0),
        hv,
    );
    let hrec = sc.push(record(), hf);

    let mut session = sc.build();
    session.run(|_, _| {}).await;

    let s: &Series<f64, 0> = session.ref_view(hrec);
    assert_eq!(s.len(), 2);
    assert_eq!(s.timestamps(), tss(&[2, 4]).as_slice());
    assert_eq!(s.values(), &[5.0, 10.0]);
}

/// `on_stable` fires once per coalesced batch (3 distinct timestamps here),
/// and the driver's event counter advances per logical event. The callback
/// runs on the session's own task, so it can borrow locals.
#[tokio::test]
async fn on_stable_per_batch() {
    let mut sc = Scenario::new(WallClock);
    let h = sc.add_source(src(&[1, 2, 3], &[10.0, 20.0, 30.0]));
    let hv = sc.push(as_view(), h);
    let _ = sc.push(record(), hv);

    let mut session = sc.build();
    assert_eq!(session.total_num_events(), Some(3));
    let mut batches = Vec::new();
    session
        .run(|s, ts| batches.push((ts.as_nanos(), s.num_events())))
        .await;
    assert_eq!(batches, vec![(1, 1), (2, 2), (3, 3)]);
}
