use tradingflow::data::{Array, ArrayView, Duration, Instant, Series, SeriesView};
use tradingflow::graph::{Builder, Pool};
use tradingflow::operators::{elem::add, series::record_all, signal, signal::filter};
use tradingflow::sources::basic::*;
use tradingflow::time::UnixTime;

fn pool() -> Pool {
    Pool::new(0)
}

fn tss(xs: &[i64]) -> Vec<Instant> {
    xs.iter()
        .copied()
        .map(|ns| Instant::from_offset(Duration::from_nanos(ns)))
        .collect()
}

fn src(ts: &[i64], vals: &[f64]) -> ArraySource<f64, 0> {
    array_source(
        Array::scalar(0.0),
        Series::from_parts([], tss(ts), vals.to_vec(), 0),
    )
}

/// Replay [10,20,30] @ [1,2,3] into a Record.
#[tokio::test]
async fn run_single_source_record_all() {
    let mut sc = Builder::new(UnixTime);
    let h = sc.source(src(&[1, 2, 3], &[10.0, 20.0, 30.0]));
    // The source is a plain array; its derived signal marks each emission, and
    // the record appends one row per pulse.
    let hc = sc.segment(signal::always(), h);
    let hrec = sc.segment(record_all(), (hc, h));

    let mut session = sc.build();
    session.run(&mut pool(), |_, _| {}).await;

    let s: SeriesView<f64, 0> = session.view(hrec);
    assert_eq!(s.instants(), tss(&[1, 2, 3]).as_slice());
    assert_eq!(&*s.to_contiguous(), &[10.0, 20.0, 30.0]);
}

/// Staggered timestamps; the un-fired input keeps its stale value.
/// ts1:10+0, ts2:10+20, ts3:30+40 → [10,30,70].
#[tokio::test]
async fn run_two_sources_add() {
    let mut sc = Builder::new(UnixTime);
    let ha = sc.source(src(&[1, 3], &[10.0, 30.0]));
    let hb = sc.source(src(&[2, 3], &[20.0, 40.0]));
    let ho = sc.segment(add(), (ha, hb));
    // A signal derived from the sum node fires on the union of the two source
    // cadences — one record row per generation that recomputed the sum.
    let hoc = sc.segment(signal::always(), ho);
    let hrec = sc.segment(record_all(), (hoc, ho));

    let mut session = sc.build();
    session.run(&mut pool(), |_, _| {}).await;

    let s: SeriesView<f64, 0> = session.view(hrec);
    assert_eq!(s.instants(), tss(&[1, 2, 3]).as_slice());
    assert_eq!(&*s.to_contiguous(), &[10.0, 30.0, 70.0]);
}

/// Two sources at the same timestamps → one coalesced batch each.
/// ts1:10+100, ts2:20+200 → [110,220].
#[tokio::test]
async fn run_coalescing() {
    let mut sc = Builder::new(UnixTime);
    let ha = sc.source(src(&[1, 2], &[10.0, 20.0]));
    let hb = sc.source(src(&[1, 2], &[100.0, 200.0]));
    let ho = sc.segment(add(), (ha, hb));
    let hoc = sc.segment(signal::always(), ho);
    let hrec = sc.segment(record_all(), (hoc, ho));

    let mut session = sc.build();
    session.run(&mut pool(), |_, _| {}).await;

    let s: SeriesView<f64, 0> = session.view(hrec);
    assert_eq!(s.instants(), tss(&[1, 2]).as_slice());
    assert_eq!(&*s.to_contiguous(), &[110.0, 220.0]);
}

/// The cutoff must survive the driver — dropped ticks produce no Record row.
/// [1,5,2,10] keep >3 → (2,5),(4,10).
#[tokio::test]
async fn run_filter_cutoff() {
    let mut sc = Builder::new(UnixTime);
    let h = sc.source(src(&[1, 2, 3, 4], &[1.0, 5.0, 2.0, 10.0]));
    let hc = sc.segment(signal::always(), h);
    let hf = sc.segment(
        filter(|v: ArrayView<f64, 0>| v.to_contiguous()[0] > 3.0),
        (hc, h),
    );
    let hrec = sc.segment(record_all(), (hf, h));

    let mut session = sc.build();
    session.run(&mut pool(), |_, _| {}).await;

    let s: SeriesView<f64, 0> = session.view(hrec);
    assert_eq!(s.len(), 2);
    assert_eq!(s.instants(), tss(&[2, 4]).as_slice());
    assert_eq!(&*s.to_contiguous(), &[5.0, 10.0]);
}

/// `on_stable` fires once per coalesced batch (3 distinct timestamps here),
/// and the driver's event counter advances per logical event. The callback
/// runs on the session's own task, so it can borrow locals.
#[tokio::test]
async fn on_stable_per_batch() {
    let mut sc = Builder::new(UnixTime);
    let h = sc.source(src(&[1, 2, 3], &[10.0, 20.0, 30.0]));
    let hc = sc.segment(signal::always(), h);
    let _ = sc.segment(record_all(), (hc, h));

    let mut session = sc.build();
    assert_eq!(session.size_hint(), Some(3));
    let mut batches = Vec::new();
    session
        .run(&mut pool(), |s, ts| {
            batches.push((ts.as_offset().as_nanos(), s.num_events()))
        })
        .await;
    assert_eq!(batches, vec![(1, 1), (2, 2), (3, 3)]);
}
