//! Benchmarks for Scenario::run() with ArraySource-driven graphs.
//!
//! Measures end-to-end throughput including channel creation, async event
//! delivery, event ordering, and DAG flush.
//!
//! Run with: `cargo bench --bench bench_scenario`

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use tradingflow::Instant;
use tradingflow::data::array::Array;
use tradingflow::data::series::Series;
use tradingflow::operators::Record;
use tradingflow::operators::num::{Add, Multiply, Negate};
use tradingflow::scenario::Scenario;
use tradingflow::sources::ArraySource;

const N: usize = 10_000;

fn instants(it: impl Iterator<Item = i64>) -> Vec<Instant> {
    it.map(Instant::from_nanos).collect()
}

fn make_series(n: usize) -> (Series<f64>, Array<f64>) {
    let ts = instants(0..n as i64);
    let vals: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    (Series::from_vec(&[], ts, vals), Array::scalar(0.0))
}

fn make_series_vec(n: usize, stride: usize) -> (Series<f64>, Array<f64>) {
    let ts = instants(0..n as i64);
    let vals: Vec<f64> = (0..n * stride).map(|i| i as f64 * 0.01).collect();
    (
        Series::from_vec(&[stride], ts, vals),
        Array::zeros(&[stride]),
    )
}

// ---------------------------------------------------------------------------
// Single source → Record
// ---------------------------------------------------------------------------

fn bench_single_source(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (series, default) = make_series(N);

    c.bench_function("scenario_single_source", |bencher| {
        bencher.iter(|| {
            let _guard = rt.enter();
            let mut sc = Scenario::new();
            let h = sc.add_source(ArraySource::new(series.clone(), default.clone()));
            let hs = sc.add_operator(Record::new(), h);
            rt.block_on(sc.run(|_, _, _| {}));
            black_box(sc.value::<Series<f64>>(hs).len());
        });
    });
}

// ---------------------------------------------------------------------------
// Two sources → Add → Record
// ---------------------------------------------------------------------------

fn bench_two_sources_add(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (sa, da) = make_series(N);
    let (sb, db) = make_series(N);

    c.bench_function("scenario_two_sources_add", |bencher| {
        bencher.iter(|| {
            let _guard = rt.enter();
            let mut sc = Scenario::new();
            let ha = sc.add_source(ArraySource::new(sa.clone(), da.clone()));
            let hb = sc.add_source(ArraySource::new(sb.clone(), db.clone()));
            let ho = sc.add_operator(Add::new(), (ha, hb));
            let hs = sc.add_operator(Record::new(), ho);
            rt.block_on(sc.run(|_, _, _| {}));
            black_box(sc.value::<Series<f64>>(hs).len());
        });
    });
}

// ---------------------------------------------------------------------------
// Two sources → Add → Negate → Record (chain depth 2)
// ---------------------------------------------------------------------------

fn bench_two_sources_chain(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (sa, da) = make_series(N);
    let (sb, db) = make_series(N);

    c.bench_function("scenario_two_sources_chain", |bencher| {
        bencher.iter(|| {
            let _guard = rt.enter();
            let mut sc = Scenario::new();
            let ha = sc.add_source(ArraySource::new(sa.clone(), da.clone()));
            let hb = sc.add_source(ArraySource::new(sb.clone(), db.clone()));
            let ho = sc.add_operator(Add::new(), (ha, hb));
            let hn = sc.add_operator(Negate::new(), ho);
            let hs = sc.add_operator(Record::new(), hn);
            rt.block_on(sc.run(|_, _, _| {}));
            black_box(sc.value::<Series<f64>>(hs).len());
        });
    });
}

// ---------------------------------------------------------------------------
// Many sources → fan-in Add tree → Record
// ---------------------------------------------------------------------------

fn bench_fan_in(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    for n_sources in [2, 4, 8, 16] {
        let sources: Vec<_> = (0..n_sources).map(|_| make_series(N)).collect();

        c.bench_function(&format!("scenario_fan_in_{n_sources}sources"), |bencher| {
            bencher.iter(|| {
                let _guard = rt.enter();
                let mut sc = Scenario::new();
                let handles: Vec<_> = sources
                    .iter()
                    .map(|(s, d)| sc.add_source(ArraySource::new(s.clone(), d.clone())))
                    .collect();
                let mut acc = handles[0];
                for &h in &handles[1..] {
                    acc = sc.add_operator(Add::new(), (acc, h));
                }
                let hs = sc.add_operator(Record::new(), acc);
                rt.block_on(sc.run(|_, _, _| {}));
                black_box(sc.value::<Series<f64>>(hs).len());
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Interleaved timestamps (sources at different times)
// ---------------------------------------------------------------------------

fn bench_interleaved(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Source A at even timestamps, source B at odd timestamps.
    let ts_a = instants((0..N as i64).map(|i| i * 2));
    let ts_b = instants((0..N as i64).map(|i| i * 2 + 1));
    let vals: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let sa = Series::from_vec(&[], ts_a, vals.clone());
    let sb = Series::from_vec(&[], ts_b, vals);
    let default = Array::scalar(0.0);

    c.bench_function("scenario_interleaved", |bencher| {
        bencher.iter(|| {
            let _guard = rt.enter();
            let mut sc = Scenario::new();
            let ha = sc.add_source(ArraySource::new(sa.clone(), default.clone()));
            let hb = sc.add_source(ArraySource::new(sb.clone(), default.clone()));
            let ho = sc.add_operator(Add::new(), (ha, hb));
            let hs = sc.add_operator(Record::new(), ho);
            rt.block_on(sc.run(|_, _, _| {}));
            black_box(sc.value::<Series<f64>>(hs).len());
        });
    });
}

// ---------------------------------------------------------------------------
// Vector-valued sources (stride > 1)
// ---------------------------------------------------------------------------

fn bench_strided(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    for stride in [4, 16, 64] {
        let (sa, da) = make_series_vec(N, stride);
        let (sb, db) = make_series_vec(N, stride);

        c.bench_function(&format!("scenario_strided_{stride}"), |bencher| {
            bencher.iter(|| {
                let _guard = rt.enter();
                let mut sc = Scenario::new();
                let ha = sc.add_source(ArraySource::new(sa.clone(), da.clone()));
                let hb = sc.add_source(ArraySource::new(sb.clone(), db.clone()));
                let ho = sc.add_operator(Add::new(), (ha, hb));
                let hs = sc.add_operator(Record::new(), ho);
                rt.block_on(sc.run(|_, _, _| {}));
                black_box(sc.value::<Series<f64>>(hs).len());
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Diamond graph: A,B → Add → C; A → Mul(A,C) → Record
// ---------------------------------------------------------------------------

fn bench_diamond(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (sa, da) = make_series(N);
    let (sb, db) = make_series(N);

    c.bench_function("scenario_diamond", |bencher| {
        bencher.iter(|| {
            let _guard = rt.enter();
            let mut sc = Scenario::new();
            let ha = sc.add_source(ArraySource::new(sa.clone(), da.clone()));
            let hb = sc.add_source(ArraySource::new(sb.clone(), db.clone()));
            let hsum = sc.add_operator(Add::new(), (ha, hb));
            let hprod = sc.add_operator(Multiply::new(), (ha, hsum));
            let hs = sc.add_operator(Record::new(), hprod);
            rt.block_on(sc.run(|_, _, _| {}));
            black_box(sc.value::<Series<f64>>(hs).len());
        });
    });
}

criterion_group!(
    benches,
    bench_single_source,
    bench_two_sources_add,
    bench_two_sources_chain,
    bench_fan_in,
    bench_interleaved,
    bench_strided,
    bench_diamond,
);
criterion_main!(benches);
