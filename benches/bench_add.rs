//! Benchmarks for the add operator across different dispatch paths.
//!
//! Run with: `cargo bench`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tradingflow::operators::{Record, num::Add};
use tradingflow::sources::ArraySource;
use tradingflow::{Array, Instant, Notify, Operator, Scenario, Series};

const N: usize = 10_000;

fn make_data() -> (Vec<Instant>, Vec<f64>, Vec<f64>) {
    let ts: Vec<Instant> = (0..N as i64).map(Instant::from_nanos).collect();
    let a: Vec<f64> = (0..N).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..N).map(|i| i as f64 * 0.2).collect();
    (ts, a, b)
}

// ---------------------------------------------------------------------------
// Baseline: plain add (element-only)
// ---------------------------------------------------------------------------

fn bench_baseline_add(c: &mut Criterion) {
    let (_, a, b) = make_data();

    c.bench_function("baseline_add", |bencher| {
        bencher.iter(|| {
            let mut elem_a;
            let mut elem_b;
            let mut elem_out;
            for i in 0..N {
                elem_a = a[i];
                elem_b = b[i];
                elem_out = elem_a + elem_b;
                black_box(elem_out);
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Baseline: plain add (output to series)
// ---------------------------------------------------------------------------

fn bench_baseline_add_series(c: &mut Criterion) {
    let (_, a, b) = make_data();

    c.bench_function("baseline_add_series", |bencher| {
        bencher.iter(|| {
            let mut elem_a;
            let mut elem_b;
            let mut vec_out = Vec::new();
            for i in 0..N {
                elem_a = a[i];
                elem_b = b[i];
                vec_out.push(elem_a + elem_b);
            }
            black_box(vec_out[N - 1]);
        });
    });
}

// ---------------------------------------------------------------------------
// Array with compute (element-only)
// ---------------------------------------------------------------------------

fn bench_direct_compute(c: &mut Criterion) {
    let (_, a, b) = make_data();

    c.bench_function("direct_compute", |bencher| {
        bencher.iter(|| {
            let mut arr_a = Array::scalar(0.0_f64);
            let mut arr_b = Array::scalar(0.0_f64);
            let mut arr_out = Array::scalar(0.0_f64);
            let (mut state, _) = Add::new().init((&arr_a, &arr_b), Instant::MIN);
            for i in 0..N {
                arr_a[0] = a[i];
                arr_b[0] = b[i];
                Add::<f64>::compute(
                    &mut state,
                    (&arr_a, &arr_b),
                    &mut arr_out,
                    Instant::from_nanos(i as i64),
                    &Notify::new(&[], 0),
                );
            }
            black_box(arr_out[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Array with compute (output to series)
// ---------------------------------------------------------------------------

fn bench_direct_compute_series(c: &mut Criterion) {
    let (_, a, b) = make_data();

    c.bench_function("direct_compute_series", |bencher| {
        bencher.iter(|| {
            let mut arr_a = Array::scalar(0.0_f64);
            let mut arr_b = Array::scalar(0.0_f64);
            let mut arr_out = Array::scalar(0.0_f64);
            let (mut state, _) = Add::new().init((&arr_a, &arr_b), Instant::MIN);
            let mut series_out = Series::new(&[]);
            for i in 0..N {
                arr_a[0] = a[i];
                arr_b[0] = b[i];
                Add::<f64>::compute(
                    &mut state,
                    (&arr_a, &arr_b),
                    &mut arr_out,
                    Instant::from_nanos(i as i64),
                    &Notify::new(&[], 0),
                );
                Record::<f64>::compute(
                    &mut (),
                    &arr_out,
                    &mut series_out,
                    Instant::from_nanos(i as i64),
                    &Notify::new(&[], 0),
                );
            }
            black_box(series_out.last().unwrap()[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Scenario operator (element-only)
// ---------------------------------------------------------------------------

fn bench_scenario_operator(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("scenario_operator", |bencher| {
        bencher.iter(|| {
            let mut sc = Scenario::new();
            let ha = sc.add_const(Array::scalar(0.0_f64));
            let hb = sc.add_const(Array::scalar(0.0_f64));
            let ho = sc.add_operator(Add::new(), (ha, hb));
            for i in 0..N {
                sc.value_mut(ha)[0] = a[i];
                sc.value_mut(hb)[0] = b[i];
                sc.flush(ts[i], &[ha.index(), hb.index()]);
            }
            black_box(sc.value(ho)[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Scenario operator (output to series)
// ---------------------------------------------------------------------------

fn bench_scenario_operator_series(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("scenario_operator_series", |bencher| {
        bencher.iter(|| {
            let mut sc = Scenario::new();
            let ha = sc.add_const(Array::scalar(0.0_f64));
            let hb = sc.add_const(Array::scalar(0.0_f64));
            let ho = sc.add_operator(Add::new(), (ha, hb));
            let hos = sc.add_operator(Record::new(), ho);
            for i in 0..N {
                sc.value_mut(ha)[0] = a[i];
                sc.value_mut(hb)[0] = b[i];
                sc.flush(ts[i], &[ha.index(), hb.index()]);
            }
            black_box(sc.value::<Series<f64>>(hos).last().unwrap()[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Scenario operator with async source (element-only)
// ---------------------------------------------------------------------------

fn bench_scenario_source(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("scenario_source", |bencher| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let series_a = Series::from_vec(&[], ts.clone(), a.clone());
        let series_b = Series::from_vec(&[], ts.clone(), b.clone());
        let default = Array::scalar(0.0_f64);
        bencher.iter(|| {
            let _guard = rt.enter();
            let mut sc = Scenario::new();
            let ha = sc.add_source(ArraySource::new(series_a.clone(), default.clone()));
            let hb = sc.add_source(ArraySource::new(series_b.clone(), default.clone()));
            let ho = sc.add_operator(Add::new(), (ha, hb));
            rt.block_on(sc.run(|_, _, _| {}));
            black_box(sc.value(ho)[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Scenario operator with async source (output to series)
// ---------------------------------------------------------------------------

fn bench_scenario_source_series(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("scenario_source_series", |bencher| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let series_a = Series::from_vec(&[], ts.clone(), a.clone());
        let series_b = Series::from_vec(&[], ts.clone(), b.clone());
        let default = Array::scalar(0.0_f64);
        bencher.iter(|| {
            let _guard = rt.enter();
            let mut sc = Scenario::new();
            let ha = sc.add_source(ArraySource::new(series_a.clone(), default.clone()));
            let hb = sc.add_source(ArraySource::new(series_b.clone(), default.clone()));
            let ho = sc.add_operator(Add::new(), (ha, hb));
            let hos = sc.add_operator(Record::new(), ho);
            rt.block_on(sc.run(|_, _, _| {}));
            black_box(sc.value::<Series<f64>>(hos).last().unwrap()[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Scenario chain (depth operators)
// ---------------------------------------------------------------------------

fn bench_scenario_chain(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    for depth in [1, 5, 10] {
        c.bench_function(&format!("scenario_chain_depth{depth}"), |bencher| {
            bencher.iter(|| {
                let mut sc = Scenario::new();
                let ha = sc.add_const(Array::scalar(0.0_f64));
                let hb = sc.add_const(Array::scalar(0.0_f64));

                let mut prev = sc.add_operator(Add::new(), (ha, hb));
                for i in 1..depth {
                    let other = if i % 2 == 0 { ha } else { hb };
                    prev = sc.add_operator(Add::new(), (prev, other));
                }

                for i in 0..N {
                    sc.value_mut(ha)[0] = a[i];
                    sc.value_mut(hb)[0] = b[i];
                    sc.flush(ts[i], &[ha.index(), hb.index()]);
                }
                black_box(sc.value(prev)[0]);
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Scenario sparse graph (many operators, few active)
// ---------------------------------------------------------------------------

fn bench_scenario_sparse(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    for (total, active) in [(100, 5), (1000, 5)] {
        c.bench_function(
            &format!("scenario_sparse_{total}total_{active}active"),
            |bencher| {
                bencher.iter(|| {
                    let mut sc = Scenario::new();
                    let ha = sc.add_const(Array::scalar(0.0_f64));
                    let hb = sc.add_const(Array::scalar(0.0_f64));
                    let hc = sc.add_const(Array::scalar(0.0_f64));
                    let hd = sc.add_const(Array::scalar(0.0_f64));

                    // Active chain
                    let mut last = sc.add_operator(Add::new(), (ha, hb));
                    for _ in 1..active {
                        last = sc.add_operator(Add::new(), (last, ha));
                    }

                    // Inactive chain
                    let inactive = total - active;
                    if inactive > 0 {
                        let mut prev = sc.add_operator(Add::new(), (hc, hd));
                        for _ in 1..inactive {
                            prev = sc.add_operator(Add::new(), (prev, hc));
                        }
                    }

                    for i in 0..N {
                        sc.value_mut(ha)[0] = a[i];
                        sc.value_mut(hb)[0] = b[i];
                        sc.flush(ts[i], &[ha.index(), hb.index()]);
                    }
                    black_box(sc.value(last)[0]);
                });
            },
        );
    }
}

criterion_group!(
    benches,
    bench_baseline_add,
    bench_baseline_add_series,
    bench_direct_compute,
    bench_direct_compute_series,
    bench_scenario_operator,
    bench_scenario_operator_series,
    bench_scenario_source,
    bench_scenario_source_series,
    bench_scenario_chain,
    bench_scenario_sparse,
);
criterion_main!(benches);
