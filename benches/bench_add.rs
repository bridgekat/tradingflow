//! Benchmarks for the add operator across different dispatch paths.
//!
//! Run with: `cargo bench`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tradingflow::operators;
use tradingflow::{Operator, Scenario, Store};

const N: usize = 10_000;

fn make_data() -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let ts: Vec<i64> = (0..N as i64).collect();
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
// Baseline: plain add (series)
// ---------------------------------------------------------------------------

fn bench_baseline_add_series(c: &mut Criterion) {
    let (_, a, b) = make_data();

    c.bench_function("baseline_add_series", |bencher| {
        bencher.iter(|| {
            let mut vec_a = Vec::new();
            let mut vec_b = Vec::new();
            let mut vec_out = Vec::new();
            for i in 0..N {
                vec_a.push(a[i]);
                vec_b.push(b[i]);
                vec_out.push(vec_a.last().unwrap() + vec_b.last().unwrap());
            }
            black_box(vec_out[N - 1]);
        });
    });
}

// ---------------------------------------------------------------------------
// Store with plain add (element-only)
// ---------------------------------------------------------------------------

fn bench_store_add(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("store_add", |bencher| {
        bencher.iter(|| {
            let mut store_a = Store::element(&[], &[0.0_f64]);
            let mut store_b = Store::element(&[], &[0.0_f64]);
            let mut store_out = Store::element(&[], &[0.0_f64]);
            for i in 0..N {
                store_a.push(ts[i], &[a[i]]);
                store_b.push(ts[i], &[b[i]]);
                store_out.push(ts[i], &[store_a.current()[0] + store_b.current()[0]]);
            }
            black_box(store_out.current()[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Store with plain add (series)
// ---------------------------------------------------------------------------

fn bench_store_add_series(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("store_add_series", |bencher| {
        bencher.iter(|| {
            let mut store_a = Store::series(&[], &[0.0_f64]);
            let mut store_b = Store::series(&[], &[0.0_f64]);
            let mut store_out = Store::series(&[], &[0.0_f64]);
            for i in 0..N {
                store_a.push(ts[i], &[a[i]]);
                store_b.push(ts[i], &[b[i]]);
                store_out.push(ts[i], &[store_a.current()[0] + store_b.current()[0]]);
            }
            black_box(store_out.current()[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Store compute (element-only)
// ---------------------------------------------------------------------------

fn bench_store_compute(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("store_compute", |bencher| {
        bencher.iter(|| {
            let mut store_a = Store::element(&[], &[0.0_f64]);
            let mut store_b = Store::element(&[], &[0.0_f64]);
            let mut store_out = Store::element(&[], &[0.0_f64]);
            let mut state = operators::add::<f64>().init();
            for i in 0..N {
                store_a.push(ts[i], &[a[i]]);
                store_b.push(ts[i], &[b[i]]);
                store_out.push_default(ts[i]);
                let produced = <operators::Add<f64> as Operator>::compute(
                    &mut state,
                    (&store_a, &store_b),
                    store_out.current_view_mut(),
                );
                if produced {
                    store_out.commit();
                } else {
                    store_out.rollback();
                }
            }
            black_box(store_out.current()[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Store compute (series)
// ---------------------------------------------------------------------------

fn bench_store_compute_series(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("store_compute_series", |bencher| {
        bencher.iter(|| {
            let mut store_a = Store::series(&[], &[0.0_f64]);
            let mut store_b = Store::series(&[], &[0.0_f64]);
            let mut store_out = Store::series(&[], &[0.0_f64]);
            let mut state = operators::add::<f64>().init();
            for i in 0..N {
                store_a.push(ts[i], &[a[i]]);
                store_b.push(ts[i], &[b[i]]);
                store_out.push_default(ts[i]);
                let produced = <operators::Add<f64> as Operator>::compute(
                    &mut state,
                    (&store_a, &store_b),
                    store_out.current_view_mut(),
                );
                if produced {
                    store_out.commit();
                } else {
                    store_out.rollback();
                }
            }
            black_box(store_out.current()[0]);
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
            let ha = sc.create_node::<f64>(&[], &[0.0]);
            let hb = sc.create_node::<f64>(&[], &[0.0]);
            let ho = sc.add_operator([ha, hb], operators::add());
            for i in 0..N {
                sc.store_mut(ha).push(ts[i], &[a[i]]);
                sc.store_mut(hb).push(ts[i], &[b[i]]);
                sc.flush(ts[i], &[ha.index(), hb.index()]);
            }
            black_box(sc.store(ho).current()[0]);
        });
    });
}

// ---------------------------------------------------------------------------
// Scenario operator (series)
// ---------------------------------------------------------------------------

fn bench_scenario_operator_series(c: &mut Criterion) {
    let (ts, a, b) = make_data();

    c.bench_function("scenario_operator_series", |bencher| {
        bencher.iter(|| {
            let mut sc = Scenario::new();
            let ha = sc.create_node::<f64>(&[], &[0.0]);
            let hb = sc.create_node::<f64>(&[], &[0.0]);
            let ho = sc.add_operator([ha, hb], operators::add());
            sc.store_mut(ho).ensure_min_window(0);
            for i in 0..N {
                sc.store_mut(ha).push(ts[i], &[a[i]]);
                sc.store_mut(hb).push(ts[i], &[b[i]]);
                sc.flush(ts[i], &[ha.index(), hb.index()]);
            }
            black_box(sc.store(ho).current()[0]);
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
                let ha = sc.create_node::<f64>(&[], &[0.0]);
                let hb = sc.create_node::<f64>(&[], &[0.0]);

                let mut prev = sc.add_operator([ha, hb], operators::add());
                for i in 1..depth {
                    let other = if i % 2 == 0 { ha } else { hb };
                    prev = sc.add_operator([prev, other], operators::add());
                }

                for i in 0..N {
                    sc.store_mut(ha).push(ts[i], &[a[i]]);
                    sc.store_mut(hb).push(ts[i], &[b[i]]);
                    sc.flush(ts[i], &[ha.index(), hb.index()]);
                }
                black_box(sc.store(prev).current()[0]);
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
                    let ha = sc.create_node::<f64>(&[], &[0.0]);
                    let hb = sc.create_node::<f64>(&[], &[0.0]);
                    let hc = sc.create_node::<f64>(&[], &[0.0]);
                    let hd = sc.create_node::<f64>(&[], &[0.0]);

                    // Active chain
                    let mut last = sc.add_operator([ha, hb], operators::add());
                    for _ in 1..active {
                        last = sc.add_operator([last, ha], operators::add());
                    }

                    // Inactive chain
                    let inactive = total - active;
                    if inactive > 0 {
                        let mut prev = sc.add_operator([hc, hd], operators::add());
                        for _ in 1..inactive {
                            prev = sc.add_operator([prev, hc], operators::add());
                        }
                    }

                    for i in 0..N {
                        sc.store_mut(ha).push(ts[i], &[a[i]]);
                        sc.store_mut(hb).push(ts[i], &[b[i]]);
                        sc.flush(ts[i], &[ha.index(), hb.index()]);
                    }
                    black_box(sc.store(last).current()[0]);
                });
            },
        );
    }
}

criterion_group!(
    benches,
    bench_baseline_add,
    bench_baseline_add_series,
    bench_store_add,
    bench_store_add_series,
    bench_store_compute,
    bench_store_compute_series,
    bench_scenario_operator,
    bench_scenario_operator_series,
    bench_scenario_chain,
    bench_scenario_sparse,
);
criterion_main!(benches);
