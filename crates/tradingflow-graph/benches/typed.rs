//! `core` (in-place, work-stealing) engine benchmarks: baselines, a single
//! segment, a fused fragment chain, a sparse graph, and a "few heavy" parallel
//! case.
//!
//! A segment is currently one node; a fused "fragment chain" is one segment's
//! body. `stabilize` submits the dirty cone to a work-stealing pool and blocks
//! while the workers drain it. Idle workers park on an event counter, so a
//! sparse generation leaves them asleep while a tight burst of generations
//! keeps the recruited workers hot across the submit + block round-trips.
//!
//! Run with: `cargo bench --bench core`

use std::hint::black_box;
use std::thread;

use criterion::{Criterion, criterion_group, criterion_main};

use tradingflow_graph::pool::Pool;
use tradingflow_graph::typed::{
    Builder, Graph, NodeHandle, Operator, Port, PortHandle, Ports, Ref, Val,
};

const DATA_LEN: usize = 1 << 16;
const DATA_MASK: usize = DATA_LEN - 1;
const SERIES_LEN: usize = 1 << 16;

fn make_data() -> (Vec<f64>, Vec<f64>) {
    let a = (0..DATA_LEN).map(|i| i as f64 * 0.1).collect();
    let b = (0..DATA_LEN).map(|i| i as f64 * 0.2).collect();
    (a, b)
}

/// A type-erased binary op, used to measure indirect-call dispatch cost.
type AddFn = Box<dyn Fn(&f64, &f64) -> f64>;

/// A deliberately CPU-heavy function.
fn heavy_work(mut x: f64, iters: usize) {
    for _ in 0..iters {
        x += 1.0;
        black_box((x + 1.0).sqrt().sin().abs() + 0.5);
    }
}

fn pool() -> Pool {
    Pool::new(thread::available_parallelism().map_or(1, |n| n.get()))
}

// -- segments ---------------------------------------------------------------

/// A generic source node.
struct Source(f64);

impl Operator for Source {
    type Inputs = ();
    type Outputs = Port<Val<f64>>;
    type Context = ();
    type State = f64;
    fn init(self, _: ()) -> f64 {
        self.0
    }
    fn passthrough<'a, 'b: 'a>(_: (), state: &'b mut f64) -> (bool, f64) {
        (true, *state)
    }
    fn compute<'a, 'b: 'a>(inputs: (), state: &'b mut f64, _: &()) -> (bool, f64) {
        Self::passthrough(inputs, state)
    }
}

/// An adder with two inputs.
struct Add;

impl Operator for Add {
    type Inputs = (Port<Val<f64>>, Port<Val<f64>>);
    type Outputs = Port<Val<f64>>;
    type Context = ();
    type State = f64; // holds the last sum, to re-emit by value on passthrough

    fn init(self, ((_, a), (_, b)): ((bool, f64), (bool, f64))) -> f64 {
        a + b
    }

    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, f64), (bool, f64)),
        state: &'b mut f64,
        _: &(),
    ) -> (bool, f64) {
        *state = a + b;
        (true, *state)
    }

    fn passthrough<'a, 'b: 'a>(_: ((bool, f64), (bool, f64)), state: &'b mut f64) -> (bool, f64) {
        (false, *state)
    }
}

/// A CPU-heavy single-input kernel; `iters` lives in `State`.
struct Heavy(usize);

impl Operator for Heavy {
    type Inputs = Port<Val<f64>>;
    type Outputs = Port<Val<()>>;
    type Context = ();
    type State = usize;

    fn init(self, _: (bool, f64)) -> usize {
        self.0
    }

    fn compute<'a, 'b: 'a>((_, x): (bool, f64), state: &'b mut usize, _: &()) -> (bool, ()) {
        heavy_work(x, *state);
        (true, ())
    }

    fn passthrough<'a, 'b: 'a>(_: (bool, f64), _: &'b mut usize) -> (bool, ()) {
        (false, ())
    }
}

/// Appends each input to a growing `Vec`, mutated in place (state owns it).
struct Record;

impl Operator for Record {
    type Inputs = Port<Val<f64>>; // scalar in by value, `Vec` out by reference
    type Outputs = Port<Ref<Vec<f64>>>;
    type Context = ();
    type State = Vec<f64>;

    fn init(self, (_, x): (bool, f64)) -> Vec<f64> {
        vec![x]
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, f64),
        state: &'b mut Vec<f64>,
        _: &(),
    ) -> (bool, &'a Vec<f64>) {
        if state.len() >= SERIES_LEN {
            black_box(state.last());
            state.clear();
        }
        state.push(x);
        (true, &*state)
    }

    fn passthrough<'a, 'b: 'a>(_: (bool, f64), state: &'b mut Vec<f64>) -> (bool, &'a Vec<f64>) {
        (false, state)
    }
}

// -- baselines (engine-independent, for reference) ---------------------------

fn bench_baseline_add(c: &mut Criterion) {
    let (a, b) = make_data();
    let mut i = 0;

    c.bench_function("baseline_add", |bencher| {
        bencher.iter(|| {
            black_box(a[i] + b[i]);
            i = (i + 1) & DATA_MASK;
        });
    });
}

fn bench_baseline_add_series(c: &mut Criterion) {
    let (a, b) = make_data();
    let mut i = 0;
    let mut series = Vec::with_capacity(0);

    c.bench_function("baseline_add_series", |bencher| {
        bencher.iter(|| {
            if series.len() >= SERIES_LEN {
                black_box(series.last());
                series.clear();
            }
            series.push(a[i] + b[i]);
            i = (i + 1) & DATA_MASK;
        });
    });
}

// Direct erased-closure compute, no engine: an indirect `Box<dyn Fn>` call per
// element (the dispatch cost a node's `compute_fn` pays once per generation).

fn bench_direct_compute(c: &mut Criterion) {
    let (a, b) = make_data();
    let mut i = 0;
    let add_dyn: AddFn = Box::new(|a, b| a + b);

    c.bench_function("direct_compute", |bencher| {
        bencher.iter(|| {
            black_box(add_dyn(&a[i], &b[i]));
            i = (i + 1) & DATA_MASK;
        });
    });
}

fn bench_direct_compute_series(c: &mut Criterion) {
    let (a, b) = make_data();
    let mut i = 0;
    let add_dyn: AddFn = Box::new(|a, b| a + b);
    let mut series = Vec::with_capacity(0);

    c.bench_function("direct_compute_series", |bencher| {
        bencher.iter(|| {
            if series.len() >= SERIES_LEN {
                black_box(series.last());
                series.clear();
            }
            series.push(add_dyn(&a[i], &b[i]));
            i = (i + 1) & DATA_MASK;
        });
    });
}

// -- engine scenarios -------------------------------------------------------

fn bench_engine_segment(c: &mut Criterion) {
    let (a, b) = make_data();
    let mut i = 0;
    let mut gb = Builder::new();
    let (ha_cell, ha) = gb.source(Source(0.0));
    let (hb_cell, hb) = gb.source(Source(0.0));
    let _ = gb.segment(Add, (ha, hb));
    let mut g = gb.build();
    let mut pool = pool();

    c.bench_function("engine_segment", |bencher| {
        bencher.iter(|| {
            *g.state_mut(ha_cell) = a[i];
            *g.state_mut(hb_cell) = b[i];
            g.stabilize(&mut pool, &());
            i = (i + 1) & DATA_MASK;
        });
    });
}

fn bench_engine_segment_series(c: &mut Criterion) {
    let (a, b) = make_data();
    let mut i = 0;
    let mut gb = Builder::new();
    let (ha_cell, ha) = gb.source(Source(0.0));
    let (hb_cell, hb) = gb.source(Source(0.0));
    let sum = gb.segment(Add, (ha, hb));
    let _ = gb.segment(Record, sum);
    let mut g = gb.build();
    let mut pool = pool();

    c.bench_function("engine_segment_series", |bencher| {
        bencher.iter(|| {
            *g.state_mut(ha_cell) = a[i];
            *g.state_mut(hb_cell) = b[i];
            g.stabilize(&mut pool, &());
            i = (i + 1) & DATA_MASK;
        });
    });
}

fn bench_engine_chain(c: &mut Criterion) {
    for depth in [1usize, 5, 10] {
        let (a, b) = make_data();
        let mut i = 0;
        let mut gb = Builder::new();
        let (ha_cell, ha) = gb.source(Source(0.0));
        let (hb_cell, hb) = gb.source(Source(0.0));
        let mut last = gb.segment(Add, (ha, hb));
        for _ in 1..depth {
            last = gb.segment(Add, (last, ha));
        }
        let mut g = gb.build();
        let mut pool = pool();

        c.bench_function(&format!("engine_chain_depth{depth}"), |bencher| {
            bencher.iter(|| {
                *g.state_mut(ha_cell) = a[i];
                *g.state_mut(hb_cell) = b[i];
                g.stabilize(&mut pool, &());
                i = (i + 1) & DATA_MASK;
            });
        });
    }
}

fn bench_engine_sparse(c: &mut Criterion) {
    for (total, active) in [(100usize, 5usize), (1000, 5)] {
        let (a, b) = make_data();
        let mut i = 0;
        let mut gb = Builder::new();
        let (ha_cell, ha) = gb.source(Source(0.0));
        let (hb_cell, hb) = gb.source(Source(0.0));
        let (_, hc) = gb.source(Source(0.0));
        let (_, hd) = gb.source(Source(0.0));

        // Active chain, fed (transitively) by a, b.
        let mut last = gb.segment(Add, (ha, hb));
        for _ in 1..active {
            last = gb.segment(Add, (last, ha));
        }

        // Inactive chain, fed by c, d (never set) -> never in the dirty cone.
        let inactive = total - active;
        if inactive > 0 {
            let mut prev = gb.segment(Add, (hc, hd));
            for _ in 1..inactive {
                prev = gb.segment(Add, (prev, hc));
            }
        }

        let mut g = gb.build();
        let mut pool = pool();

        c.bench_function(
            &format!("engine_sparse_{total}total_{active}active"),
            |bencher| {
                bencher.iter(|| {
                    *g.state_mut(ha_cell) = a[i];
                    *g.state_mut(hb_cell) = b[i];
                    g.stabilize(&mut pool, &());
                    i = (i + 1) & DATA_MASK;
                });
            },
        );
    }
}

fn bench_few_heavy(c: &mut Criterion) {
    const K: usize = 4;
    const ITERS: usize = 1_000_000;

    let mut gb = Builder::new();
    let (src_cell, src) = gb.source(Source(1.0));
    let _ = (0..K)
        .map(|_| gb.segment(Heavy(ITERS), src))
        .collect::<Vec<_>>();
    let mut g = gb.build();
    let mut pool = pool();
    let mut group = c.benchmark_group("few_heavy");

    group.bench_function("engine_parallel", |bencher| {
        let mut x = 0.0f64;
        bencher.iter(|| {
            x += 1.0;
            *g.state_mut(src_cell) = x;
            g.stabilize(&mut pool, &());
        });
    });

    group.bench_function("serial_baseline", |bencher| {
        let mut x = 0.0f64;
        bencher.iter(|| {
            x += 1.0;
            for _ in 0..K {
                heavy_work(x, ITERS);
            }
        });
    });

    group.finish();
}

// -- complex mesh (large, non-trivial topology, variable workload) -----------

/// `iters` rounds of an LCG, folded mod `MODULUS` so node values (and thus
/// full-layer sums) stay bounded.
fn lcg(seed: i64, iters: u32) -> i64 {
    const MODULUS: i64 = 1_000_000_007;
    let mut h = seed.rem_euclid(MODULUS);
    for _ in 0..iters {
        h = h.wrapping_mul(48271).wrapping_add(1).rem_euclid(MODULUS);
    }
    h
}

fn work(a: i64, b: i64, c: i64, iters: u32) -> i64 {
    let seed = a
        .wrapping_mul(3)
        .wrapping_add(b.wrapping_mul(5))
        .wrapping_add(c.wrapping_mul(7));
    lcg(seed, iters)
}

/// Two local predecessors and one long-range one.
fn preds(layer: usize, j: usize, n: usize) -> [usize; 3] {
    [j, (j + 1) % n, (j * 13 + layer * 7 + 1) % n]
}

/// Mostly light (0..18 rounds), ~1/16 deliberately heavy.
fn iters_of(layer: usize, j: usize) -> u32 {
    let base = ((layer * 31 + j * 17) % 19) as u32;
    if (layer + j).is_multiple_of(16) {
        base * 40 + 200
    } else {
        base
    }
}

struct Work {
    iters: u32,
}
impl Operator for Work {
    type Inputs = (Port<Val<f64>>, Port<Val<f64>>, Port<Val<f64>>);
    type Outputs = Port<Val<f64>>;
    type State = u32; // iters
    type Context = ();
    fn init(self, _: ((bool, f64), (bool, f64), (bool, f64))) -> Self::State {
        self.iters
    }
    fn passthrough<'a, 'b: 'a>(
        ((_, a), (_, b), (_, c)): ((bool, f64), (bool, f64), (bool, f64)),
        iters: &'b mut Self::State,
    ) -> (bool, f64) {
        (true, work(a as i64, b as i64, c as i64, *iters) as f64)
    }
    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b), (_, c)): ((bool, f64), (bool, f64), (bool, f64)),
        iters: &'b mut Self::State,
        _: &(),
    ) -> (bool, f64) {
        let out = work(a as i64, b as i64, c as i64, *iters);
        (true, out as f64)
    }
}

/// Stateless unary map `x -> f(x)`; `passthrough` and `compute` are identical
/// (it recomputes from the input every generation, gate or no gate).
struct UnaryMap(fn(f64) -> f64);
impl Operator for UnaryMap {
    type Inputs = Port<Val<f64>>;
    type Outputs = Port<Val<f64>>;
    type State = fn(f64) -> f64;
    type Context = ();
    fn init(self, _: (bool, f64)) -> Self::State {
        self.0
    }
    fn passthrough<'a, 'b: 'a>((_, x): (bool, f64), f: &'b mut Self::State) -> (bool, f64) {
        (true, f(x))
    }
    fn compute<'a, 'b: 'a>(inputs: (bool, f64), f: &'b mut Self::State, _: &()) -> (bool, f64) {
        Self::passthrough(inputs, f)
    }
}
fn inc() -> UnaryMap {
    UnaryMap(|x| x + 1.0)
}
fn double() -> UnaryMap {
    UnaryMap(|x| x * 2.0)
}

struct SumAll;
impl Operator for SumAll {
    type Inputs = Ports<Val<f64>>;
    type Outputs = Port<Val<f64>>;
    type State = ();
    type Context = ();
    fn init(self, _: (&[bool], &[f64])) {}
    fn passthrough<'a, 'b: 'a>((_, xs): (&'a [bool], &'a [f64]), _: &'b mut ()) -> (bool, f64) {
        (true, xs.iter().sum())
    }
    fn compute<'a, 'b: 'a>(inputs: (&'a [bool], &'a [f64]), state: &'b mut (), _: &()) -> (bool, f64) {
        Self::passthrough(inputs, state)
    }
}

/// Build the `n` x `depth` mesh, delegating each cell's construction to
/// `cell` (a plain or fused node of the same 3-in/1-out interface), plus a
/// full-layer `SumAll` aggregate per layer. Returns the sources (to poke)
/// and the aggregates (the last of which transitively depends on everything).
fn build(
    gb: &mut Builder<()>,
    n: usize,
    depth: usize,
    mut cell: impl FnMut(
        &mut Builder<()>,
        usize,
        usize,
        [PortHandle<Val<f64>>; 3],
    ) -> PortHandle<Val<f64>>,
) -> (Vec<NodeHandle<Source>>, Vec<PortHandle<Val<f64>>>) {
    let (src, wires): (Vec<_>, Vec<_>) = (0..n).map(|j| gb.source(Source(j as f64 + 1.0))).unzip();
    let mut layers: Vec<Vec<PortHandle<Val<f64>>>> = vec![wires];
    for layer in 1..=depth {
        let prev = layers[layer - 1].clone();
        let mut cur = Vec::with_capacity(n);
        for j in 0..n {
            let [a, b, c] = preds(layer, j, n);
            cur.push(cell(&mut *gb, layer, j, [prev[a], prev[b], prev[c]]));
        }
        layers.push(cur);
    }
    let aggs = layers.iter().map(|l| gb.segment(SumAll, &l[..])).collect();
    (src, aggs)
}

fn drive_mesh<'a>(
    src: &'a [NodeHandle<Source>],
    g: &'a mut Graph<()>,
    pool: &'a mut Pool,
    _: PortHandle<Val<f64>>,
) -> impl FnMut() + 'a {
    let mut base = 0.0;
    move || {
        base += 1.0;
        for (j, &s) in src.iter().enumerate() {
            *g.state_mut(s) = base + j as f64;
        }
        g.stabilize(pool, &());
    }
}

fn bench_mesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh");
    for (w, d) in [(16usize, 16usize), (32, 32)] {
        let mut gb = Builder::new();
        let (src, aggs) = build(&mut gb, w, d, |gb, l, j, [a, b, c]| {
            gb.segment(
                Work {
                    iters: iters_of(l, j),
                },
                (a, b, c),
            )
        });
        let mut g = gb.build();
        let mut pool = pool();
        let out = *aggs.last().unwrap();
        group.bench_function(format!("plain_{w}x{d}"), |bencher| {
            bencher.iter(drive_mesh(&src, &mut g, &mut pool, out));
        });
    }
    group.finish();
}

fn bench_mesh_fusion(c: &mut Criterion) {
    let (w, d) = (24usize, 24usize);
    let mut group = c.benchmark_group("mesh_fusion");

    // The cell diamond `Add(Inc(w), Double(w))` fused into ONE scheduled node.
    {
        let mut gb = Builder::new();
        let (src, aggs) = build(&mut gb, w, d, |gb, l, j, [a, b, c]| {
            let it = iters_of(l, j);
            let seg = tradingflow::segment!(|x: Port<Val<f64>>, y: Port<Val<f64>>, z: Port<Val<f64>>| -> Port<Val<f64>> {
                let ww = Work { iters: it } @ (x, y, z);
                let p = inc() @ ww;
                let q = double() @ ww;
                let r = Add @ (p, q);
                r
            });
            gb.segment(seg, (a, b, c))
        });
        let mut g = gb.build();
        let mut pool = pool();
        let out = *aggs.last().unwrap();
        group.bench_function(format!("fused_{w}x{d}"), |bencher| {
            bencher.iter(drive_mesh(&src, &mut g, &mut pool, out));
        });
    }

    // The same diamond as FOUR scheduled nodes (identical result, 4x the nodes).
    {
        let mut gb = Builder::new();
        let (src, aggs) = build(&mut gb, w, d, |gb, l, j, [a, b, c]| {
            let w_h = gb.segment(
                Work {
                    iters: iters_of(l, j),
                },
                (a, b, c),
            );
            let p = gb.segment(inc(), w_h);
            let q = gb.segment(double(), w_h);
            gb.segment(Add, (p, q))
        });
        let mut g = gb.build();
        let mut pool = pool();
        let out = *aggs.last().unwrap();
        group.bench_function(format!("unfused_{w}x{d}"), |bencher| {
            bencher.iter(drive_mesh(&src, &mut g, &mut pool, out));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_add,
    bench_baseline_add_series,
    bench_direct_compute,
    bench_direct_compute_series,
    bench_engine_segment,
    bench_engine_segment_series,
    bench_engine_chain,
    bench_engine_sparse,
    bench_few_heavy,
    bench_mesh,
    bench_mesh_fusion,
);
criterion_main!(benches);
