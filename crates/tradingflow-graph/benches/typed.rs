//! `core` (in-place, work-stealing) engine benchmarks: baselines, a single
//! operator, a fused subgraph, a sparse graph, and a "few heavy" parallel
//! case.
//!
//! An operator is one node; a fused "subgraph" is still one operator's
//! body. `stabilize` submits the dirty cone to a work-stealing pool and blocks
//! while the workers drain it. Idle workers park on an event counter, so a
//! sparse generation leaves them asleep while a tight burst of generations
//! keeps the recruited workers hot across the submit + block round-trips.
//!
//! Run with: `cargo bench --bench core`

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use tradingflow_graph::pool::Pool;
use tradingflow_graph::typed::{
    Builder, Graph, NodeHandle, Operator, Port, PortHandle, Ports, Val,
};

const DATA_LEN: usize = 1 << 16;
const DATA_MASK: usize = DATA_LEN - 1;

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
    fn reset<'a, 'b: 'a>(_: (), state: &'b mut f64) -> f64 {
        *state
    }
    fn compute<'a, 'b: 'a>(inputs: (), state: &'b mut f64, _: &()) -> f64 {
        Self::reset(inputs, state)
    }
}

/// An adder with two inputs.
struct Add;

impl Operator for Add {
    type Inputs = (Port<Val<f64>>, Port<Val<f64>>);
    type Outputs = Port<Val<f64>>;
    type Context = ();
    type State = f64; // holds the last sum, to re-emit by value on passthrough

    fn init(self, (a, b): (f64, f64)) -> f64 {
        a + b
    }

    fn compute<'a, 'b: 'a>((a, b): (f64, f64), state: &'b mut f64, _: &()) -> f64 {
        *state = a + b;
        *state
    }

    fn reset<'a, 'b: 'a>(_: (f64, f64), state: &'b mut f64) -> f64 {
        *state
    }
}

/// A CPU-heavy single-input kernel; `iters` lives in `State`.
struct Heavy(usize);

impl Operator for Heavy {
    type Inputs = Port<Val<f64>>;
    type Outputs = Port<Val<()>>;
    type Context = ();
    type State = usize;

    fn is_heavy(&self) -> bool {
        true
    }

    fn init(self, _: f64) -> usize {
        self.0
    }

    fn compute<'a, 'b: 'a>(x: f64, state: &'b mut usize, _: &()) {
        heavy_work(x, *state);
    }

    fn reset<'a, 'b: 'a>(_: f64, _: &'b mut usize) {}
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

// -- engine scenarios -------------------------------------------------------

fn bench_engine_operator(c: &mut Criterion) {
    let (a, b) = make_data();
    let mut i = 0;
    let mut gb = Builder::new();
    let (ha_cell, ha) = gb.source(Source(0.0));
    let (hb_cell, hb) = gb.source(Source(0.0));
    let _ = gb.op(Add, (ha, hb));
    let mut g = gb.build();

    for (threads, suffix) in [(16, ""), (0, "_1t")] {
        let mut pool = Pool::new(threads);

        c.bench_function(&format!("engine_operator{suffix}"), |bencher| {
            bencher.iter(|| {
                *g.state_mut(ha_cell) = a[i];
                *g.state_mut(hb_cell) = b[i];
                g.stabilize(&mut pool, &());
                i = (i + 1) & DATA_MASK;
            });
        });
    }
}

fn bench_engine_chain(c: &mut Criterion) {
    for depth in [1usize, 5, 10] {
        let (a, b) = make_data();
        let mut i = 0;
        let mut gb = Builder::new();
        let (ha_cell, ha) = gb.source(Source(0.0));
        let (hb_cell, hb) = gb.source(Source(0.0));
        let mut last = gb.op(Add, (ha, hb));
        for _ in 1..depth {
            last = gb.op(Add, (last, ha));
        }
        let mut g = gb.build();
        let mut pool = Pool::new(16);

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
        let mut last = gb.op(Add, (ha, hb));
        for _ in 1..active {
            last = gb.op(Add, (last, ha));
        }

        // Inactive chain, fed by c, d (never set) -> never in the dirty cone.
        let inactive = total - active;
        if inactive > 0 {
            let mut prev = gb.op(Add, (hc, hd));
            for _ in 1..inactive {
                prev = gb.op(Add, (prev, hc));
            }
        }

        let mut g = gb.build();
        let mut pool = Pool::new(16);

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
    let _ = (0..K).map(|_| gb.op(Heavy(ITERS), src)).collect::<Vec<_>>();
    let mut g = gb.build();
    let mut pool = Pool::new(16);
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

// -- complex mesh (large, non-trivial topology, light workload) -----------

/// Two local predecessors and one long-range one.
fn preds(layer: usize, j: usize, n: usize) -> [usize; 3] {
    [j, (j + 1) % n, (j * 13 + layer * 7 + 1) % n]
}

/// Stateless unary map `x -> f(x)`; `passthrough` and `compute` are identical
/// (it recomputes from the input every generation, gate or no gate).
struct UnaryMap(fn(f64) -> f64);
impl Operator for UnaryMap {
    type Inputs = Port<Val<f64>>;
    type Outputs = Port<Val<f64>>;
    type State = fn(f64) -> f64;
    type Context = ();
    fn init(self, _: f64) -> Self::State {
        self.0
    }
    fn reset<'a, 'b: 'a>(x: f64, f: &'b mut Self::State) -> f64 {
        f(x)
    }
    fn compute<'a, 'b: 'a>(inputs: f64, f: &'b mut Self::State, _: &()) -> f64 {
        Self::reset(inputs, f)
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
    fn init(self, _: &[f64]) {}
    fn reset<'a, 'b: 'a>(xs: &'a [f64], _: &'b mut ()) -> f64 {
        xs.iter().sum()
    }
    fn compute<'a, 'b: 'a>(inputs: &'a [f64], state: &'b mut (), _: &()) -> f64 {
        Self::reset(inputs, state)
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
    let aggs = layers.iter().map(|l| gb.op(SumAll, &l[..])).collect();
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

fn bench_mesh_fusion(c: &mut Criterion) {
    let (w, d) = (24usize, 24usize);
    let mut group = c.benchmark_group("mesh_fusion");

    for (threads, suffix) in [(16, ""), (0, "_1t")] {
        // The cell diamond `Add(Inc(w), Double(w))` fused into ONE scheduled
        // node.
        {
            let mut gb = Builder::new();
            let (src, aggs) = build(&mut gb, w, d, |gb, _, _, [a, b, c]| {
                let op = tradingflow::fuse!(|x: Port<Val<f64>>, y: Port<Val<f64>>, z: Port<Val<f64>>| -> Port<Val<f64>> {
                    let xy = Add @ (x, y);
                    let w = Add @ (xy, z);
                    let p = inc() @ w;
                    let q = double() @ w;
                    let r = Add @ (p, q);
                    r
                });
                gb.op(op, (a, b, c))
            });
            let mut g = gb.build();
            let mut pool = Pool::new(threads);
            let out = *aggs.last().unwrap();
            group.bench_function(format!("fused_{w}x{d}{suffix}"), |bencher| {
                bencher.iter(drive_mesh(&src, &mut g, &mut pool, out));
            });
        }

        // The same diamond as 5 scheduled nodes.
        {
            let mut gb = Builder::new();
            let (src, aggs) = build(&mut gb, w, d, |gb, _, _, [x, y, z]| {
                let xy = gb.op(Add, (x, y));
                let w = gb.op(Add, (xy, z));
                let p = gb.op(inc(), w);
                let q = gb.op(double(), w);
                gb.op(Add, (p, q))
            });
            let mut g = gb.build();
            let mut pool = Pool::new(threads);
            let out = *aggs.last().unwrap();
            group.bench_function(format!("unfused_{w}x{d}{suffix}"), |bencher| {
                bencher.iter(drive_mesh(&src, &mut g, &mut pool, out));
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_add,
    bench_direct_compute,
    bench_engine_operator,
    bench_engine_chain,
    bench_engine_sparse,
    bench_few_heavy,
    bench_mesh_fusion,
);
criterion_main!(benches);
