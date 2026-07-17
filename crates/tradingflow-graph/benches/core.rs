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

use tradingflow_graph::core::Pool;
use tradingflow_graph::typed::{Builder, Graph, Operator, Port, PortHandle, RefPort, Source};

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

/// An adder with two inputs.
struct Add;

impl Operator for Add {
    type Inputs = (Port<f64>, Port<f64>);
    type Outputs = Port<f64>;
    type Context = ();
    type State = f64; // holds the last sum, to re-emit by value on passthrough

    fn init(self) -> f64 {
        0.0
    }

    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, f64), (bool, f64)),
        _: &(),
        state: &'b mut f64,
        _: bool,
    ) -> (bool, f64) {
        *state = a + b;
        (true, *state)
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, f64), (bool, f64)),
        _: &(),
        state: &'b f64,
    ) -> (bool, f64) {
        (false, *state)
    }
}

/// A CPU-heavy single-input kernel; `iters` lives in `State`.
struct Heavy(usize);

impl Operator for Heavy {
    type Inputs = Port<f64>;
    type Outputs = Port<()>;
    type Context = ();
    type State = usize;

    fn init(self) -> usize {
        self.0
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, f64),
        _: &(),
        state: &'b mut usize,
        _: bool,
    ) -> (bool, ()) {
        heavy_work(x, *state);
        (true, ())
    }

    fn passthrough<'a, 'b: 'a>(_: (bool, f64), _: &(), _: &'b usize) -> (bool, ()) {
        (false, ())
    }
}

/// Appends each input to a growing `Vec`, mutated in place (state owns it).
struct Record;

impl Operator for Record {
    type Inputs = Port<f64>; // scalar in by value, `Vec` out by reference
    type Outputs = RefPort<Vec<f64>>;
    type Context = ();
    type State = Vec<f64>;

    fn init(self) -> Vec<f64> {
        Vec::new()
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, f64),
        _: &(),
        state: &'b mut Vec<f64>,
        _: bool,
    ) -> (bool, &'a Vec<f64>) {
        if state.len() >= SERIES_LEN {
            black_box(state.last());
            state.clear();
        }
        state.push(x);
        (true, &*state)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, f64),
        _: &(),
        state: &'b Vec<f64>,
    ) -> (bool, &'a Vec<f64>) {
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
    let mut gb = Builder::new(());
    let (ha_cell, ha) = gb.source(Source::new(0.0));
    let (hb_cell, hb) = gb.source(Source::new(0.0));
    let _ = gb.segment(Add, (ha, hb));
    let mut g = gb.build();
    let mut pool = pool();

    c.bench_function("engine_segment", |bencher| {
        bencher.iter(|| {
            *g.state_mut(ha_cell) = a[i];
            *g.state_mut(hb_cell) = b[i];
            g.stabilize(&mut pool);
            i = (i + 1) & DATA_MASK;
        });
    });
}

fn bench_engine_segment_series(c: &mut Criterion) {
    let (a, b) = make_data();
    let mut i = 0;
    let mut gb = Builder::new(());
    let (ha_cell, ha) = gb.source(Source::new(0.0));
    let (hb_cell, hb) = gb.source(Source::new(0.0));
    let sum = gb.segment(Add, (ha, hb));
    let _ = gb.segment(Record, sum);
    let mut g = gb.build();
    let mut pool = pool();

    c.bench_function("engine_segment_series", |bencher| {
        bencher.iter(|| {
            *g.state_mut(ha_cell) = a[i];
            *g.state_mut(hb_cell) = b[i];
            g.stabilize(&mut pool);
            i = (i + 1) & DATA_MASK;
        });
    });
}

fn bench_engine_chain(c: &mut Criterion) {
    for depth in [1usize, 5, 10] {
        let (a, b) = make_data();
        let mut i = 0;
        let mut gb = Builder::new(());
        let (ha_cell, ha) = gb.source(Source::new(0.0));
        let (hb_cell, hb) = gb.source(Source::new(0.0));
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
                g.stabilize(&mut pool);
                i = (i + 1) & DATA_MASK;
            });
        });
    }
}

fn bench_engine_sparse(c: &mut Criterion) {
    for (total, active) in [(100usize, 5usize), (1000, 5)] {
        let (a, b) = make_data();
        let mut i = 0;
        let mut gb = Builder::new(());
        let (ha_cell, ha) = gb.source(Source::new(0.0));
        let (hb_cell, hb) = gb.source(Source::new(0.0));
        let (_, hc) = gb.source(Source::new(0.0));
        let (_, hd) = gb.source(Source::new(0.0));

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
                    g.stabilize(&mut pool);
                    i = (i + 1) & DATA_MASK;
                });
            },
        );
    }
}

fn bench_few_heavy(c: &mut Criterion) {
    const K: usize = 4;
    const ITERS: usize = 1_000_000;

    let mut gb = Builder::new(());
    let (src_cell, src) = gb.source(Source::new(1.0));
    let _: Vec<PortHandle<Port<()>>> = (0..K).map(|_| gb.segment(Heavy(ITERS), src)).collect();
    let mut g = gb.build();
    let mut pool = pool();
    let mut group = c.benchmark_group("few_heavy");

    group.bench_function("engine_parallel", |bencher| {
        let mut x = 0.0f64;
        bencher.iter(|| {
            x += 1.0;
            *g.state_mut(src_cell) = x;
            g.stabilize(&mut pool);
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
//
// A wide local+long-range mesh of 3-input `Work` nodes (per-node workload via
// `iters_of`, a few deliberately heavy) plus a full-layer `SumAll` per layer --
// the integration tests' graph, here driven for throughput. `bench_mesh`
// stabilizes the whole graph per generation; `bench_mesh_fusion` builds the same
// topology two ways -- each cell's `Add(Inc(w), Double(w))` diamond as one fused
// node vs. four separate ones -- isolating the per-node scheduling overhead that
// fusion removes.
mod mesh {
    use tradingflow_graph::typed::{
        Builder, NodeHandle, PortHandle, RefPort, RefPorts, RefSource, Segment,
    };

    /// A pokeable handle to one mesh source (a by-reference `i64`, since sources
    /// feed the per-layer `SumAll`'s `RefPorts`).
    pub type Src = NodeHandle<RefSource<i64, ()>>;

    pub const MODULUS: i64 = 1_000_000_007;

    /// `iters` rounds of an LCG, folded mod `MODULUS` so node values (and thus
    /// full-layer sums) stay bounded.
    pub fn lcg(seed: i64, iters: u32) -> i64 {
        let mut h = seed.rem_euclid(MODULUS);
        for _ in 0..iters {
            h = h.wrapping_mul(48271).wrapping_add(1).rem_euclid(MODULUS);
        }
        h
    }

    pub fn work(a: i64, b: i64, c: i64, iters: u32) -> i64 {
        let seed = a
            .wrapping_mul(3)
            .wrapping_add(b.wrapping_mul(5))
            .wrapping_add(c.wrapping_mul(7));
        lcg(seed, iters)
    }

    /// Two local predecessors and one long-range one.
    pub fn preds(layer: usize, j: usize, n: usize) -> [usize; 3] {
        [j, (j + 1) % n, (j * 13 + layer * 7 + 1) % n]
    }

    /// Mostly light (0..18 rounds), ~1/16 deliberately heavy.
    pub fn iters_of(layer: usize, j: usize) -> u32 {
        let base = ((layer * 31 + j * 17) % 19) as u32;
        if (layer + j).is_multiple_of(16) {
            base * 40 + 200
        } else {
            base
        }
    }

    pub struct Work {
        pub iters: u32,
    }
    impl Segment for Work {
        type Inputs = (RefPort<i64>, RefPort<i64>, RefPort<i64>);
        type Outputs = RefPort<i64>;
        type State = (u32, i64); // (iters, output)
        type Context = ();
        fn init(self) -> Self::State {
            (self.iters, 0)
        }
        fn compute<'a, 'b: 'a>(
            ((_, a), (_, b), (_, c)): ((bool, &'a i64), (bool, &'a i64), (bool, &'a i64)),
            _: &(),
            (iters, out): &'b mut Self::State,
            _: bool,
        ) -> (bool, &'a i64) {
            *out = work(*a, *b, *c, *iters);
            (true, &*out)
        }
    }

    pub struct Inc;
    impl Segment for Inc {
        type Inputs = RefPort<i64>;
        type Outputs = RefPort<i64>;
        type State = i64;
        type Context = ();
        fn init(self) -> i64 {
            0
        }
        fn compute<'a, 'b: 'a>(
            (_, x): (bool, &'a i64),
            _: &(),
            state: &'b mut i64,
            _: bool,
        ) -> (bool, &'a i64) {
            *state = x + 1;
            (true, &*state)
        }
    }

    pub struct Double;
    impl Segment for Double {
        type Inputs = RefPort<i64>;
        type Outputs = RefPort<i64>;
        type State = i64;
        type Context = ();
        fn init(self) -> i64 {
            0
        }
        fn compute<'a, 'b: 'a>(
            (_, x): (bool, &'a i64),
            _: &(),
            state: &'b mut i64,
            _: bool,
        ) -> (bool, &'a i64) {
            *state = x * 2;
            (true, &*state)
        }
    }

    pub struct Add;
    impl Segment for Add {
        type Inputs = (RefPort<i64>, RefPort<i64>);
        type Outputs = RefPort<i64>;
        type State = i64;
        type Context = ();
        fn init(self) -> i64 {
            0
        }
        fn compute<'a, 'b: 'a>(
            ((_, a), (_, b)): ((bool, &'a i64), (bool, &'a i64)),
            _: &(),
            state: &'b mut i64,
            _: bool,
        ) -> (bool, &'a i64) {
            *state = a + b;
            (true, &*state)
        }
    }

    pub struct SumAll;
    impl Segment for SumAll {
        type Inputs = RefPorts<i64>;
        type Outputs = RefPort<i64>;
        type State = i64;
        type Context = ();
        fn init(self) -> i64 {
            0
        }
        fn compute<'a, 'b: 'a>(
            (_, xs): (&'a [bool], &'a [&'a i64]),
            _: &(),
            state: &'b mut i64,
            _: bool,
        ) -> (bool, &'a i64) {
            *state = xs.iter().map(|&v| *v).sum();
            (true, &*state)
        }
    }

    /// Build the `n` x `depth` mesh, delegating each cell's construction to
    /// `cell` (a plain or fused node of the same 3-in/1-out interface), plus a
    /// full-layer `SumAll` aggregate per layer. Returns the sources (to poke)
    /// and the aggregates (the last of which transitively depends on everything).
    pub fn build(
        gb: &mut Builder<()>,
        n: usize,
        depth: usize,
        mut cell: impl FnMut(
            &mut Builder<()>,
            usize,
            usize,
            [PortHandle<RefPort<i64>>; 3],
        ) -> PortHandle<RefPort<i64>>,
    ) -> (Vec<Src>, Vec<PortHandle<RefPort<i64>>>) {
        let (src, wires): (Vec<Src>, Vec<_>) = (0..n)
            .map(|j| gb.source(RefSource::new(j as i64 + 1)))
            .unzip();
        let mut layers: Vec<Vec<PortHandle<RefPort<i64>>>> = vec![wires];
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
}

fn drive_mesh<'a>(
    src: &'a [mesh::Src],
    g: &'a mut Graph<()>,
    pool: &'a mut Pool,
    _: PortHandle<RefPort<i64>>,
) -> impl FnMut() + 'a {
    let mut base = 0i64;
    move || {
        base += 1;
        for (j, &s) in src.iter().enumerate() {
            *g.state_mut(s) = base + j as i64;
        }
        g.stabilize(pool);
    }
}

fn bench_mesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh");
    for (w, d) in [(16usize, 16usize), (32, 32)] {
        let mut gb = Builder::new(());
        let (src, aggs) = mesh::build(&mut gb, w, d, |gb, l, j, [a, b, c]| {
            gb.segment(
                mesh::Work {
                    iters: mesh::iters_of(l, j),
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
        let mut gb = Builder::new(());
        let (src, aggs) = mesh::build(&mut gb, w, d, |gb, l, j, [a, b, c]| {
            let it = mesh::iters_of(l, j);
            let seg = tradingflow_graph::segment!(|x: RefPort<i64>, y: RefPort<i64>, z: RefPort<i64>| -> RefPort<i64> {
                let ww = mesh::Work { iters: it } @ (x, y, z);
                let p = mesh::Inc @ ww;
                let q = mesh::Double @ ww;
                let r = mesh::Add @ (p, q);
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
        let mut gb = Builder::new(());
        let (src, aggs) = mesh::build(&mut gb, w, d, |gb, l, j, [a, b, c]| {
            let w_h = gb.segment(
                mesh::Work {
                    iters: mesh::iters_of(l, j),
                },
                (a, b, c),
            );
            let p = gb.segment(mesh::Inc, w_h);
            let q = gb.segment(mesh::Double, w_h);
            gb.segment(mesh::Add, (p, q))
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
