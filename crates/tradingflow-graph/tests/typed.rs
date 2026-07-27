use std::thread;

use bumpalo::Bump;
use tradingflow_graph::cb::*;
use tradingflow_graph::pool::Pool;
use tradingflow_graph::typed::{
    Builder, Graph, NodeHandle, Pass, Port, PortHandle, Ports, Ref, Segment, Slice, Val,
};

fn pool() -> Pool {
    Pool::new(thread::available_parallelism().unwrap().get())
}

fn fresh(arena: &mut Bump) -> &Bump {
    arena.reset();
    arena
}

/// A pass-by-value source node.
struct Source<T>(T);
impl<T: Copy + Send + Sync + 'static> Segment for Source<T> {
    type Inputs = ();
    type Outputs = Port<Val<T>>;
    type Context = ();
    type State = T;
    fn init(self, _: ()) -> T {
        self.0
    }
    fn reset<'a, 'b: 'a>(_: (), state: &'b mut T) -> T {
        *state
    }
    fn compute<'a, 'b: 'a>(_: (), state: &'b mut T, _: &()) -> T {
        *state
    }
}

/// A pokeable **event** source: each poke is one event (`Some`), and the wire
/// resets to the quiescent `None` in between, so a downstream consumer can
/// tell "a value arrived this generation" from "the same value is still here".
struct EventSource<T>(T);
impl<T: Copy + Send + Sync + 'static> Segment for EventSource<T> {
    type Inputs = ();
    type Outputs = Port<Val<Option<T>>>;
    type Context = ();
    type State = T;
    fn init(self, _: ()) -> T {
        self.0
    }
    fn reset<'a, 'b: 'a>(_: (), _: &'b mut T) -> Option<T> {
        None
    }
    fn compute<'a, 'b: 'a>(_: (), state: &'b mut T, _: &()) -> Option<T> {
        Some(*state)
    }
}

/// Maps an event through `x -> x + 1`, preserving quiescence — the event
/// analogue of [`inc`].
struct IncEvent;
impl Segment for IncEvent {
    type Inputs = Port<Val<Option<i64>>>;
    type Outputs = Port<Val<Option<i64>>>;
    type Context = ();
    type State = ();
    fn init(self, _: Option<i64>) {}
    fn reset<'a, 'b: 'a>(_: Option<i64>, _: &'b mut ()) -> Option<i64> {
        None
    }
    fn compute<'a, 'b: 'a>(x: Option<i64>, _: &'b mut (), _: &()) -> Option<i64> {
        x.map(|v| v + 1)
    }
}

/// A pass-by-reference source node.
struct RefSource<T>(T);
impl<T: Send + Sync + 'static> Segment for RefSource<T> {
    type Inputs = ();
    type Outputs = Port<Ref<T>>;
    type Context = ();
    type State = T;
    fn init(self, _: ()) -> T {
        self.0
    }
    fn reset<'a, 'b: 'a>(_: (), state: &'b mut T) -> &'a T {
        state
    }
    fn compute<'a, 'b: 'a>(_: (), state: &'b mut T, _: &()) -> &'a T {
        state
    }
}

/// A pass-by-slice source node.
struct SliceSource<T>(Box<[T]>);
impl<T: Send + Sync + 'static> Segment for SliceSource<T> {
    type Inputs = ();
    type Outputs = Port<Slice<T>>;
    type Context = ();
    type State = Box<[T]>;
    fn init(self, _: ()) -> Box<[T]> {
        self.0
    }
    fn reset<'a, 'b: 'a>(_: (), state: &'b mut Box<[T]>) -> &'a [T] {
        state
    }
    fn compute<'a, 'b: 'a>(_: (), state: &'b mut Box<[T]>, _: &()) -> &'a [T] {
        state
    }
}

/// Stateless unary map `x -> f(x)`; `passthrough` and `compute` are identical
/// (it recomputes from the input every generation, gate or no gate).
struct UnaryMap(fn(i64) -> i64);
impl Segment for UnaryMap {
    type Inputs = Port<Val<i64>>;
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = fn(i64) -> i64;
    fn init(self, _: i64) -> Self::State {
        self.0
    }
    fn reset<'a, 'b: 'a>(x: i64, f: &'b mut Self::State) -> i64 {
        f(x)
    }
    fn compute<'a, 'b: 'a>(inputs: i64, f: &'b mut Self::State, _: &()) -> i64 {
        Self::reset(inputs, f)
    }
}
/// Stateless binary map `(a, b) -> f(a, b)`.
struct BinaryMap(fn(i64, i64) -> i64);
impl Segment for BinaryMap {
    type Inputs = (Port<Val<i64>>, Port<Val<i64>>);
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = fn(i64, i64) -> i64;
    fn init(self, _: (i64, i64)) -> Self::State {
        self.0
    }
    fn reset<'a, 'b: 'a>((a, b): (i64, i64), f: &'b mut Self::State) -> i64 {
        f(a, b)
    }
    fn compute<'a, 'b: 'a>(inputs: (i64, i64), f: &'b mut Self::State, _: &()) -> i64 {
        Self::reset(inputs, f)
    }
}
fn inc() -> UnaryMap {
    UnaryMap(|x| x + 1)
}
fn double() -> UnaryMap {
    UnaryMap(|x| x * 2)
}
fn add() -> BinaryMap {
    BinaryMap(|a, b| a + b)
}

/// Sum over a runtime-sized list of inputs.
struct SumAll;
impl Segment for SumAll {
    type Inputs = Ports<Val<i64>>;
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = ();
    fn init(self, _: &[i64]) {}
    fn reset<'a, 'b: 'a>(xs: &'a [i64], _: &'b mut ()) -> i64 {
        xs.iter().sum()
    }
    fn compute<'a, 'b: 'a>(inputs: &'a [i64], state: &'b mut (), _: &()) -> i64 {
        Self::reset(inputs, state)
    }
}

/// Output is `1` iff input 0 carried an event this generation, else `0`.
/// Under value-level event semantics the presence bit lives *in* the payload,
/// so input 0 is an `Option` event port rather than a flagged `i64`.
struct DidFirstNotify;
impl Segment for DidFirstNotify {
    type Inputs = (Port<Val<Option<i64>>>, Port<Val<i64>>);
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = i64;
    fn init(self, _: (Option<i64>, i64)) -> i64 {
        0
    }
    fn reset<'a, 'b: 'a>(_: (Option<i64>, i64), state: &'b mut i64) -> i64 {
        *state
    }
    fn compute<'a, 'b: 'a>((e0, _): (Option<i64>, i64), state: &'b mut i64, _: &()) -> i64 {
        *state = i64::from(e0.is_some());
        *state
    }
}

/// Two outputs from one segment: `(a, b) -> (a + b, a - b)`.
struct AddSub;
impl Segment for AddSub {
    type Inputs = (Port<Val<i64>>, Port<Val<i64>>);
    type Outputs = (Port<Val<i64>>, Port<Val<i64>>);
    type Context = ();
    type State = ();
    fn init(self, _: (i64, i64)) {}
    fn reset<'a, 'b: 'a>((a, b): (i64, i64), _: &'b mut ()) -> (i64, i64) {
        ((a + b), (a - b))
    }
    fn compute<'a, 'b: 'a>(inputs: (i64, i64), state: &'b mut (), _: &()) -> (i64, i64) {
        Self::reset(inputs, state)
    }
}

/// A runtime-sized output list: `x -> [x, x+1, x+2]` (fixed arity 3). Both
/// the values and the output planes live in the node's arena.
struct Fanout3;
impl Segment for Fanout3 {
    type Inputs = Port<Val<i64>>;
    type Outputs = Ports<Val<i64>>;
    type Context = ();
    type State = Bump;
    fn init(self, _: i64) -> Self::State {
        Bump::new()
    }
    fn reset<'a, 'b: 'a>(x: i64, arena: &'b mut Self::State) -> &'a [i64] {
        let a = fresh(arena);
        let vals: &[i64] = a.alloc_slice_fill_iter((0..3usize).map(|i| x + i as i64));
        vals
    }
    fn compute<'a, 'b: 'a>(inputs: i64, arena: &'b mut Self::State, _: &()) -> &'a [i64] {
        Self::reset(inputs, arena)
    }
}

// ===== stateful / value-cutoff behaviors =====

/// `|x|` as a change-cutoff: it emits an *event* (`Some`) only when the value
/// actually changes, and the quiescent `None` otherwise. The cutoff now lives
/// in the payload rather than in a side-channel flag, so `reset` emits `None`.
struct Abs;
impl Segment for Abs {
    type Inputs = Port<Val<i64>>;
    type Outputs = Port<Val<Option<i64>>>;
    type Context = ();
    type State = i64;
    fn init(self, x: i64) -> i64 {
        x.abs()
    }
    fn reset<'a, 'b: 'a>(_: i64, _: &'b mut i64) -> Option<i64> {
        None
    }
    fn compute<'a, 'b: 'a>(x: i64, state: &'b mut i64, _: &()) -> Option<i64> {
        let new = x.abs();
        let changed = new != *state;
        *state = new;
        changed.then_some(new)
    }
}

/// Counts the generations in which its input carried an event: it must NOT
/// advance on a quiescent (`None`) input. Doubling as a change-counter, it
/// reports how many times an upstream value-cutoff producer actually fired.
/// Its own output is a *state* (the running count), so `reset` re-lends it.
struct GatedCounter;
impl Segment for GatedCounter {
    type Inputs = Port<Val<Option<i64>>>;
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = i64;
    fn init(self, _: Option<i64>) -> i64 {
        0
    }
    fn compute<'a, 'b: 'a>(x: Option<i64>, state: &'b mut i64, _: &()) -> i64 {
        if x.is_some() {
            *state += 1;
        }
        *state
    }
    fn reset<'a, 'b: 'a>(_: Option<i64>, state: &'b mut i64) -> i64 {
        *state
    }
}

/// Writes input into a 4-element buffer, in place (allocated once).
struct BufWrite;
impl Segment for BufWrite {
    type Inputs = Port<Val<f64>>;
    type Outputs = Port<Ref<Vec<f64>>>;
    type Context = ();
    type State = Vec<f64>;
    fn init(self, x: f64) -> Vec<f64> {
        let mut v = vec![0.0; 4];
        v[0] = x;
        v
    }
    fn reset<'a, 'b: 'a>(_: f64, state: &'b mut Vec<f64>) -> &'a Vec<f64> {
        state
    }
    fn compute<'a, 'b: 'a>(x: f64, state: &'b mut Vec<f64>, _: &()) -> &'a Vec<f64> {
        state[0] = x;
        state
    }
}

/// Running sum: seeds on the build call, adds its input on every generation the
/// input notifies. As a stateful accumulator it must NOT re-add an un-notified
/// (unchanged) input, so it is an `Operator`; `passthrough` retains the sum.
struct Fold;
impl Segment for Fold {
    type Inputs = Port<Val<i64>>;
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = i64;
    fn init(self, x: i64) -> i64 {
        x
    }
    fn compute<'a, 'b: 'a>(x: i64, state: &'b mut i64, _: &()) -> i64 {
        *state += x;
        *state
    }
    fn reset<'a, 'b: 'a>(_: i64, state: &'b mut i64) -> i64 {
        *state
    }
}

/// A typed DAG fused into one operator body: `x=a+b; y=x*c; z=x+a; out=y*z`.
struct FusedDag;
impl Segment for FusedDag {
    type Inputs = (Port<Val<f64>>, Port<Val<f64>>, Port<Val<f64>>);
    type Outputs = Port<Val<f64>>;
    type Context = ();
    type State = ();
    fn init(self, _: (f64, f64, f64)) {}
    fn reset<'a, 'b: 'a>((a, b, c): (f64, f64, f64), _: &'b mut ()) -> f64 {
        let x = a + b;
        let y = x * c;
        let z = x + a;
        y * z
    }
    fn compute<'a, 'b: 'a>(inputs: (f64, f64, f64), state: &'b mut (), _: &()) -> f64 {
        Self::reset(inputs, state)
    }
}

/// Heterogeneous inputs `(f64, i32)`, two outputs `(sum, calls)`, carried counter.
struct HeteroState;
impl Segment for HeteroState {
    type Inputs = (Port<Val<f64>>, Port<Val<i32>>);
    type Outputs = (Port<Val<f64>>, Port<Val<u64>>);
    type Context = ();
    type State = u64;
    fn init(self, _: (f64, i32)) -> u64 {
        0
    }
    fn reset<'a, 'b: 'a>((a, b): (f64, i32), calls: &'b mut u64) -> (f64, u64) {
        (a + b as f64, *calls)
    }
    fn compute<'a, 'b: 'a>((a, b): (f64, i32), calls: &'b mut u64, _: &()) -> (f64, u64) {
        *calls += 1;
        (a + b as f64, *calls)
    }
}

/// Homogeneous slice input, two outputs: `(sum, max)`.
struct SumMax;
impl Segment for SumMax {
    type Inputs = Ports<Val<f64>>;
    type Outputs = (Port<Val<f64>>, Port<Val<f64>>);
    type Context = ();
    type State = ();
    fn init(self, _: &[f64]) {}
    fn reset<'a, 'b: 'a>(xs: &'a [f64], _: &'b mut ()) -> (f64, f64) {
        let sum = xs.iter().sum();
        let max = xs.iter().copied().fold(f64::MIN, f64::max);
        ((sum), (max))
    }
    fn compute<'a, 'b: 'a>(inputs: &'a [f64], state: &'b mut (), _: &()) -> (f64, f64) {
        Self::reset(inputs, state)
    }
}

/// Dot product over two zipped `Ports`. (`Ports` is a *leaf*: "many of pairs"
/// is spelled as a pair of `Ports`, zipped in `compute`.)
struct DotPairs;
impl Segment for DotPairs {
    type Inputs = (Ports<Val<i64>>, Ports<Val<i64>>);
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = ();
    fn init(self, _: (&[i64], &[i64])) {}
    fn reset<'a, 'b: 'a>((xs, ys): (&'a [i64], &'a [i64]), _: &'b mut ()) -> i64 {
        xs.iter().zip(ys).map(|(&x, &y)| x * y).sum()
    }
    fn compute<'a, 'b: 'a>(inputs: (&'a [i64], &'a [i64]), state: &'b mut (), _: &()) -> i64 {
        Self::reset(inputs, state)
    }
}

/// Two variadic input groups around a scalar: `sum(a)*k + sum(c)`.
struct TwoArrays;
impl Segment for TwoArrays {
    type Inputs = (Ports<Val<i64>>, Port<Val<i64>>, Ports<Val<i64>>);
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = ();
    fn init(self, _: (&[i64], i64, &[i64])) {}
    fn reset<'a, 'b: 'a>((a, k, c): (&'a [i64], i64, &'a [i64]), _: &'b mut ()) -> i64 {
        a.iter().sum::<i64>() * k + c.iter().sum::<i64>()
    }
    fn compute<'a, 'b: 'a>(inputs: (&'a [i64], i64, &'a [i64]), state: &'b mut (), _: &()) -> i64 {
        Self::reset(inputs, state)
    }
}

/// Two variadic OUTPUT groups: `x -> (k copies of x, k copies of -x)`. ONE
/// arena serves both groups' values and all four planes.
struct SplitTwo(usize);
impl Segment for SplitTwo {
    type Inputs = Port<Val<i64>>;
    type Outputs = (Ports<Val<i64>>, Ports<Val<i64>>);
    type Context = ();
    type State = (usize, Bump); // (k, per-gen storage)
    fn init(self, _: i64) -> Self::State {
        (self.0, Bump::new())
    }
    fn reset<'a, 'b: 'a>(x: i64, (k, arena): &'b mut Self::State) -> (&'a [i64], &'a [i64]) {
        let a = fresh(arena);
        let pos: &[i64] = a.alloc_slice_fill_iter((0..*k).map(|_| x));
        let neg: &[i64] = a.alloc_slice_fill_iter((0..*k).map(|_| -x));
        (pos, neg)
    }
    fn compute<'a, 'b: 'a>(
        inputs: i64,
        state: &'b mut Self::State,
        _: &(),
    ) -> (&'a [i64], &'a [i64]) {
        Self::reset(inputs, state)
    }
}

/// CAPABILITY: output refs point *into the inputs*. Forwards a fixed-width
/// window `[start, start+W)` of the input array as a plain subslice (zero
/// copy, no state buffer); the window's addresses move as `start` changes.
struct Window(usize); // W
impl Segment for Window {
    type Inputs = (Ports<Val<i64>>, Port<Val<usize>>);
    type Outputs = Ports<Val<i64>>;
    type Context = ();
    type State = usize; // W
    fn init(self, _: (&[i64], usize)) -> usize {
        self.0
    }
    fn reset<'a, 'b: 'a>((arr, start): (&'a [i64], usize), w: &'b mut usize) -> &'a [i64] {
        &arr[start..start + *w]
    }
    fn compute<'a, 'b: 'a>(inputs: (&'a [i64], usize), w: &'b mut usize, _: &()) -> &'a [i64] {
        Self::reset(inputs, w)
    }
}

/// CAPABILITY: input-sized init. The output buffer's size depends on the
/// input arity, which `init` sees at build: it allocates a buffer sized from
/// the input and every call fills it in place, returning refs into it.
struct MapScale(i64); // k
impl Segment for MapScale {
    type Inputs = Ports<Val<i64>>;
    type Outputs = Ports<Val<i64>>;
    type Context = ();
    type State = (i64, Vec<i64>, Vec<bool>); // (k, output buffer, flags)
    fn init(self, xs: &[i64]) -> Self::State {
        let k = self.0;
        (k, xs.iter().map(|&e| e * k).collect(), vec![true; xs.len()])
    }
    fn reset<'a, 'b: 'a>(_: &'a [i64], (_, vals, _flags): &'b mut Self::State) -> &'a [i64] {
        vals
    }
    fn compute<'a, 'b: 'a>(
        xs: &'a [i64],
        (k, vals, _flags): &'b mut Self::State,
        _: &(),
    ) -> &'a [i64] {
        for (slot, &e) in vals.iter_mut().zip(xs.iter()) {
            *slot = e * *k;
        }
        vals
    }
}

// ===== tests =====

#[test]
fn diamond_recomputes_on_source_change() {
    // S -> A=S+1, (S,A) -> D=S+A.
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(1));
    let a = b.segment(inc(), s);
    let d = b.segment(add(), (s, a));
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(s), 1);
    assert_eq!(g.view(a), 2);
    assert_eq!(g.view(d), 3);

    *g.state_mut(s_cell) = 5;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(a), 6);
    assert_eq!(g.view(d), 11);
}

#[test]
fn an_event_input_carries_its_event_for_exactly_one_generation() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(EventSource(0));
    let (s2_cell, s2) = b.source(Source(0));
    let a = b.segment(DidFirstNotify, (s, s2));
    let mut g = b.build();
    let mut pool = pool();

    *g.state_mut(s_cell) = 1;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(a), 1);

    *g.state_mut(s2_cell) = 1;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(a), 0);
}

#[test]
fn multi_output_tuple() {
    let mut b = Builder::new();
    let (s1_cell, s1) = b.source(Source(7));
    let (_, s2) = b.source(Source(3));
    let (sum, diff) = b.segment(AddSub, (s1, s2));
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(sum), 10);
    assert_eq!(g.view(diff), 4);

    *g.state_mut(s1_cell) = 20;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(sum), 23);
    assert_eq!(g.view(diff), 17);
}

#[test]
fn independent_branch_not_recomputed() {
    let mut b = Builder::new();
    let (s1_cell, s1) = b.source(EventSource(0i64));
    let (_, s2) = b.source(EventSource(0i64));
    let a = b.segment(GatedCounter, s1);
    let c = b.segment(GatedCounter, s2);
    let mut g = b.build();
    let mut pool = pool();

    *g.state_mut(s1_cell) = 9;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(a), 1); // branch 1 ran once
    assert_eq!(g.view(c), 0); // branch 2 never recomputed (outside the cone)
}

#[test]
fn value_cutoff_via_notify() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(0i64));
    let abs = b.segment(Abs, s);
    let c = b.segment(GatedCounter, abs);
    let mut g = b.build();
    let mut pool = pool();

    *g.state_mut(s_cell) = 3;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(c), 1); // 0 -> 3: changed

    *g.state_mut(s_cell) = -3;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(c), 1); // |−3| == |3|: no notify, c gated out

    *g.state_mut(s_cell) = 5;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(c), 2); // 3 -> 5: changed
}

#[test]
fn in_place_buffer_is_stable() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(0.0f64));
    let buf = b.segment(BufWrite, s);
    let mut g = b.build();
    let mut pool = pool();

    let p0 = g.view(buf).as_ptr();
    *g.state_mut(s_cell) = 1.0;
    g.stabilize(&mut pool, &());
    let p1 = g.view(buf).as_ptr();
    *g.state_mut(s_cell) = 2.0;
    g.stabilize(&mut pool, &());
    let p2 = g.view(buf).as_ptr();

    assert_eq!(p0, p1, "buffer reused in place");
    assert_eq!(p1, p2, "buffer reused in place");
    assert_eq!(g.view(buf)[0], 2.0);
}

#[test]
fn stateful_fold_accumulates() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(0i64));
    let acc = b.segment(Fold, s);
    let mut g = b.build();
    let mut pool = pool();

    *g.state_mut(s_cell) = 5;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(acc), 5);
    *g.state_mut(s_cell) = 3;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(acc), 8);
}

#[test]
fn fused_subgraph_in_one_segment() {
    let mut b = Builder::new();
    let (a_cell, a) = b.source(Source(2.0f64));
    let (_, bb) = b.source(Source(3.0f64));
    let (_, c) = b.source(Source(5.0f64));
    let out = b.segment(FusedDag, (a, bb, c));
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(out), 175.0); // x=5, y=25, z=7
    *g.state_mut(a_cell) = 1.0;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out), 100.0); // x=4, y=20, z=5
}

#[test]
fn both_outputs_feed_one_consumer() {
    let mut b = Builder::new();
    let (s1_cell, s1) = b.source(Source(7i64));
    let (_, s2) = b.source(Source(3i64));
    let (sum, diff) = b.segment(AddSub, (s1, s2));
    let out = b.segment(add(), (sum, diff)); // (a+b) + (a-b) = 2a
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(out), 14);
    *g.state_mut(s1_cell) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out), 20);
}

#[test]
fn multi_output_cross_port_ordering() {
    let mut b = Builder::new();
    let (_, s1) = b.source(Source(10i64));
    let (_, s2) = b.source(Source(4i64));
    let (sum, diff) = b.segment(AddSub, (s1, s2)); // sum=14, diff=6
    let doubled = b.segment(double(), sum); // 28
    let out = b.segment(add(), (doubled, diff)); // 28 + 6 = 34
    let g = b.build();
    assert_eq!(g.view(out), 34);
}

#[test]
fn heterogeneous_inputs_multi_output_with_state() {
    let mut b = Builder::new();
    let (a_cell, a) = b.source(Source(10.0f64));
    let (_, bb) = b.source(Source(3i32));
    let (sum, calls) = b.segment(HeteroState, (a, bb));
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(sum), 13.0);
    *g.state_mut(a_cell) = 20.0;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(sum), 23.0);
    assert_eq!(g.view(calls), 1); // state carried across gens
}

#[test]
fn slice_input_multi_output_sum_max() {
    let mut b = Builder::new();
    let handles: Vec<PortHandle<Val<f64>>> = [1.0, 5.0, 3.0, 2.0]
        .into_iter()
        .map(|v| b.source(Source(v)).1)
        .collect();
    let (sum, max) = b.segment(SumMax, &handles[..]);
    let g = b.build();

    assert_eq!(g.view(sum), 11.0);
    assert_eq!(g.view(max), 5.0);
}

#[test]
fn zipped_manys_dot_product() {
    let mut b = Builder::new();
    let (_, a0) = b.source(Source(2));
    let (_, b0) = b.source(Source(3));
    let (a1_cell, a1) = b.source(Source(4));
    let (_, b1) = b.source(Source(5));
    let out = b.segment(DotPairs, (&[a0, a1][..], &[b0, b1][..]));
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(out), 26); // 2*3 + 4*5

    *g.state_mut(a1_cell) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out), 56); // 6 + 10*5
}

#[test]
fn two_variadic_input_groups() {
    let mut b = Builder::new();
    let (a, ah): (Vec<_>, Vec<PortHandle<Val<i64>>>) =
        [1, 2, 3].into_iter().map(|v| b.source(Source(v))).unzip();
    let (_, k) = b.source(Source(10));
    let ch: Vec<PortHandle<Val<i64>>> = [4, 5].into_iter().map(|v| b.source(Source(v)).1).collect();
    let out = b.segment(TwoArrays, (&ah[..], k, &ch[..]));
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(out), (1 + 2 + 3) * 10 + (4 + 5)); // 69

    *g.state_mut(a[0]) = 11;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out), (11 + 2 + 3) * 10 + (4 + 5)); // 169
}

#[test]
fn two_variadic_output_groups() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(7));
    let (pos, neg) = b.segment(SplitTwo(3), s);
    assert_eq!(pos.len(), 3);
    assert_eq!(neg.len(), 3);
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(pos[0]), 7);
    assert_eq!(g.view(neg[0]), -7);

    *g.state_mut(s_cell) = 4;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(pos[2]), 4);
    assert_eq!(g.view(neg[1]), -4);
}

// ===== new-design capability tests =====

#[test]
fn window_forwards_input_refs() {
    // The window's outputs are references INTO the input array (no copy), and
    // their addresses move as `start` changes -- the dynamic-address case.
    let mut b = Builder::new();
    let (arr, arrh): (Vec<_>, Vec<PortHandle<Val<i64>>>) = [10, 20, 30, 40, 50]
        .into_iter()
        .map(|v| b.source(Source(v)))
        .unzip();
    let (start_cell, start) = b.source(Source(0usize));
    let win = b.segment(Window(2), (&arrh[..], start));
    assert_eq!(win.len(), 2);
    let mut g = b.build();
    let mut pool = pool();

    // window [0, 2) = [10, 20]
    assert_eq!(g.view(win[0]), 10);
    assert_eq!(g.view(win[1]), 20);

    // move the window: start=2 -> [30, 40] (the output pointers now target
    // different source cells)
    *g.state_mut(start_cell) = 2;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(win[0]), 30);
    assert_eq!(g.view(win[1]), 40);

    // change a source the window currently points at: arr[3]=99 -> [30, 99]
    *g.state_mut(arr[3]) = 99;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(win[0]), 30);
    assert_eq!(g.view(win[1]), 99);
}

#[test]
fn two_stage_init_allocates_from_input() {
    // The output buffer's size is not known until the inputs arrive, so it is
    // allocated by `init` (which sees them at build) and filled in place
    // thereafter.
    let mut b = Builder::new();
    let (arr, arrh): (Vec<_>, Vec<PortHandle<Val<i64>>>) = [1, 2, 3, 4]
        .into_iter()
        .map(|v| b.source(Source(v)))
        .unzip();
    let out = b.segment(MapScale(10), &arrh[..]);
    assert_eq!(out.len(), 4); // sized from the 4-element input at build
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(out[0]), 10);
    assert_eq!(g.view(out[2]), 30);

    *g.state_mut(arr[1]) = 5;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out[1]), 50); // 5 * 10, written into the buffer in place
}

/// Exposes refs INTO its input `Vec`'s heap buffer as a `Ports` (the
/// dangling-read repro): the output points into the source's buffer, so poking
/// the source to a fresh `Vec` frees the memory the output slots point at.
struct Spread;
impl Segment for Spread {
    type Inputs = Port<Ref<Vec<i64>>>;
    type Outputs = Ports<Val<i64>>;
    type Context = ();
    type State = Vec<bool>;
    fn init(self, v: &Vec<i64>) -> Self::State {
        vec![true; v.len()]
    }
    fn reset<'a, 'b: 'a>(v: &'a Vec<i64>, _flags: &'b mut Self::State) -> &'a [i64] {
        v
    }
    fn compute<'a, 'b: 'a>(inputs: &'a Vec<i64>, flags: &'b mut Self::State, _: &()) -> &'a [i64] {
        Self::reset(inputs, flags)
    }
}

#[test]
#[should_panic(expected = "stabilize")]
fn read_before_stabilize_after_poke_is_rejected() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(RefSource(vec![10i64, 20, 30]));
    let out = b.segment(Spread, s);
    let mut g = b.build();
    assert_eq!(g.view(out[0]), 10); // stabilized read is fine

    // Poke to a fresh, same-length buffer: the old buffer (which `out` points
    // into, from the build call) is freed. Reading `out` now -- before stabilize
    // -- would dereference the dangling pointer, so it must panic, not read UB.
    *g.state_mut(s_cell) = vec![11, 21, 31];
    let _ = g.view(out[0]); // <-- rejected: graph is unstabilized
}

/// A node that reallocates its output buffer and *then* panics at runtime; the
/// realloc frees the buffer its (build-set) output slots point into, so without
/// poisoning a later read would dangle.
struct ReallocPanic;
impl Segment for ReallocPanic {
    type Inputs = Port<Val<i64>>;
    type Outputs = Ports<Val<i64>>;
    type Context = ();
    type State = (Vec<i64>, Vec<bool>);
    fn init(self, x: i64) -> Self::State {
        (vec![x, x], vec![true, true])
    }
    fn reset<'a, 'b: 'a>(_: i64, (vals, _flags): &'b mut Self::State) -> &'a [i64] {
        vals
    }
    fn compute<'a, 'b: 'a>(x: i64, (vals, flags): &'b mut Self::State, _: &()) -> &'a [i64] {
        *flags = vec![true, true];
        *vals = vec![x, x]; // realloc: frees the buffer the output slots point into
        assert!(x >= 0, "negative input"); // panic *after* the realloc
        vals
    }
}

#[test]
fn realloc_then_panic_does_not_dangle() {
    // ReallocPanic reallocates its output buffer (freeing the one its slots
    // point into) and then panics, leaving `out` dangling into freed memory.
    // Poisoning must make the subsequent read panic, not dereference it.
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(1i64));
    let out = b.segment(ReallocPanic, s);
    let mut g = b.build();
    let mut pool = pool();
    assert_eq!(g.view(out[0]), 1); // build buffer = [1, 1]

    *g.state_mut(s_cell) = -1;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.stabilize(&mut pool, &())));
    assert!(r.is_err());

    // Poisoned: reading the now-dangling forwarded slot panics rather than UB.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = g.view(out[0]);
    }));
    assert!(r.is_err());
}

// ===== typed Arrow combinators (then / par / fanout) =====

#[test]
fn composed_series_then() {
    let mut gb = Builder::new();
    let (s_cell, s) = gb.source(Source(5));
    let stage = gb.segment(inc(), s);
    let sep = gb.segment(double(), stage); // separate nodes
    let (s2_cell, s2) = gb.source(Source(5));
    let comp = gb.segment(inc().then(double()), s2); // composed into ONE segment
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(sep), 12); // (5+1)*2
    assert_eq!(g.view(comp), 12);

    *g.state_mut(s_cell) = 10;
    *g.state_mut(s2_cell) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(sep), 22);
    assert_eq!(g.view(comp), 22);
}

#[test]
fn composed_par() {
    let mut gb = Builder::new();
    let (a_cell, a) = gb.source(Source(3));
    let (_, bh) = gb.source(Source(4));
    let (oa, ob) = gb.segment(inc().par(double()), (a, bh));
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(oa), 4); // 3+1
    assert_eq!(g.view(ob), 8); // 4*2

    // Change only `a`: Inc's branch reruns; Double recomputes the same 8 (b
    // unchanged), so ob is retained.
    *g.state_mut(a_cell) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(oa), 11);
    assert_eq!(g.view(ob), 8);
}

#[test]
fn composed_fanout_diamond() {
    // `Inc &&& Double` then `Add`: a -> (a+1, 2a) -> (a+1)+(2a) = 3a+1.
    let mut gb = Builder::new();
    let (s_cell, s) = gb.source(Source(5));
    let comp = gb.segment(inc().fork(double()).then(add()), s);
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(comp), 16); // 3*5 + 1

    *g.state_mut(s_cell) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(comp), 31); // 3*10 + 1
}

#[test]
fn composed_variadic_intermediate() {
    // `SumAll >>> Fanout3 >>> SumAll`: xs -> total; [total, total+1, total+2];
    // sum = 3*total + 3.
    let mut gb = Builder::new();
    let (xs, xh): (Vec<_>, Vec<PortHandle<Val<i64>>>) =
        [1, 2, 3].into_iter().map(|v| gb.source(Source(v))).unzip();
    let comp = gb.segment(SumAll.then(Fanout3).then(SumAll), &xh[..]);
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(comp), 21); // total=6 -> [6,7,8] -> 21

    *g.state_mut(xs[0]) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(comp), 48); // total=15 -> [15,16,17] -> 48
}

#[test]
fn composed_stateful_then() {
    let mut gb = Builder::new();
    let (s_cell, s) = gb.source(Source(5));
    let fold = gb.segment(Fold, s);
    let sep = gb.segment(double(), fold); // separate nodes
    let (s2_cell, s2) = gb.source(Source(5));
    let comp = gb.segment(Fold.then(double()), s2); // composed into ONE segment
    let mut g = gb.build();
    let mut pool = pool();

    for v in [3, 2, 7] {
        *g.state_mut(s_cell) = v;
        *g.state_mut(s2_cell) = v;
        g.stabilize(&mut pool, &());
        assert_eq!(
            g.view(comp),
            g.view(sep),
            "stateful compose diverged at {v}"
        );
    }
}

#[test]
fn composed_with_id() {
    let mut gb = Builder::new();
    let (s_cell, s) = gb.source(Source(5));
    let comp = gb.segment(inc().then(Id::default()).then(double()), s);
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(comp), 12); // (5+1)*2

    *g.state_mut(s_cell) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(comp), 22); // (10+1)*2
}

#[test]
fn operator_gate_blocks_unnotified_branch() {
    // `GatedCounter *** Inc`. Change only the second input: Inc reruns, but
    // GatedCounter's input carried no event, so its gate skips the increment.
    let mut gb = Builder::new();
    let (a_cell, a) = gb.source(EventSource(0));
    let (bh_cell, bh) = gb.source(Source(0));
    let (count, inc) = gb.segment(GatedCounter.par(inc()), (a, bh));
    let mut g = gb.build();
    let mut pool = pool();

    *g.state_mut(bh_cell) = 5; // change only b
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(inc), 6); // Inc ran: 5 + 1
    assert_eq!(g.view(count), 0); // GatedCounter gated out: counter unchanged

    *g.state_mut(a_cell) = 1; // now change a
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(count), 1); // GatedCounter ran once
}

#[test]
fn route_reorders_by_forwarding_refs() {
    // Stateless `Arr`: a pure ref-shuffle (here a swap). Outputs are refs
    // INTO the inputs, reordered -- no state, no copy. Standalone construction
    // needs the turbofish: nothing else pins T/U (`Values` is non-injective).
    let mut gb = Builder::new();
    let (s0_cell, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let swap = Arr::<(Port<Val<i64>>, Port<Val<i64>>), (Port<Val<i64>>, Port<Val<i64>>), _, _>::new(
        |(a, b), _| (b, a),
    );
    let (o0, o1) = gb.segment(swap, (s0, s1));
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(o0), 20); // forwards s1
    assert_eq!(g.view(o1), 10); // forwards s0

    *g.state_mut(s0_cell) = 99;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(o1), 99); // o1 still forwards s0
}

#[test]
fn hand_lowered_segment_chain_infers() {
    // The exact shape `segment!` emits -- every type except the seed `Id` and the
    // final `route` turbofish must be INFERRED (this is the empirical risk the
    // bind/route argument order exists to discharge). One fused node:
    //   c = a + b; d = c + 1; result (d, c).
    let seg = Id::<(Port<Val<i64>>, Port<Val<i64>>), _>::default()
        .bind(add(), |(a, b), _| (a, b))
        .bind(inc(), |(_, c), _| c)
        .route::<(Port<Val<i64>>, Port<Val<i64>>), _>(|((_, c), d), _| (d, c));

    let mut gb = Builder::new();
    let (s0_cell, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let (od, oc) = gb.segment(seg, (s0, s1));
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(oc), 30);
    assert_eq!(g.view(od), 31);

    *g.state_mut(s0_cell) = 11;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(oc), 31);
    assert_eq!(g.view(od), 32);
}

// ===== segment! notation =====

#[test]
fn segment_notation_diamond() {
    // Same graph as `hand_lowered_segment_chain_infers`, via the macro.
    let seg = tradingflow::segment!(@[tradingflow_graph::cb] |a: Port<Val<i64>>, b: Port<Val<i64>>| -> (Port<Val<i64>>, Port<Val<i64>>) {
        let c = add() @ (a, b);
        let d = inc() @ c;
        (d, c)
    });

    let mut gb = Builder::new();
    let (s0_cell, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let (od, oc) = gb.segment(seg, (s0, s1));
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(oc), 30);
    assert_eq!(g.view(od), 31);

    *g.state_mut(s0_cell) = 11;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(oc), 31);
    assert_eq!(g.view(od), 32);
}

#[test]
fn segment_notation_result_is_last_binding() {
    // The result is exactly the last binding: the closing `route` projects it
    // straight out of the environment.
    let seg = tradingflow::segment!(@[tradingflow_graph::cb] |a: Port<Val<i64>>, b: Port<Val<i64>>| -> Port<Val<i64>> {
        let c = add() @ (a, b);
        let d = inc() @ c;
        d
    });

    let mut gb = Builder::new();
    let (_, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let od = gb.segment(seg, (s0, s1));
    let g = gb.build();
    assert_eq!(g.view(od), 31);
}

#[test]
fn segment_notation_runtime_path_override() {
    // A leading `@[path]` overrides the `::tradingflow::graph::cb` the
    // expansion names — the hook a facade crate's `macro_rules!` wrapper uses
    // (passing `@[$crate::...]`) so its users need no direct facade dependency.
    // Here the override is an alias of the same module.
    use tradingflow_graph::cb as alt;
    let seg = tradingflow::segment!(@[alt] |a: Port<Val<i64>>, b: Port<Val<i64>>| -> Port<Val<i64>> {
        let c = add() @ (a, b);
        c
    });

    let mut gb = Builder::new();
    let (_, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let oc = gb.segment(seg, (s0, s1));
    let g = gb.build();
    assert_eq!(g.view(oc), 30);
}

#[test]
fn segment_notation_duplicates_reorders_and_drops() {
    // `a` feeds two statements AND the result; `b` is dropped entirely.
    let seg = tradingflow::segment!(@[tradingflow_graph::cb] |a: Port<Val<i64>>, b: Port<Val<i64>>| -> (Port<Val<i64>>, Port<Val<i64>>) {
        let s = add() @ (a, a);
        let t = add() @ (s, a);
        (t, s)
    });

    let mut gb = Builder::new();
    let (_, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let (ot, os) = gb.segment(seg, (s0, s1));
    let g = gb.build();
    assert_eq!(g.view(os), 20); // a + a
    assert_eq!(g.view(ot), 30); // s + a
}

#[test]
fn segment_notation_destructures_and_shadows() {
    // Destructure a two-output segment; shadow `p` to rebind it.
    let seg = tradingflow::segment!(@[tradingflow_graph::cb] |a: Port<Val<i64>>, b: Port<Val<i64>>| -> Port<Val<i64>> {
        let (p, q) = AddSub @ (a, b);
        let p = inc() @ p;
        let r = add() @ (p, q);
        r
    });

    let mut gb = Builder::new();
    let (_, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(4));
    let or = gb.segment(seg, (s0, s1));
    let g = gb.build();
    assert_eq!(g.view(or), 21); // (10+4)+1 + (10-4)
}

#[test]
fn segment_notation_ports_wire() {
    // A `Ports` wire rides the environment like any other.
    let seg = tradingflow::segment!(@[tradingflow_graph::cb] |xs: Ports<Val<i64>>| -> Port<Val<i64>> {
        let s = SumAll @ xs;
        let t = inc() @ s;
        t
    });

    let mut gb = Builder::new();
    let (s0_cell, s0) = gb.source(Source(1));
    let (_, s1) = gb.source(Source(2));
    let (_, s2) = gb.source(Source(3));
    let ot = gb.segment(seg, &[s0, s1, s2][..]);
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(ot), 7);

    *g.state_mut(s0_cell) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(ot), 16);
}

/// A gated `Ports` producer whose output view cannot be rebuilt from `&State`
/// alone -- yet both `compute` and `passthrough` re-derive it afresh from the
/// inputs and state each call, so the auto-gate drives it directly: `compute`
/// advances the counter and lights the notify plane, `passthrough` re-lends the
/// retained values with the plane cleared.
struct GatedFanout;
impl Segment for GatedFanout {
    type Inputs = Port<Val<Option<i64>>>;
    type Outputs = Ports<Val<i64>>;
    type Context = ();
    type State = (i64, Vec<i64>, Bump); // (compute count, values, planes)
    fn init(self, x: Option<i64>) -> Self::State {
        (0, vec![x.unwrap_or(0), 0], Bump::new())
    }
    fn reset<'a, 'b: 'a>(_: Option<i64>, (_, vals, arena): &'b mut Self::State) -> &'a [i64] {
        let _ = fresh(arena);
        vals
    }
    fn compute<'a, 'b: 'a>(
        x: Option<i64>,
        (count, vals, arena): &'b mut Self::State,
        _: &(),
    ) -> &'a [i64] {
        // The gate is explicit now: a quiescent input recomputes nothing.
        if let Some(x) = x {
            *count += 1;
            (vals[0], vals[1]) = (x, *count);
        }
        let _ = fresh(arena);
        vals
    }
}

#[test]
fn gated_ports_producer_rederives_each_generation() {
    // Src -> Abs (value cutoff, event out) -> GatedFanout (gates on the event,
    // Ports out). `out[1]` is the recompute count, so it is the probe itself.
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(5i64));
    let a = b.segment(Abs, s);
    let out = b.segment(GatedFanout, a);
    let mut g = b.build();
    let mut pool = pool();

    // The build render of an event producer is quiescent, so the fanout has
    // seen no event yet.
    assert_eq!(g.view(out[0]), 0);
    assert_eq!(g.view(out[1]), 0); // build call: count = 0

    *g.state_mut(s_cell) = -5; // |x| unchanged: Abs emits no event
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out[1]), 0); // gated: no recount

    *g.state_mut(s_cell) = 7; // |x| changes: Abs emits an event
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out[0]), 7);
    assert_eq!(g.view(out[1]), 1);
}

#[test]
fn segment_notation_pure_permutation() {
    // No statements at all: an annotated pure shuffle (swap + dup).
    let seg = tradingflow::segment!(@[tradingflow_graph::cb] |a: Port<Val<i64>>,
                                     b: Port<Val<i64>>|
     -> (Port<Val<i64>>, (Port<Val<i64>>, Port<Val<i64>>)) {
        (b, (a, a))
    });

    let mut gb = Builder::new();
    let (_, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let (ob, (oa0, oa1)) = gb.segment(seg, (s0, s1));
    let g = gb.build();
    assert_eq!(g.view(ob), 20);
    assert_eq!(g.view(oa0), 10);
    assert_eq!(g.view(oa1), 10);
}

#[test]
fn segment_notation_apply_nests() {
    // Prefix application `seg @ wires` nests inside wire expressions; each
    // nesting desugars to a fresh intermediate wire: d = a + (a + b).
    let seg = tradingflow::segment!(@[tradingflow_graph::cb] |a: Port<Val<i64>>, b: Port<Val<i64>>| -> Port<Val<i64>> {
        let d = add() @ (a, add() @ (a, b));
        d
    });

    let mut gb = Builder::new();
    let (s0_cell, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let od = gb.segment(seg, (s0, s1));
    let mut g = gb.build();
    let mut pool = pool();

    assert_eq!(g.view(od), 40); // 10 + (10 + 20)

    *g.state_mut(s0_cell) = 11;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(od), 42); // 11 + (11 + 20)
}

#[test]
fn segment_notation_apply_in_result_position() {
    // A result-position application chain (no `let`) desugars to a final
    // statement, which the closing `route` projects out.
    // `@` chains right-associatively: Inc @ Add @ (a, b) is Inc(Add(a, b)).
    let seg = tradingflow::segment!(@[tradingflow_graph::cb] |a: Port<Val<i64>>, b: Port<Val<i64>>| -> Port<Val<i64>> {
        inc() @ add() @ (a, b)
    });

    let mut gb = Builder::new();
    let (_, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(20));
    let o = gb.segment(seg, (s0, s1));
    let g = gb.build();
    assert_eq!(g.view(o), 31);
}

#[test]
fn segment_notation_apply_in_statement_args() {
    // Applications nest inside a statement's argument tuple, and multi-output
    // destructuring works on an application statement.
    let seg = tradingflow::segment!(
        |a: Port<Val<i64>>, b: Port<Val<i64>>| -> (Port<Val<i64>>, Port<Val<i64>>) {
            let (p, q) = AddSub @ (inc() @ a, b);
            let r = add() @ (p, inc() @ q);
            (r, p)
        }
    );

    let mut gb = Builder::new();
    let (_, s0) = gb.source(Source(10));
    let (_, s1) = gb.source(Source(4));
    let (or, op) = gb.segment(seg, (s0, s1));
    let g = gb.build();
    assert_eq!(g.view(op), 15); // (10+1) + 4
    assert_eq!(g.view(or), 23); // 15 + ((11-4) + 1)
}

// ===== regression tests =====

#[test]
fn out_of_cone_producer_does_not_spuriously_notify() {
    // S0 -> P=Inc(S0); S1; D=DidFirstNotify(P, S1). Poke ONLY S1: P is a
    // predecessor of D but not in the dirty cone, so P does not run this
    // generation and must not be seen as having notified. Regression: build-time
    // notify flags used to linger for out-of-cone producers (Inc's build call
    // leaves notify=true), so on D's first participating generation it read
    // input 0 as notified and returned 1.
    let mut b = Builder::new();
    let (_, s0) = b.source(EventSource(0));
    let p = b.segment(IncEvent, s0);
    let (s1_cell, s1) = b.source(Source(0));
    let d = b.segment(DidFirstNotify, (p, s1));
    let mut g = b.build();
    let mut pool = pool();

    *g.state_mut(s1_cell) = 1;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(d), 0); // P did not notify this generation
}

/// A `Ports` producer whose output SHRINKS (and reallocates) after the build
/// call: build emits 2 leaves, later calls 1. The shrink leaves a stale
/// build-time pointer in the tail output slot, so the engine must poison rather
/// than scatter it (a dangling read in a release build).
struct ShrinkPorts;
impl Segment for ShrinkPorts {
    type Inputs = Port<Val<i64>>;
    type Outputs = Ports<Val<i64>>;
    type Context = ();
    type State = (Vec<i64>, Vec<bool>);
    fn init(self, x: i64) -> Self::State {
        (vec![x, x], vec![true, true])
    }
    fn reset<'a, 'b: 'a>(_: i64, (vals, _flags): &'b mut Self::State) -> &'a [i64] {
        vals
    }
    fn compute<'a, 'b: 'a>(x: i64, (vals, flags): &'b mut Self::State, _: &()) -> &'a [i64] {
        *vals = vec![x]; // 2 -> 1, reallocates
        *flags = vec![true; vals.len()];
        vals
    }
}

#[test]
#[should_panic(expected = "output shape changed since build")]
fn shrinking_ports_output_poisons_instead_of_dangling() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(1i64));
    let out = b.segment(ShrinkPorts, s);
    assert_eq!(out.len(), 2); // sized to 2 at build
    let mut g = b.build();
    let mut pool = pool();

    *g.state_mut(s_cell) = 5; // ShrinkPorts now writes 1 leaf -> must poison
    g.stabilize(&mut pool, &());
}

// ===== Port leaf: borrowed fat references through one wire slot =============

/// Emits a window of its input as ONE contiguous `&[i64]` view -- what `Ports`
/// cannot express: per-generation position AND length, zero per-element cost.
/// Views travel BY VALUE: the engine homes them in the node's output scratch,
/// so this producer is completely stateless.
struct WindowView;
impl Segment for WindowView {
    type Inputs = (Port<Ref<Vec<i64>>>, Port<Val<(usize, usize)>>);
    type Outputs = Port<Slice<i64>>;
    type Context = ();
    type State = ();
    fn init(self, _: (&Vec<i64>, (usize, usize))) {}
    fn reset<'a, 'b: 'a>(
        (data, (start, len)): (&'a Vec<i64>, (usize, usize)),
        _: &'b mut (),
    ) -> &'a [i64] {
        &data[start..start + len]
    }
    fn compute<'a, 'b: 'a>(
        inputs: (&'a Vec<i64>, (usize, usize)),
        state: &'b mut (),
        _: &(),
    ) -> &'a [i64] {
        Self::reset(inputs, state)
    }
}

/// Consumes the window view: `(sum, len)`.
struct ViewStats;
impl Segment for ViewStats {
    type Inputs = Port<Slice<i64>>;
    type Outputs = (Port<Val<i64>>, Port<Val<usize>>);
    type Context = ();
    type State = ();
    fn init(self, _: &[i64]) {}
    fn reset<'a, 'b: 'a>(view: &'a [i64], _: &'b mut ()) -> (i64, usize) {
        ((view.iter().sum()), (view.len()))
    }
    fn compute<'a, 'b: 'a>(inputs: &'a [i64], state: &'b mut Self::State, _: &()) -> (i64, usize) {
        Self::reset(inputs, state)
    }
}

#[test]
fn view_window_moves_and_resizes_per_generation() {
    let mut b = Builder::new();
    let (data_cell, data) = b.source(RefSource(vec![1i64, 2, 3, 4, 5, 6, 7, 8]));
    let (range_cell, range) = b.source(Source((0usize, 3usize)));
    let win = b.segment(WindowView, (data, range));
    // Forward the view through an extra node: the forwarder re-homes the fat
    // reference in its OWN output scratch (a copy, not an alias). `Id`'s type
    // parameter is inferred from `win`'s handle -- no turbofish.
    let fwd = b.segment(Id::default(), win);
    let (sum, len) = b.segment(ViewStats, fwd);
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(sum), 6); // [1, 2, 3]
    assert_eq!(g.view(len), 3);

    *g.state_mut(range_cell) = (2, 5); // the window MOVES and RESIZES
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(sum), 3 + 4 + 5 + 6 + 7);
    assert_eq!(g.view(len), 5);

    *g.state_mut(data_cell) = vec![10; 8]; // fresh buffer; the view is re-derived
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(sum), 50);
    assert_eq!(g.view(len), 5);
}

/// A custom user view struct with embedded references (the n-d `ArrayView`
/// shape). The dynamically-sized metadata (`shape`) is itself arena-resident:
/// `Copy` rules out *ownership*, not dynamic size.
#[derive(Clone, Copy)]
struct Strided<'a> {
    data: &'a [f64],
    shape: &'a [usize], // [count, stride]: dyn-sized, lives in the arena
}

/// Its `'static` name on the wire.
struct StridedF64;
// SAFETY: `Strided` is a plain read-only view (covariant, no interior
// mutability).
unsafe impl Pass for StridedF64 {
    type View<'a> = Strided<'a>;
}

/// Wraps its input buffer in a `Strided` view chosen by the stride input. The
/// view itself travels by value; only its variable-length metadata needs the
/// arena in state.
struct MakeStrided;
impl Segment for MakeStrided {
    type Inputs = (Port<Ref<Vec<f64>>>, Port<Val<usize>>);
    type Outputs = Port<StridedF64>;
    type Context = ();
    type State = Bump;
    fn init(self, _: (&Vec<f64>, usize)) -> Bump {
        Bump::new()
    }
    fn reset<'a, 'b: 'a>((v, stride): (&'a Vec<f64>, usize), arena: &'b mut Bump) -> Strided<'a> {
        let shape = fresh(arena).alloc_slice_fill_iter([v.len().div_ceil(stride), stride]);
        Strided { data: v, shape }
    }
    fn compute<'a, 'b: 'a>(
        inputs: (&'a Vec<f64>, usize),
        arena: &'b mut Bump,
        _: &(),
    ) -> Strided<'a> {
        Self::reset(inputs, arena)
    }
}

/// Sums every `stride`-th element of the view, walking its metadata.
struct StridedSum;
impl Segment for StridedSum {
    type Inputs = Port<StridedF64>;
    type Outputs = Port<Val<f64>>;
    type Context = ();
    type State = ();
    fn init(self, _: Strided<'_>) {}
    fn reset<'a, 'b: 'a>(view: Strided<'a>, _: &'b mut ()) -> f64 {
        (0..view.shape[0])
            .map(|i| view.data[i * view.shape[1]])
            .sum()
    }
    fn compute<'a, 'b: 'a>(inputs: Strided<'a>, state: &'b mut (), _: &()) -> f64 {
        Self::reset(inputs, state)
    }
}

#[test]
fn custom_view_struct_through_the_wire() {
    let mut b = Builder::new();
    let (_, v) = b.source(RefSource(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let (k_cell, k) = b.source(Source(2usize));
    let view = b.segment(MakeStrided, (v, k));
    let out = b.segment(StridedSum, view);
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(out), 1.0 + 3.0 + 5.0);

    *g.state_mut(k_cell) = 3;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out), 1.0 + 4.0);
}

/// Sums a borrowed slice view plus a scalar into owned state.
struct SliceTotal;
impl Segment for SliceTotal {
    type Inputs = (Port<Slice<i64>>, Port<Val<i64>>);
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = ();
    fn init(self, _: (&[i64], i64)) {}
    fn reset<'a, 'b: 'a>((xs, k): (&'a [i64], i64), _: &'b mut ()) -> i64 {
        xs.iter().sum::<i64>() + k
    }
    fn compute<'a, 'b: 'a>(inputs: (&'a [i64], i64), state: &'b mut (), _: &()) -> i64 {
        Self::reset(inputs, state)
    }
}

#[test]
fn view_source_lends_borrowing_view_from_owned_cell() {
    // A borrowing-view source cell: `Source<Slice<i64>>` owns a `Vec<i64>`
    // in node state (`Value::Owned`) and lends `&[i64]` by value. Pokes go
    // through `state_mut`, which dirties the node, so the view is re-derived
    // and re-homed before any consumer in the cone reads it.
    let mut b = Builder::new();
    let (xs_cell, xs) = b.source(SliceSource(vec![1, 2, 3].into()));
    let (k_cell, k) = b.source(Source(100i64));
    let out = b.segment(SliceTotal, (xs, k));
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(xs), &[1, 2, 3][..]);
    assert_eq!(g.view(out), 106);

    // Poke only the scalar: the consumer re-runs and re-reads the slice wire
    // (un-notified, unchanged) alongside the fresh scalar.
    *g.state_mut(k_cell) = 1000;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out), 1006);

    // Poke the owned cell with a write that reallocates its buffer: the node
    // is dirtied, so the lent view tracks the moved storage.
    *g.state_mut(xs_cell) = (1..=64).collect();
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(xs).len(), 64);
    assert_eq!(g.view(out), (1..=64).sum::<i64>() + 1000);
}

/// State containing a `Cell` -- `Send` but `!Sync`. Legal: the engine hands a
/// node's state off exclusively between workers, never sharing it; only the
/// *exposed* output value (the `i64` second field) must be `Sync`.
struct CellCounter;
impl Segment for CellCounter {
    type Inputs = Port<Val<i64>>;
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = std::cell::Cell<i64>;
    fn init(self, _: i64) -> Self::State {
        std::cell::Cell::new(0)
    }
    fn reset<'a, 'b: 'a>(x: i64, calls: &'b mut Self::State) -> i64 {
        x + calls.get()
    }
    fn compute<'a, 'b: 'a>(x: i64, calls: &'b mut Self::State, _: &()) -> i64 {
        calls.set(calls.get() + 1);
        x + calls.get()
    }
}

#[test]
fn send_only_state_is_accepted() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(10i64));
    let out = b.segment(CellCounter, s);
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(out), 10); // build call: 10 + 0
    *g.state_mut(s_cell) = 20;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out), 21); // 20 + 1
}

// ===== large complex-graph stress tests (multi-generation) ==================
//
// A wide, locally-and-globally connected mesh: `NM` columns x `LM` layers of
// 3-input `Work` nodes (each with its own, deliberately uneven, workload via
// `iters_of`) plus a full-layer `SumAll` aggregate per layer -- ~1.1k scheduled
// nodes with non-trivial topology (two local predecessors + one long-range one
// per node). Every node's value is a deterministic function of its inputs, so a
// plain-Rust `simulate` mirror predicts every slot exactly; we drive several
// generations of partial source pokes and check the whole graph each time.

const NM: usize = 32; // mesh width
const LM: usize = 32; // mesh depth
const GENS: usize = 6;
const MODULUS: i64 = 1_000_000_007;

/// `iters` rounds of an LCG, folded mod `MODULUS` so node values stay bounded
/// (keeping every full-layer sum well inside `i64`).
fn lcg(seed: i64, iters: u32) -> i64 {
    let mut h = seed.rem_euclid(MODULUS);
    for _ in 0..iters {
        h = h.wrapping_mul(48271).wrapping_add(1).rem_euclid(MODULUS);
    }
    h
}

/// A mesh node's raw output: mix the three inputs, then churn `iters` rounds.
fn work(a: i64, b: i64, c: i64, iters: u32) -> i64 {
    let seed = a
        .wrapping_mul(3)
        .wrapping_add(b.wrapping_mul(5))
        .wrapping_add(c.wrapping_mul(7));
    lcg(seed, iters)
}

/// Wiring of node `(layer, j)`: two local predecessors and one long-range one.
fn preds(layer: usize, j: usize, n: usize) -> [usize; 3] {
    [j, (j + 1) % n, (j * 13 + layer * 7 + 1) % n]
}

/// Per-node workload: mostly light (0..18 rounds), ~1/16 deliberately heavy.
fn iters_of(layer: usize, j: usize) -> u32 {
    let base = ((layer * 31 + j * 17) % 19) as u32;
    if (layer + j).is_multiple_of(16) {
        base * 40 + 200
    } else {
        base
    }
}

/// Source value for column `j` at generation `gen` (deterministic, in range).
fn src_val(gn: usize, j: usize) -> i64 {
    (gn as i64 * 7_654_321 + j as i64 * 1_234_567 + 1) % MODULUS
}

/// 3-input mixer node; `iters` (the workload knob) lives in state because
/// `compute` is static and cannot see `self`.
struct Work {
    iters: u32,
}
impl Segment for Work {
    type Inputs = (Port<Val<i64>>, Port<Val<i64>>, Port<Val<i64>>);
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = u32;
    fn init(self, _: (i64, i64, i64)) -> Self::State {
        self.iters
    }
    fn reset<'a, 'b: 'a>((a, b, c): (i64, i64, i64), iters: &'b mut Self::State) -> i64 {
        work(a, b, c, *iters)
    }
    fn compute<'a, 'b: 'a>((a, b, c): (i64, i64, i64), iters: &'b mut Self::State, _: &()) -> i64 {
        work(a, b, c, *iters)
    }
}

/// Plain-Rust mirror of the whole mesh: per-layer node values (layer 0 =
/// sources) and per-layer aggregates. `post` maps a node's raw `work` output to
/// its emitted value -- identity for the plain mesh, the fused post-processing
/// for the fused mesh.
fn simulate(srcs: &[i64], post: impl Fn(usize, usize, i64) -> i64) -> (Vec<Vec<i64>>, Vec<i64>) {
    let n = srcs.len();
    let mut layers: Vec<Vec<i64>> = vec![srcs.to_vec()];
    for layer in 1..=LM {
        let prev = &layers[layer - 1];
        let cur: Vec<i64> = (0..n)
            .map(|j| {
                let [a, b, c] = preds(layer, j, n);
                post(
                    layer,
                    j,
                    work(prev[a], prev[b], prev[c], iters_of(layer, j)),
                )
            })
            .collect();
        layers.push(cur);
    }
    let aggs: Vec<i64> = layers.iter().map(|l| l.iter().sum::<i64>()).collect();
    (layers, aggs)
}

/// Build the shared mesh topology, delegating each mesh node's construction to
/// `node` (a plain or fused segment of the same interface). Returns the source
/// handles (to poke), the per-layer node handles, and the per-layer aggregates.
#[allow(clippy::type_complexity)]
fn build_mesh(
    b: &mut Builder<()>,
    srcs: &[i64],
    mut node: impl FnMut(
        &mut Builder<()>,
        usize,
        usize,
        [PortHandle<Val<i64>>; 3],
    ) -> PortHandle<Val<i64>>,
) -> (
    Vec<NodeHandle<Source<i64>>>,
    Vec<Vec<PortHandle<Val<i64>>>>,
    Vec<PortHandle<Val<i64>>>,
) {
    let (src, wires): (Vec<NodeHandle<Source<i64>>>, Vec<_>) =
        srcs.iter().map(|&v| b.source(Source(v))).unzip();
    let mut layers: Vec<Vec<PortHandle<Val<i64>>>> = vec![wires];
    for layer in 1..=LM {
        let prev = layers[layer - 1].clone();
        let mut cur = Vec::with_capacity(NM);
        for j in 0..NM {
            let [a, b2, c] = preds(layer, j, NM);
            cur.push(node(&mut *b, layer, j, [prev[a], prev[b2], prev[c]]));
        }
        layers.push(cur);
    }
    let aggs: Vec<PortHandle<Val<i64>>> =
        layers.iter().map(|l| b.segment(SumAll, &l[..])).collect();
    (src, layers, aggs)
}

#[test]
#[cfg_attr(miri, ignore)] // ~1.1k nodes x 7 generations: far too slow interpreted
fn complex_mesh_multi_generation() {
    let mut srcs: Vec<i64> = (0..NM).map(|j| src_val(0, j)).collect();
    let mut b = Builder::new();
    let (src, nodes, aggs) = build_mesh(&mut b, &srcs, |b, layer, j, [a, b2, c]| {
        b.segment(
            Work {
                iters: iters_of(layer, j),
            },
            (a, b2, c),
        )
    });
    let mut g = b.build();
    let mut pool = pool();

    let check = |g: &Graph<()>, srcs: &[i64], gn: usize| {
        let (sl, sa) = simulate(srcs, |_, _, w| w);
        for layer in 0..=LM {
            for j in 0..NM {
                assert_eq!(
                    g.view(nodes[layer][j]),
                    sl[layer][j],
                    "node ({layer},{j}) gen {gn}"
                );
            }
            assert_eq!(g.view(aggs[layer]), sa[layer], "agg {layer} gen {gn}");
        }
    };
    check(&g, &srcs, 0);

    for gn in 1..=GENS {
        for j in 0..NM {
            if (j * 7 + gn * 3) % 5 == 0 {
                srcs[j] = src_val(gn, j);
                *g.state_mut(src[j]) = srcs[j];
            }
        }
        g.stabilize(&mut pool, &());
        check(&g, &srcs, gn);
    }
}

#[test]
#[cfg_attr(miri, ignore)] // ~1.1k nodes x 7 generations: far too slow interpreted
fn complex_mesh_with_fused_nodes() {
    // The same mesh, but every node is a *fused* segment running a small internal
    // DAG: even nodes via the `segment!` macro (a fork/join diamond, net 3w+1),
    // odd nodes via the combinator API (`Work.then(Inc).then(Double)`, net 2w+2).
    // The scheduled topology -- and node count -- is identical to the plain mesh.
    let mut srcs: Vec<i64> = (0..NM).map(|j| src_val(0, j)).collect();
    let mut b = Builder::new();
    let (src, nodes, aggs) = build_mesh(&mut b, &srcs, |b, layer, j, [a, b2, c]| {
        let it = iters_of(layer, j);
        if (layer + j) % 2 == 0 {
            let seg = tradingflow::segment!(@[tradingflow_graph::cb] |x: Port<Val<i64>>, y: Port<Val<i64>>, z: Port<Val<i64>>| -> Port<Val<i64>> {
                let w = Work { iters: it } @ (x, y, z);
                let p = inc() @ w;
                let q = double() @ w;
                let r = add() @ (p, q);
                r
            });
            b.segment(seg, (a, b2, c))
        } else {
            b.segment(Work { iters: it }.then(inc()).then(double()), (a, b2, c))
        }
    });
    let mut g = b.build();
    let mut pool = pool();

    let check = |g: &Graph<()>, srcs: &[i64], gn: usize| {
        let (sl, sa) = simulate(srcs, |layer, j, w| {
            if (layer + j) % 2 == 0 {
                3 * w + 1
            } else {
                2 * w + 2
            }
        });
        for layer in 0..=LM {
            for j in 0..NM {
                assert_eq!(
                    g.view(nodes[layer][j]),
                    sl[layer][j],
                    "node ({layer},{j}) gen {gn}"
                );
            }
            assert_eq!(g.view(aggs[layer]), sa[layer], "agg {layer} gen {gn}");
        }
    };
    check(&g, &srcs, 0);

    for gn in 1..=GENS {
        for j in 0..NM {
            if (j * 11 + gn * 5) % 4 == 0 {
                srcs[j] = src_val(gn, j);
                *g.state_mut(src[j]) = srcs[j];
            }
        }
        g.stabilize(&mut pool, &());
        check(&g, &srcs, gn);
    }
}

// ===== ViewPorts leaf: by-value fan-in groups ================================

/// Sum over a runtime-sized list of BY-VALUE inputs (`ViewPorts<Scalar<T>>` =
/// `ViewPorts<Scalar<T>>`): the group wires against plain `Port` producers
/// with no bridging adapters, and deserialization gathers the scattered wire
/// values into the node's input scratch, so the payload arrives as ONE
/// contiguous `&[i64]`. Both sides by value: completely stateless.
struct SumAllPorts;
impl Segment for SumAllPorts {
    type Inputs = Ports<Val<i64>>;
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = ();
    fn init(self, _: &[i64]) {}
    fn reset<'a, 'b: 'a>(xs: &'a [i64], _: &'b mut ()) -> i64 {
        xs.iter().sum()
    }
    fn compute<'a, 'b: 'a>(inputs: &'a [i64], state: &'b mut (), _: &()) -> i64 {
        Self::reset(inputs, state)
    }
}

#[test]
fn ports_group_input_sums() {
    let mut b = Builder::new();
    let (s0_cell, s0) = b.source(Source(1i64));
    let (s1_cell, s1) = b.source(Source(2i64));
    let (s2_cell, s2) = b.source(Source(3i64));
    let total = b.segment(SumAllPorts, &[s0, s1, s2]);
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(total), 6);

    *g.state_mut(s0_cell) = 10;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(total), 15);

    *g.state_mut(s1_cell) = -2;
    *g.state_mut(s2_cell) = 30;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(total), 38);
}

#[test]
fn empty_ports_group_builds_and_sums_zero() {
    let mut b = Builder::new();
    let total = b.segment(SumAllPorts, &[] as &[PortHandle<Val<i64>>]);
    let g = b.build();
    assert_eq!(g.view(total), 0);
}

/// A runtime-sized BY-VALUE output list: `x -> [x, x+1, x+2]`. The payload
/// slices are borrowed from the node's arena only until serialization homes
/// each element in the node's output scratch, slot by slot -- so each slot
/// then feeds single-`Port` consumers and `Ports` groups alike.
struct FanoutPorts3;
impl Segment for FanoutPorts3 {
    type Inputs = Port<Val<i64>>;
    type Outputs = Ports<Val<i64>>;
    type Context = ();
    type State = Bump;
    fn init(self, _: i64) -> Bump {
        Bump::new()
    }
    fn reset<'a, 'b: 'a>(x: i64, arena: &'b mut Bump) -> &'a [i64] {
        let a = fresh(arena);
        a.alloc_slice_fill_iter((0..3).map(|i| x + i as i64))
    }
    fn compute<'a, 'b: 'a>(inputs: i64, arena: &'b mut Bump, _: &()) -> &'a [i64] {
        Self::reset(inputs, arena)
    }
}

#[test]
fn ports_group_output_dynamic_arity() {
    let mut b = Builder::new();
    let (s_cell, s) = b.source(Source(10i64));
    let outs = b.segment(FanoutPorts3, s);
    assert_eq!(outs.len(), 3);
    // Each group slot is an ordinary `Port` producer: read it directly, and
    // fan the whole group back into a `Ports` consumer.
    let total = b.segment(SumAllPorts, &outs[..]);
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(outs[0]), 10);
    assert_eq!(g.view(outs[1]), 11);
    assert_eq!(g.view(outs[2]), 12);
    assert_eq!(g.view(total), 33);

    *g.state_mut(s_cell) = 100;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(outs[0]), 100);
    assert_eq!(g.view(outs[2]), 102);
    assert_eq!(g.view(total), 303);
}

/// Forward a `Ports` group through `Id`: the forwarder's output payload
/// borrows its OWN input scratch (the gathered views), and serialization
/// re-homes each element in its output scratch. Exercises the
/// input-scratch-to-output-scratch aliasing path under Miri.
#[test]
fn ports_group_forwarded_through_id() {
    let mut b = Builder::new();
    let (_, s0) = b.source(Source(4i64));
    let (s1_cell, s1) = b.source(Source(5i64));
    let fwd = b.segment(Id::<Ports<Val<i64>>, ()>::default(), &[s0, s1]);
    let total = b.segment(SumAllPorts, &fwd[..]);
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(total), 9);

    *g.state_mut(s1_cell) = 50;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(fwd[1]), 50);
    assert_eq!(g.view(total), 54);
}

/// Mixed variadic tree: a by-value group, a fixed by-ref leaf, and a by-ref
/// group in one input tuple. Checks the shape cursor stays aligned across
/// value/ref variadic leaves: `sum(ports) * k + sum(ports)`.
struct MixedGroups;
impl Segment for MixedGroups {
    type Inputs = (Ports<Val<i64>>, Port<Ref<i64>>, Ports<Ref<i64>>);
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = ();
    fn init(self, _: (&[i64], &i64, &[&i64])) {}
    fn reset<'a, 'b: 'a>((xs, k, ys): (&'a [i64], &'a i64, &'a [&'a i64]), _: &'b mut ()) -> i64 {
        xs.iter().sum::<i64>() * k + ys.iter().map(|&v| *v).sum::<i64>()
    }
    fn compute<'a, 'b: 'a>(
        inputs: (&'a [i64], &'a i64, &'a [&'a i64]),
        state: &'b mut (),
        _: &(),
    ) -> i64 {
        Self::reset(inputs, state)
    }
}

#[test]
fn mixed_value_and_ref_variadic_groups() {
    let mut b = Builder::new();
    let (_, p0) = b.source(Source(1i64));
    let (p1_cell, p1) = b.source(Source(2i64));
    let (_, k) = b.source(RefSource(10i64));
    let (r0_cell, r0) = b.source(RefSource(100i64));
    let (_, r1) = b.source(RefSource(200i64));
    let (_, r2) = b.source(RefSource(300i64));
    let out = b.segment(MixedGroups, (&[p0, p1], k, &[r0, r1, r2]));
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(out), (1 + 2) * 10 + 600);

    *g.state_mut(p1_cell) = 5;
    *g.state_mut(r0_cell) = 111;
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(out), (1 + 5) * 10 + 611);
}

/// `ViewPorts` over a non-scalar view (`Slice<i64>`): each element is a fat
/// `&[i64]` window into a different source's data, gathered into one
/// contiguous slice-of-windows payload (views by value).
struct ConcatLens;
impl Segment for ConcatLens {
    type Inputs = Ports<Slice<i64>>;
    type Outputs = Port<Val<i64>>;
    type Context = ();
    type State = ();
    fn init(self, _: &[&[i64]]) {}
    fn reset<'a, 'b: 'a>(windows: &'a [&'a [i64]], _: &'b mut ()) -> i64 {
        windows.iter().map(|w| w.iter().sum::<i64>()).sum()
    }
    fn compute<'a, 'b: 'a>(inputs: &'a [&'a [i64]], state: &'b mut (), _: &()) -> i64 {
        Self::reset(inputs, state)
    }
}

#[test]
fn view_ports_group_of_slice_views() {
    let mut b = Builder::new();
    let (_, d0) = b.source(RefSource(vec![1i64, 2, 3, 4]));
    let (_, d1) = b.source(RefSource(vec![10i64, 20]));
    let (r0_cell, r0) = b.source(Source((0usize, 2usize)));
    let (_, r1) = b.source(Source((0usize, 2usize)));
    let w0 = b.segment(WindowView, (d0, r0));
    let w1 = b.segment(WindowView, (d1, r1));
    let total = b.segment(ConcatLens, &[w0, w1]);
    let mut g = b.build();
    let mut pool = pool();

    assert_eq!(g.view(total), (1 + 2) + (10 + 20));

    *g.state_mut(r0_cell) = (1, 3); // window slides and grows
    g.stabilize(&mut pool, &());
    assert_eq!(g.view(total), (2 + 3 + 4) + (10 + 20));
}
