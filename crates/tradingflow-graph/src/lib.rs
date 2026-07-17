//! A multithreaded executor for static computation graphs.
//!
//! Each compute node can hold a mutable state, read inputs and produce outputs
//! *that may contain references into the node's state or into its inputs*: it
//! is guaranteed that no dangling references can be read.
//!
//! Multiple nodes can be fused together into a single node, or be scheduled
//! independently to be run in parallel.
//!
//! # Examples
//!
//! A diamond-shaped graph - source `s`, `a = s + 1`, and `d = s + a`:
//!
//! ```rust
//! use tradingflow_graph::core::Pool;
//! use tradingflow_graph::typed::{Graph, Builder, Port, Segment, Source};
//!
//! /// A stateless increment operator.
//! struct Inc;
//!
//! impl Segment for Inc {
//!     type Inputs = Port<i64>;
//!     type Outputs = Port<i64>;
//!     type Context = ();
//!     type State = ();
//!
//!     fn init(self) {}
//!
//!     fn compute<'a, 'b: 'a>(
//!         (x_notify, x): (bool, i64),
//!         _context: &(),
//!         _state: &'b mut (),
//!         _is_first_run: bool
//!     ) -> (bool, i64) {
//!         (x_notify, x + 1)
//!     }
//! }
//!
//! /// A stateless adder with two inputs.
//! struct Add;
//!
//! impl Segment for Add {
//!     type Inputs = (Port<i64>, Port<i64>);
//!     type Outputs = Port<i64>;
//!     type Context = ();
//!     type State = ();
//!
//!     fn init(self) {}
//!
//!     fn compute<'a, 'b: 'a>(
//!         ((a_notify, a), (b_notify, b)): ((bool, i64), (bool, i64)),
//!         _context: &(),
//!         _state: &'b mut (),
//!         _is_first_run: bool
//!     ) -> (bool, i64) {
//!         (a_notify || b_notify, a + b)
//!     }
//! }
//!
//! // Create the thread pool.
//! let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
//!
//! // Create the graph, which runs each segment for the first time. A source
//! // returns a pokeable handle to its state cell plus its output wire.
//! let mut b = Builder::new(());
//! let (s_cell, s) = b.source(Source::new(1));
//! let a = b.segment(Inc, s);
//! let d = b.segment(Add, (s, a));
//! let mut g = b.build();
//!
//! // Check initial values.
//! assert_eq!(g.view(s), 1);
//! assert_eq!(g.view(a), 2);
//! assert_eq!(g.view(d), 3);
//!
//! // Update the source value and recompute in parallel.
//! *g.state_mut(s_cell) = 5;
//! g.stabilize(&mut pool);
//!
//! // Check updated values.
//! assert_eq!(g.view(a), 6);
//! assert_eq!(g.view(d), 11);
//! ```
//!
//! # Segments
//!
//! Each unit of scheduling is a [`Segment`](typed::Segment): it has typed
//! inputs and outputs, a mutable state, and a compute function.
//!
//! ```rust
//! use tradingflow_graph::typed::*;
//!
//! pub trait Segment {
//!     type Inputs: Interface;
//!     type Outputs: Interface;
//!     type Context: Send + Sync + 'static;
//!     type State: Send + 'static;
//!
//!     fn init(self) -> Self::State;
//!
//!     fn compute<'a, 'b: 'a>(
//!         inputs: <Self::Inputs as Interface>::Values<'a>,
//!         context: &Self::Context,
//!         state: &'b mut Self::State,
//!         is_first_run: bool,
//!     ) -> <Self::Outputs as Interface>::Values<'a>;
//! }
//! ```
//!
//! The [`init`](typed::Segment::init) method will be called once during graph
//! construction to create the node's state. Immediately after, the
//! [`compute`](typed::Segment::compute) method will be run once with
//! `is_first_run == true` to obtain initial/default output values.
//!
//! On each subsequent [`graph.stabilize`](typed::Graph::stabilize) call, the
//! `compute` method will be run again with new inputs and
//! `is_first_run == false`.
//!
//! # Interfaces
//!
//! The associated types [`Inputs`](typed::Segment::Inputs) and
//! [`Outputs`](typed::Segment::Outputs) define the [`Interface`](typed::Interface)
//! of a segment. They can be constructed from the following building blocks:
//!
//! - [`Port<T>`](typed::Port) — a single pass-by-value port. It carries
//!   `(bool, T)` where `T: Copy + Send + Sync`.
//! - [`Ports<T>`](typed::Ports) — a dynamic-length group of pass-by-value
//!   ports. It carries `(&[bool], &[T])` where `T: Copy + Send + Sync`, and is
//!   compatible with a group of `Port<T>`s.
//! - [`RefPort<T>`](typed::RefPort) — a single pass-by-reference port. It
//!   carries `(bool, &T)` where `T: Send + Sync`.
//! - [`RefPorts<T>`](typed::RefPorts) — a dynamic-length group of
//!   pass-by-reference ports. It carries `(&[bool], &[&T])` where
//!   `T: Send + Sync`, and is compatible with a group of `RefPort<T>`s.
//! - Generalizations of the above: [`ViewPort<V>`](typed::ViewPort) and
//!   [`ViewPorts<V>`](typed::ViewPorts) over any [`Value`](typed::Value) kind
//!   `V`, which allow passing custom lifetime-carrying `Copy` views (like slices
//!   or custom array views) to some underlying data. The four aliases above are
//!   `ViewPort[s]` over the built-in kinds [`Scalar<T>`](typed::Scalar) (the
//!   value itself) and [`Ref<T>`](typed::Ref) (a plain `&T` into producer
//!   state).
//! - Arbitrarily nested tuples of the above (each branch up to arity 12).
//!
//! For dynamic-length groups, the length must remain fixed after the first run,
//! so that we have a well-defined static computation graph. Violations will be
//! caught and panic at runtime.
//!
//! Producing a [`Ports`](typed::Ports) or [`RefPorts`](typed::RefPorts) output
//! requires creating slices with lifetime `'a` matching the inputs. This allows
//! for simple forwarding of input references, but it also creates difficulty for
//! a node that computes its own values: we have to put the arrays somewhere, but
//! the node state must have static lifetime. To address this difficulty, use a
//! lifetime-erased bump arena (such as [`bumpalo`](https://crates.io/crates/bumpalo))
//! in the node state as a scratch buffer storing the arrays on each recompute.
//! The arena can be cleared at the beginning of each recompute, so that buffer
//! size is kept bounded.
//!
//! > Interface values are constrained to [`Copy`] because they are required to
//! > have trivial [`Drop`] implementations. Data ownership is always inside node
//! > states and never passed through an interface; only simple scalar values,
//! > references and views do. This encourages pass-by-reference for complex data
//! > types and simplifies the internal implementation, but may be less ergonomic
//! > in some cases.
//!
//! # Notification flags
//!
//! Each port carries a `bool` flag alongside the value or reference, indicating
//! whether the value is a *notification*. At each generation, an unmodified
//! output value can have its flag set to `false` (no notify), so a downstream
//! node can choose to skip heavy computation.
//!
//! ```rust
//! use tradingflow_graph::typed::*;
//!
//! struct Abs;
//!
//! impl Segment for Abs {
//!     type Inputs = Port<i64>;
//!     type Outputs = Port<i64>;
//!     type Context = ();
//!     type State = i64; // Remembers the previous output.
//!
//!     fn init(self) -> i64 { 0 }
//!
//!     fn compute<'a, 'b: 'a>(
//!         (x_notify, x): (bool, i64),
//!         _context: &(),
//!         state: &'b mut i64,
//!         _is_first_run: bool
//!     ) -> (bool, i64) {
//!         if x_notify {
//!             // Input notified, recompute.
//!             let output = x.abs();
//!             let changed = output != *state; // Notify downstream only when |x| actually changed.
//!             *state = output;
//!             (changed, output)
//!         } else {
//!             // Input did not notify, simply pass through.
//!             (false, *state)
//!         }
//!     }
//! }
//! ```
//!
//! Treat the flag as a contract that every well-behaved segment upholds:
//! whenever an output flag is `false`, the value at that port is equal (by
//! [`Eq`]) to the value it held in the previous generation. Therefore:
//!
//! - A port with `notify == true` indicates a possible change in its value;
//! - A port with `notify == false` indicates no change in its value.
//!
//! The flag can be interpreted in an alternative way:
//!
//! - A port with `notify == true` indicates a new event has arrived, with its
//!   value being the payload of the event;
//! - A port with `notify == false` indicates no event has arrived, with its
//!   value being the payload of the last known event.
//!
//! The two interpretations are compatible: a new event *is* a change in the
//! event payload's conceptual identity, and a value change *is* a new event.
//!
//! The notification flags also help in scheduling. Each generation, the graph
//! executor sets the flags of modified source nodes, and skips a node completely
//! if none of its upstream source nodes were modified. This makes stabilization
//! after a sparse update touches only a fraction of the graph.
//!
//! # Graph-level context
//!
//! Besides its inputs and state, every [`compute`](typed::Segment::compute)
//! receives a shared reference to a single graph-owned **context** value — the
//! [`Context`](typed::Segment::Context) associated type. The context is seeded
//! by [`Builder::new(context)`](typed::Builder::new) and overwritten between
//! generations through [`graph.context_mut()`](typed::Graph::context_mut).
//! Writing it dirties nothing: the context is ambient data, not a graph
//! dependency, so a segment that reads it still recomputes only when its own
//! inputs notify. Segments that ignore the context declare `type Context = ()`;
//! combinators and [`segment!`] formulas are generic over it.
//!
//! The context reference's lifetime is deliberately unrelated to the
//! per-generation lifetime `'a`, so outputs can never borrow from the context:
//! out-of-cone output slots must stay valid across generations while the context
//! keeps changing.
//!
//! The canonical use is an event-loop driver holding the current *event
//! timestamp*: the driver writes each batch's timestamp into the context between
//! the batch's source-node writes and its
//! [`stabilize`](typed::Graph::stabilize), so a time-stamping segment declares
//! `type Context = Timestamp` and is simply handed the batch time in its
//! `compute` — no clock handle is threaded through construction, and stale-input
//! segments never observe a half-advanced clock. (This is how the
//! timestamp-ordered ingestion driver in
//! [TradingFlow](https://github.com/bridgekat/tradingflow/) uses it.) Until the
//! first write, the context holds the value passed to
//! [`Builder::new`](typed::Builder::new); a segment that must tell the
//! build-time first run apart already has `is_first_run`.
//!
//! # Segment fusion
//!
//! Multi-threaded execution has a cost: every unit of work is a thread-pool
//! task, and waking a worker to run a trivial node can take much longer time
//! than the actual computation inside the node itself. Therefore, it is
//! desirable to fuse tiny segments together into a single node. *(Fusing heavy
//! nodes is equally possible, but fusing prevents parallel execution.)*
//!
//! The library provides combinator methods to fuse segments into larger
//! segments:
//!
//! | Combinator | Method | Meaning |
//! | --- | --- | --- |
//! | [`Id<T>`](typed::Id) | — | Identity: outputs are inputs unchanged. |
//! | [`Comp(f, g)`](typed::Comp) | [`f.then(g)`](typed::SegmentExt::then) | Composition: outputs `g(f(x))`. |
//! | [`Left<T, U>`](typed::Left) / [`Right<T, U>`](typed::Right) | — | Projection: outputs the first / second element of a pair. |
//! | [`Fork(f, g)`](typed::Fork) | [`f.fork(g)`](typed::SegmentExt::fork) | Fan-out: feed the same input to both, then pair outputs. |
//! | [`Par(f, g)`](typed::Par) | [`f.par(g)`](typed::SegmentExt::par) | Parallel composition: run `f`, `g` on a pair of inputs, then pair outputs. Equivalent to `Fork(Comp(Left, f), Comp(Right, g))`. |
//! | [`Arr::new(\|values, _\| …)`](typed::Arr::new) | — | Applies a stateful closure to the inputs. |
//!
//! However, point-free combinators get unreadable fast. As an alternative, the
//! library also provides the [`segment!`] macro, which is a DSL that compiles
//! down to combinators like the Arrow notation in Haskell[^1]:
//!
//! ```rust
//! use tradingflow_graph::typed::*;
//! use tradingflow_graph::segment;
//!
//! struct Add;
//!
//! impl Segment for Add {
//!     type Inputs = (Port<i64>, Port<i64>);
//!     type Outputs = Port<i64>;
//!     type Context = ();
//!     type State = ();
//!
//!     fn init(self) {}
//!
//!     fn compute<'a, 'b: 'a>(
//!         ((a_notify, a), (b_notify, b)): ((bool, i64), (bool, i64)),
//!         _context: &(),
//!         _state: &'b mut (),
//!         _is_first_run: bool
//!     ) -> (bool, i64) {
//!         (a_notify || b_notify, a + b)
//!     }
//! }
//!
//! let mut b = Builder::new(());
//! let (_, s) = b.source(Source::new(1));
//! let (_, t) = b.source(Source::new(2));
//!
//! // One fused node computing: c = b + a; d = c + a; e = d + a; result (e, c).
//! // Type annotation is needed on inputs and outputs.
//! let seg = segment!(|a: Port<i64>, b: Port<i64>| -> (Port<i64>, Port<i64>) {
//!     let c = Add @ (b, a);
//!     let e = Add @ (Add @ (c, a), a);
//!     (e, c)
//! });
//!
//! let (e, c) = b.segment(seg, (s, t));
//! ```
//!
//! `Seg @ wires` applies a segment (any [`Segment`](typed::Segment)-typed Rust
//! expression) to wires. Applications nest inside any wire expression and chain
//! right-associatively.
//!
//! # Safety
//!
//! The library uses `unsafe` internally, but the [typed] API should be safe to
//! use. For more details, the following runtime invariants are maintained
//! internally:
//!
//! - **Single writer per slot.** Each state and output slot has exactly one
//!   producing node; each input slot is scattered into by exactly that one
//!   producer. Concurrent `compute`s write disjoint slots.
//! - **Per-generation lifetime.** Output pointers stay valid as long as inputs
//!   and state are unchanged; [`passthrough`](typed::Operator::passthrough)'s
//!   `&State` forbids reallocation, and the per-node scratch buffers are only
//!   ever overwritten in place by the node's own run, keeping out-of-cone
//!   consumers' pointers live across generations.
//! - **Read guard.** Reading a slot after poking a source but before
//!   [`stabilize`](typed::Graph::stabilize) panics rather than dereferencing a
//!   possibly-stale forwarded pointer.
//! - **Poison on panic.** If a `compute` panics, downstream slots may hold
//!   dangling forwarded pointers, so the graph is **poisoned**: the pool catches
//!   the panic, settles the batch (no hang), re-raises out of `stabilize`, and
//!   every later `stabilize` or slot read panics. There is no recovery — treat
//!   the graph as dead. For *recoverable* failures, make the failure a value
//!   (e.g. a `Result<T, E>` cell) instead of panicking.
//!
//! Tests have been run on Miri to check for memory safety and common UBs.
//!
//! [^1]: <https://www.haskell.org/arrows/syntax.html>

pub mod core;
pub mod typed;

pub use tradingflow_macros::segment;
