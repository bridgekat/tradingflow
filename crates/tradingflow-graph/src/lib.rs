//! This crate implements a multithreaded executor for static computation
//! graphs.
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
//! Here is an example graph which takes in streams `s` and `t` of values,
//! and computes `x := s + t` and `y := s + x` whenever `s` or `t` changes:
//!
//! ```rust
//! use futures::Stream;
//! use tradingflow::clock::UnixClock;
//! use tradingflow::data::*;
//! use tradingflow::graph::*;
//!
//! /// A stream of `i64` values.
//! struct SeqSource(Vec<i64>);
//!
//! impl Source for SeqSource {
//!     type Instant = Instant;
//!     type Payload = i64;
//!     type Outputs = Port<Val<i64>>;
//!     type State = i64;
//!
//!     fn init(self) -> (i64, impl Stream<Item = Event<i64, Instant>> + 'static) {
//!         // The event stream.
//!         let items = self.0.into_iter().map(|x| Event {
//!             stamp: Stamp::Now,
//!             payload: Some(x),
//!         });
//!         (0, futures::stream::iter(items))
//!     }
//!
//!     fn write(payload: i64, _instant: Instant, state: &mut i64) -> usize {
//!         // Write event payload into stored value.
//!         *state = payload;
//!         1
//!     }
//!
//!     fn output(state: &mut i64) -> (bool, i64) {
//!         // Output current stored value.
//!         (true, *state)
//!     }
//! }
//!
//! /// A stateless adder with two inputs.
//! struct Add;
//!
//! impl Segment for Add {
//!     type Inputs = (Port<Val<i64>>, Port<Val<i64>>);
//!     type Outputs = Port<Val<i64>>;
//!     type Context = Instant;
//!     type State = ();
//!
//!     fn init(self, _inputs: ((bool, i64), (bool, i64))) {}
//!
//!     fn output<'a, 'b: 'a>(
//!         _inputs: ((bool, i64), (bool, i64)),
//!         _state: &'b mut (),
//!     ) -> (bool, i64) {
//!         // The initial placeholder output.
//!         (false, 0)
//!     }
//!
//!     fn compute<'a, 'b: 'a>(
//!         ((a_notify, a), (b_notify, b)): ((bool, i64), (bool, i64)),
//!         _state: &'b mut (),
//!         _instant: &Instant,
//!     ) -> (bool, i64) {
//!         // Compute the sum.
//!         (a_notify || b_notify, a + b)
//!     }
//! }
//!
//! #[pollster::main]
//! async fn main() {
//!     // Create the thread pool.
//!     let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
//!
//!     // Build the graph.
//!     let mut b = Builder::new(UnixClock);
//!     let s = b.source(SeqSource(vec![1, 2, 3, 4, 5]));
//!     let t = b.source(SeqSource(vec![10, 20, 30, 40, 50]));
//!     let x = b.segment(Add, (s, t));
//!     let y = b.segment(Add, (s, x));
//!     let mut g = b.build();
//!
//!     // Update the source value and then recompute in parallel.
//!     g.run(&mut pool, |_, _| {}).await;
//!
//!     // Check final values.
//!     assert_eq!(g.view(s), 5);
//!     assert_eq!(g.view(t), 50);
//!     assert_eq!(g.view(x), 55);
//!     assert_eq!(g.view(y), 60);
//! }
//! ```
//!
//! # Constructing graphs
//!
//! A [`Builder`] is used to construct a [`Graph`].
//!
//! - [`Builder::source`] adds a source stream to the graph, returning its
//!   output port handle.
//! - [`Builder::segment`] adds a segment node to the graph, taking input port
//!   handles and returning its output port handles.
//! - [`Builder::build`] finalizes into a [`Graph`].
//!
//! A [`Graph`] represents a complete computation graph with source streams.
//!
//! - [`Graph::step`] ingests a single batch of events from source streams,
//!   writing the new data to their corresponding source ports.
//! - [`Graph::stabilize`] propagates changes throughout the graph.
//! - [`Graph::run`] ingests and propagates repeatedly, calling a custom
//!   closure after each batch, until all sources are exhausted.
//! - [`Graph::view`] reads values on wires. This can only be used in-between
//!   batches.
//!
//! # Creating sources
//!
//! Each source is a stream of [`Event`]s, associated onto a source node.
//! It must implement [`Source`], which declares its events and outputs,
//! its timestamp type, its mutable state, and the following methods:
//!
//! - The [`Source::size_hint`] method should return the total number of
//!   events in the stream, or `None` if unbounded or unknown.
//! - The [`Source::init`] method is called once during graph construction,
//!   to create the node's state and the event stream.
//! - The [`Source::output`] method will be called immediately after *and*
//!   on each subsequent [`Graph::stabilize`], to obtain current output values.
//! - The [`Source::write`] method will be called on each subsequent
//!   [`Graph::step`] with new events, to write the events into the node.
//!
//! # Creating segments
//!
//! Each segment is a single node of scheduling. It must implement [`Segment`],
//! which declares its inputs and outputs, its graph context (must be the
//! timestamp type), its mutable state, and the following methods:
//!
//! - The [`Segment::init`] method is called once during graph construction,
//!   to create the node's state.
//! - The [`Segment::output`] method is called immediately after (possibly
//!   multiple times), to obtain initial output values (typically placeholders).
//! - The [`Segment::compute`] method will be called on each subsequent
//!   [`Graph::stabilize`] with new inputs, to obtain updated output values.
//!   It can also access the graph context (typically the current timestamp)
//!   or mutate the node's state.
//!
//! Most users should implement the similar [`Operator`] trait instead,
//! which provides a more natural semantics (see below). [`Segment`] is mainly
//! used for composition; [`Operator`] is used for individual operations.
//!
//! # Interfaces and passing policies
//!
//! The associated types [`Segment::Inputs`] and [`Segment::Outputs`] define
//! the [`Interface`] of a segment. They can be constructed from the following
//! basic building blocks:
//!
//! - [`Port<Val<T>>`] — a single pass-by-value port. It carries `(bool, T)`
//!   where `T: Copy + Send + Sync`.
//! - [`Ports<Val<T>>`] — a dynamic-length group of pass-by-value ports. It
//!   carries `(&[bool], &[T])` where `T: Copy + Send + Sync`, and is
//!   compatible with a group of [`Port<Val<T>>`]s.
//! - [`Port<Ref<T>>`] — a single pass-by-reference port. It carries
//!   `(bool, &T)` where `T: Sync`.
//! - [`Ports<Ref<T>>`] — a dynamic-length group of pass-by-reference ports.
//!   It carries `(&[bool], &[&T])` where `T: Sync`, and is compatible
//!   with a group of [`Port<Ref<T>>`]s.
//! - Arbitrarily nested tuples of the above (each branch up to arity 12).
//!
//! In fact, the [`Port<V>`] and [`Ports<V>`] markers can be used over any
//! [`Pass`] policy `V`, which allow passing custom lifetime-carrying
//! `Copy` views (e.g. slices or custom array views) to some underlying data.
//! The [`Val<T>`] and [`Ref<T>`] are built-in [`Pass`] policies, representing
//! simple pass-by-value and pass-by-reference. There is also [`Slice<T>`]
//! allowing passing a slice across a single port.
//!
//! For dynamic-length [`Ports`] groups, the length must remain fixed after the
//! first run, so that we have a well-defined static computation graph.
//! Violations will be caught and panic at runtime.
//!
//! Producing a [`Ports`] output requires creating slices with lifetime `'a`
//! matching the inputs. This allows for simple forwarding of input references,
//! but it also creates difficulty for a node that computes its own values:
//! we have to put the arrays somewhere, but the node state must have static
//! lifetime. To address this difficulty, use a lifetime-erased bump arena
//! (such as [`bumpalo`](https://crates.io/crates/bumpalo)) in the node state
//! as a scratch buffer storing the arrays on each recompute. The arena can be
//! cleared at the beginning of each recompute, so that buffer size is kept
//! bounded.
//!
//! > Interface values are constrained to [`Copy`] because they are required to
//! > have trivial [`Drop`] implementations. Data ownership is always inside
//! > node states and never passed through an interface; only simple scalar
//! > values, references and views do. This encourages pass-by-reference for
//! > complex data types and simplifies the internal implementation, but may be
//! > less ergonomic in some cases.
//!
//! # Notification flags
//!
//! Each port carries a `bool` flag alongside the value or reference, indicating
//! whether the value is a *notification*. At each generation, an unmodified
//! output value can have its flag set to `false` (no notify), so a downstream
//! node can choose to skip heavy computation.
//!
//! ```rust
//! use tradingflow::data::*;
//! use tradingflow::graph::*;
//!
//! struct Abs;
//!
//! impl Segment for Abs {
//!     type Inputs = Port<Val<i64>>;
//!     type Outputs = Port<Val<i64>>;
//!     type Context = Instant;
//!     type State = i64;
//!
//!     fn init(self, _inputs: (bool, i64)) -> i64 {
//!         // The initial state.
//!         0
//!     }
//!
//!     fn output<'a, 'b: 'a>(
//!         _inputs: (bool, i64),
//!         _state: &'b mut i64
//!     ) -> (bool, i64) {
//!         // The initial placeholder output.
//!         (false, 0)
//!     }
//!
//!     fn compute<'a, 'b: 'a>(
//!         (x_notify, x): (bool, i64),
//!         state: &'b mut i64,
//!         _instant: &Instant,
//!     ) -> (bool, i64) {
//!         if x_notify {
//!             // Input notified, recompute.
//!             let output = x.abs();
//!             let changed = output != *state; // Notify downstream only when |x| actually changed.
//!             *state = output;
//!             (changed, *state)
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
//! executor sets the flags of modified source nodes, and skips a node
//! completely if none of its upstream source nodes were modified. This makes
//! stabilization after a sparse update touches only a fraction of the graph.
//!
//! The [`Operator`] trait supports fine-grained skipping. It splits the
//! compute function of [`Segment`] into two branches:
//!
//! - [`Operator::compute`] is called only when at least one input is notified;
//! - [`Operator::passthrough`] is called otherwise. One can cache the previous
//!   output value in the node state, and return it directly on passthrough.
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
//! | Combinator | Method (via [`cb::SegmentExt`]) | Meaning |
//! | --- | --- | --- |
//! | [`cb::Id<T>`] | — | Identity: outputs are inputs unchanged. |
//! | [`cb::Comp<F, G>`] | `f.then(g)` | Composition: outputs `g(f(x))`. |
//! | [`cb::Left<T, U>`] / [`cb::Right<T, U>`] | — | Projection: outputs the first / second element of a pair. |
//! | [`cb::Fork<F, G>`] | `f.fork(g)` | Fan-out: feed the same input to both, then pair outputs. |
//! | [`cb::Par<F, G>`] | `f.par(g)` | Parallel composition: run `f`, `g` on a pair of inputs, then pair outputs. Equivalent to `Fork<Comp<Left, F>, Comp<Right, G>>`. |
//! | [`cb::Arr`] | — | Applies a stateless closure to the inputs. |
//!
//! However, point-free combinators get unreadable fast. As an alternative, the
//! `tradingflow-macros` crate provides the `segment!` macro, which is a DSL
//! that compiles down to combinators like the Arrow notation in Haskell[^1]:
//!
//! ```rust
//! use tradingflow::clock::UnixClock;
//! use tradingflow::data::*;
//! use tradingflow::graph::*;
//! use tradingflow::segment;
//! use tradingflow::sources::basic::*;
//!
//! /// An adder which passes inputs and outputs by reference.
//! ///
//! /// The `Operator` trait is a convenience wrapper around `Segment`, which
//! /// calls `compute` only when at least one input has notified, and calls
//! /// `passthrough` otherwise. This is a common pattern.
//! struct Add;
//!
//! impl Operator for Add {
//!     type Inputs = (Port<Ref<i64>>, Port<Ref<i64>>);
//!     type Outputs = Port<Ref<i64>>;
//!     type Context = Instant;
//!     type State = i64;
//!
//!     fn init(self, _inputs: ((bool, &i64), (bool, &i64))) -> i64 {
//!         // The initial state.
//!         0
//!     }
//!
//!     fn passthrough<'a, 'b: 'a>(
//!         _inputs: ((bool, &'a i64), (bool, &'a i64)),
//!         state: &'b mut i64
//!     ) -> (bool, &'a i64) {
//!         // A simple forwarding.
//!         (false, state)
//!     }
//!
//!     fn compute<'a, 'b: 'a>(
//!         ((_, a), (_, b)): ((bool, &'a i64), (bool, &'a i64)),
//!         state: &'b mut i64,
//!         _instant: &Instant,
//!     ) -> (bool, &'a i64) {
//!         // Compute the sum.
//!         *state = *a + *b;
//!         (true, state)
//!     }
//! }
//!
//! let mut b = Builder::new(UnixClock);
//! let s = b.source(vec_source(vec![(Instant::EPOCH, 1)]));
//! let t = b.source(vec_source(vec![(Instant::EPOCH, 2)]));
//!
//! // One fused node computing: c = b + a; d = c + a; e = d + a; result (e, c).
//! // Type annotation is needed on inputs and outputs.
//! let seg = segment!(|a: Port<Ref<i64>>, b: Port<Ref<i64>>| -> (Port<Ref<i64>>, Port<Ref<i64>>) {
//!     let c = Add @ (b, a);
//!     let e = Add @ (Add @ (c, a), a);
//!     (e, c)
//! });
//!
//! let (e, c) = b.segment(seg, (s, t));
//! ```
//!
//! The expression `seg @ wires` applies a segment (can be any Rust expression
//! whose type `T` implements the [`Segment`] trait) to wires. Applications
//! nest inside any wire expression and chain right-associatively.
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
//!   and state are unchanged; the per-node scratch buffers are only
//!   ever overwritten in place by the node's own run, keeping out-of-cone
//!   consumers' pointers live across generations.
//! - **Read guard.** Reading a slot after poking a source but before
//!   [`stabilize`](typed::Graph::stabilize) panics rather than dereferencing a
//!   possibly-stale forwarded pointer.
//! - **Poison on panic.** If a `compute` panics, downstream slots may hold
//!   dangling forwarded pointers, so the graph is **poisoned**: the pool
//!   catches the panic, settles the batch (no hang), re-raises out of
//!   `stabilize`, and every later `stabilize` or slot read panics.
//!   There is no recovery — treat the graph as dead. For *recoverable*
//!   failures, make the failure a value (e.g. a `Result<T, E>` cell) instead
//!   of panicking.
//!
//! Tests have been run on Miri to check for memory safety and common UBs.
//!
//! [^1]: <https://www.haskell.org/arrows/syntax.html>

pub mod cb;
pub mod core;
pub mod driver;
pub mod pool;
pub mod typed;

pub use cb::SegmentExt;
pub use driver::{Builder, Clock, Event, Graph, Source, Stamp};
pub use pool::Pool;
pub use typed::{
    Interface, NodeHandle, Operator, Pass, Port, PortHandle, Ports, Ref, Segment, Slice, Val,
};
