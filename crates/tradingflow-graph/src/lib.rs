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
//! use tradingflow::data::*;
//! use tradingflow::graph::*;
//! use tradingflow::time::*;
//!
//! // A stream of `i64` values.
//! struct Sequence(Vec<i64>);
//!
//! impl Source for Sequence {
//!     type Instant = Instant;
//!     type Payload = i64;
//!     type Outputs = Port<Ref<i64>>;
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
//!     fn output(state: &mut i64) -> &i64 {
//!         // Output current stored value.
//!         state
//!     }
//!
//!     fn reset(state: &mut i64) -> &i64 {
//!         // The stored value is retained on reset.
//!         state
//!     }
//! }
//!
//! // An adder with two inputs.
//! struct Add;
//!
//! impl Segment for Add {
//!     type Inputs = (Port<Ref<i64>>, Port<Ref<i64>>);
//!     type Outputs = Port<Ref<i64>>;
//!     type Context = Instant;
//!     type State = i64;
//!
//!     fn init(self, _inputs: (&i64, &i64)) -> i64 {
//!         // The initial state.
//!         0
//!     }
//!
//!     fn compute<'a, 'b: 'a>(
//!         (a, b): (&'a i64, &'a i64),
//!         state: &'b mut i64,
//!         _instant: &Instant,
//!     ) -> &'a i64 {
//!         // Compute the sum.
//!         *state = *a + *b;
//!         state
//!     }
//!
//!     fn reset<'a, 'b: 'a>(
//!         _inputs: (&'a i64, &'a i64),
//!         state: &'b mut i64
//!     ) -> &'a i64 {
//!         // The sum is retained on reset.
//!         state
//!     }
//! }
//!
//! #[pollster::main]
//! async fn main() {
//!     // Create the thread pool.
//!     let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
//!
//!     // Build the graph.
//!     let mut b = Builder::new(UnixTime);
//!     let s = b.source(Sequence(vec![1, 2, 3, 4, 5]));
//!     let t = b.source(Sequence(vec![10, 20, 30, 40, 50]));
//!     let x = b.segment(Add, (s, t));
//!     let y = b.segment(Add, (s, x));
//!     let mut g = b.build();
//!
//!     // Update the source value and then recompute in parallel.
//!     g.run(&mut pool, |_, _| {}).await;
//!
//!     // Check final values.
//!     assert_eq!(*g.view(s), 5);
//!     assert_eq!(*g.view(t), 50);
//!     assert_eq!(*g.view(x), 55);
//!     assert_eq!(*g.view(y), 60);
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
//! - The [`Segment::reset`] method is called immediately after, to obtain the
//!   node's initial output values, and again after every [`Graph::stabilize`]
//!   to reset its outputs to quiescent states (either retained or `None`,
//!   depending on semantics).
//! - The [`Segment::compute`] method will be called on each subsequent
//!   [`Graph::stabilize`] with new inputs, to obtain updated output values.
//!   It can also access the graph context (typically the current timestamp)
//!   or mutate the node's state.
//!
//! # Interfaces and passing protocols
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
//! [`Pass`] protocol `V`, which allow passing custom lifetime-carrying
//! `Copy` views (e.g. slices or custom array views) to some underlying data.
//! The [`Val<T>`] and [`Ref<T>`] are built-in [`Pass`] protocols, representing
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
//! # Operator fusion
//!
//! Multi-threaded execution has a cost: waking a worker to run a trivial node
//! can take much longer time than the actual computation inside the node
//! itself. The scheduler therefore only creates parallel tasks for nodes
//! marked [`Segment::is_heavy`]; light nodes keep running in the same thread.
//!
//! Optionally, we can go one step further: fusing small operators together
//! into a single segment node lets the compiler monomorphize and inline the
//! whole chain, which still pays on the hottest fine-grained paths.
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
//! # use futures::Stream;
//! # use tradingflow::data::*;
//! # use tradingflow::graph::*;
//! # use tradingflow::time::*;
//! #
//! # // A stream of `i64` values.
//! # struct Sequence(Vec<i64>);
//! #
//! # impl Source for Sequence {
//! #     type Instant = Instant;
//! #     type Payload = i64;
//! #     type Outputs = Port<Ref<i64>>;
//! #     type State = i64;
//! #
//! #     fn init(self) -> (i64, impl Stream<Item = Event<i64, Instant>> + 'static) {
//! #         // The event stream.
//! #         let items = self.0.into_iter().map(|x| Event {
//! #             stamp: Stamp::Now,
//! #             payload: Some(x),
//! #         });
//! #         (0, futures::stream::iter(items))
//! #     }
//! #
//! #     fn write(payload: i64, _instant: Instant, state: &mut i64) -> usize {
//! #         // Write event payload into stored value.
//! #         *state = payload;
//! #         1
//! #     }
//! #
//! #     fn output(state: &mut i64) -> &i64 {
//! #         // Output current stored value.
//! #         state
//! #     }
//! #
//! #     fn reset(state: &mut i64) -> &i64 {
//! #         // The stored value is retained on reset.
//! #         state
//! #     }
//! # }
//! #
//! # // An adder with two inputs.
//! # struct Add;
//! #
//! # impl Segment for Add {
//! #     type Inputs = (Port<Ref<i64>>, Port<Ref<i64>>);
//! #     type Outputs = Port<Ref<i64>>;
//! #     type Context = Instant;
//! #     type State = i64;
//! #
//! #     fn init(self, _inputs: (&i64, &i64)) -> i64 {
//! #         // The initial state.
//! #         0
//! #     }
//! #
//! #     fn compute<'a, 'b: 'a>(
//! #         (a, b): (&'a i64, &'a i64),
//! #         state: &'b mut i64,
//! #         _instant: &Instant,
//! #     ) -> &'a i64 {
//! #         // Compute the sum.
//! #         *state = *a + *b;
//! #         state
//! #     }
//! #
//! #     fn reset<'a, 'b: 'a>(
//! #         _inputs: (&'a i64, &'a i64),
//! #         state: &'b mut i64
//! #     ) -> &'a i64 {
//! #         // The sum is retained on reset.
//! #         state
//! #     }
//! # }
//! #
//! use tradingflow::segment;
//!
//! let mut b = Builder::new(UnixTime);
//! let s = b.source(Sequence(vec![1, 2, 3, 4, 5]));
//! let t = b.source(Sequence(vec![10, 20, 30, 40, 50]));
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
//! Run Miri with **tree borrows** — under the default Stacked Borrows model,
//! crossbeam's deque/epoch internals produce third-party false positives
//! (retag violations in `crossbeam-epoch`), and its global garbage queue is
//! reported as leaked:
//!
//! ```text
//! MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks -Zmiri-disable-isolation" \
//!     cargo +nightly miri test -p tradingflow-graph
//! ```
//!
//! (`-Zmiri-disable-isolation` is only needed by doctests that read the
//! system clock via `Stamp::Now`.)
//!
//! [^1]: <https://www.haskell.org/arrows/syntax.html>

pub mod cb;
pub mod core;
pub mod driver;
pub mod pool;
pub mod typed;

pub use cb::SegmentExt;
pub use driver::{Builder, Event, Graph, Source, Stamp, Time};
pub use pool::Pool;
pub use typed::{Interface, NodeHandle, Pass, Port, PortHandle, Ports, Ref, Segment, Slice, Val};
