//! Experimental wavefront runtime — proof-of-concept.
//!
//! This module contains a ground-up reimplementation of the core runtime
//! (`Operator`, `Source`, `Scenario`) designed around a **wavefront**
//! execution model:
//!
//! 1. **Node-axis parallelism** — independent DAG branches at a single
//!    timestamp `t` run concurrently on a worker pool.  Each node owns
//!    its `&mut State` and its freshly-allocated output slot, so no
//!    operator-level locks are needed.
//! 2. **Time-axis pipelining** — `t` and `t+1` overlap.  Writing `t+1`'s
//!    output no longer clobbers a reader of `t` in flight because each
//!    operator's outputs live in an auto-managed queue where slots are
//!    reference-counted and retired when the last reader releases them.
//!
//! The scheduling unit is **one node** per dispatch.  Subgraph units and
//! user-defined scheduling granularity are deferred to a future phase.
//!
//! # Architecture
//!
//! - [`operator`] — new [`Operator`](operator::Operator) trait and
//!   [`ErasedOperator`](operator::ErasedOperator).  Signature is
//!   deliberately compatible with the existing trait: the only PoC-level
//!   additions are `Output: Clone` (so the scheduler can mint fresh slots
//!   per timestamp) and a node-kind classification (see
//!   [`NodeKind`](operator::NodeKind)).
//! - [`source`] — re-exports the unchanged
//!   [`Source`](crate::source::Source) trait and
//!   [`ErasedSource`](crate::source::ErasedSource).
//! - [`data`] — re-exports unchanged [`Array`](crate::data::Array),
//!   [`Series`](crate::data::Series), [`Input`](crate::data::Input),
//!   [`InputTypes`](crate::data::InputTypes), cursors, and scalar types.
//! - [`queue`] — [`OutputQueue<T>`](queue::OutputQueue), the auto-GC
//!   output container backed by a `VecDeque<Arc<Slot<T>>>`.
//! - [`scenario`] — the new [`Scenario`](scenario::Scenario), its graph,
//!   node layout, scheduler, and ingest loop.
//! - [`operators`] — ported subset of built-in operators
//!   (`Const`, `Id`, `Add`, `Filter`, `Record`, `Lag`, `RollingMean`,
//!   `Clocked`, `ConcatSync`).
//! - [`sources`] — re-exports of `ArraySource` and the `clock` helper.

pub mod data;
pub mod operator;
pub mod operators;
pub mod queue;
pub mod scenario;
pub mod source;
pub mod sources;

pub use data::{
    Array, BitRead, Duration, FlatRead, FlatWrite, Input, InputTypes, Instant, PeekableReceiver,
    Scalar, Series, SliceProduced, SliceRefs,
};
pub use operator::{ErasedOperator, NodeKind, Operator};
pub use queue::OutputQueue;
pub use scenario::{Handle, Scenario};
pub use source::{ErasedSource, Source};
