// The project README is the crate's front page (it lives at the repo root,
// beside the examples and the Python package it also documents).
#![doc = include_str!("../../../README.md")]
//!
//! # Core data types
//!
//! * [`Array`] — dense N-dimensional array in standard (C-contiguous) layout,
//!   parameterised by a [`Scalar`] element type. [`ArrayView`] is its borrowed,
//!   zero-copy strided view (the edge currency of the computation engine).
//! * [`Series`] — append-only time series with temporal (as-of) lookups; each
//!   element is a uniformly-shaped `Array`-compatible slice.
//! * [`Schema`] — bidirectional name ↔ position mapping for labelling array axes.
//!
//! # Engine
//!
//! The DAG engine — parallel sparse restabilization plus the graph-level
//! context — lives in `tradingflow-graph`; the driver on top of it
//! lives here, in [`ingest`]: [`Scenario`] couples a graph builder with its
//! event feeds, and [`Session`] is the self-driving graph (async
//! timestamp-merge event loop) over the TAI [`Instant`] and a wall
//! [`Clock`](ingest::Clock) (the real [`WallClock`] by default).
//!
//! There is **no builder extension trait**: every node registers through
//! the engine's own [`push`](graph::Builder::push) /
//! [`push_source`](graph::Builder::push_source), applied to one of
//! the operator library's free constructors. Even the view-currency bridges
//! and the Python operators are constructors
//! ([`as_view`](operators::as_view), [`own`](operators::own),
//! `py_class_operator`); a record is
//! [`record()`](operators::record) /
//! [`record_bounded(retention)`](operators::record_bounded) — no clock
//! parameter, because event time reaches it another way (below).
//!
//! TradingFlow is the **only dependency a strategy crate needs**: the
//! graph-building vocabulary ([`Handle`](graph::Handle),
//! [`ViewPort`](graph::ViewPort), [`Segment`](graph::Segment), …) is
//! re-exported as [`graph`], and [`segment!`] is the facade form of the
//! fusion macro (its expansions resolve through [`graph`], so no engine
//! dependency is required). The engine itself is the sibling
//! `tradingflow-graph` workspace crate.
//!
//! Operators ([`operators`]) are implemented directly on
//! [`Operator`](graph::Operator) / [`Segment`](graph::Segment) (no
//! TradingFlow-side operator trait) and composed with the [`segment!`]
//! fusion macro. A data source implements [`ingest::EventSource`], streaming
//! timestamped events into its source cell.
//!
//! # Event time is ambient
//!
//! Every operator declares `Context = Instant` — the engine's *graph-level
//! context*, which the driving [`Session`] sets to the current batch's event
//! time before each `stabilize`. So an operator that stamps time
//! ([`Record`](operators::Record)) is simply *handed* the timestamp in its
//! `compute`: nothing is threaded through construction, no constructor takes a
//! clock, and a `segment!` formula captures nothing. Time is not a graph
//! dependency either — writing the context dirties no cone, so a time-reading
//! operator still recomputes only when its own inputs notify.
//!
//! Before the first batch the context holds `Instant::MIN`, the floor set by
//! `Scenario::new(WallClock)`; an operator that must tell that build call apart
//! uses its `init` flag, not the timestamp.
//!
//! Two distinct notions carry a clock-like name, and they are not the same
//! thing: [`WallClock`] is the wall clock that *drives* the event loop (an
//! impl of [`ingest::Clock`]); [`sources::pulse()`] is a *source*
//! emitting bare `()` triggers at given timestamps, which clock-gated
//! operators take as a leading input port.
//!
//! # Modules
//!
//! * [`data`] — primitive containers: [`Array`] / [`ArrayView`], [`Series`],
//!   [`Instant`] / [`Duration`] (SI nanoseconds since the 1970 TAI epoch),
//!   plus [`Scalar`].
//! * [`graph`] — the engine's graph-building vocabulary, re-exported
//!   ([`Segment`](graph::Segment), ports, handles, combinators), plus the
//!   [`segment!`] fusion macro.
//! * [`ingest`] — the event-ingestion driver: [`Scenario`] / [`Session`], the
//!   timestamp-ordered merge ([`Queue`](ingest::Queue)),
//!   [`EventSource`](ingest::EventSource), and the real TAI [`WallClock`].
//! * [`sources`] — built-in data sources: `ArraySource`, `IterSource`, the
//!   columnar panel sources (`ParquetPanelSource` / `ParquetFinancialReportPanelSource`), and
//!   the [`pulse()`](sources::pulse()) trigger.
//! * [`operators`] — the operator library. Behind the `python` feature it also
//!   runs Python operators on an embedded interpreter.
//! * [`utils`] — [`Schema`].

pub mod data;
pub mod graph;
pub mod ingest;
pub mod operators;
pub mod sources;
pub mod utils;

pub use data::{
    Array, ArrayView, Duration, Instant, Retention, Scalar, Series, SeriesView, Shape, tai_to_utc,
    utc_to_tai,
};
pub use ingest::{Scenario, Session, WallClock};
pub use utils::Schema;
