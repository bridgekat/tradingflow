#![doc = include_str!("../README.md")]
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
//! The engine lives in `flowgraph` itself: graph building plus event feeds is
//! [`flowgraph::ingest::Builder`], and the self-driving graph (async
//! timestamp-merge event loop) is [`flowgraph::ingest::Graph`]. TradingFlow
//! instantiates them at the TAI [`Instant`] and the real [`WallClock`] as
//! [`Scenario`] / [`Session`].
//!
//! There is **no builder extension trait**: every node registers through
//! flowgraph's own [`push`](flowgraph::typed::Builder::push) /
//! [`push_source`](flowgraph::typed::Builder::push_source), applied to one of
//! the operator library's free constructors. Even the view-currency bridges
//! and the Python operators are constructors
//! ([`as_view`](operators::as_view), [`own`](operators::own),
//! `py_class_operator`); a record is
//! [`record`](operators::record) / [`record_bounded`](operators::record_bounded),
//! taking the driver clock ([`Builder::time`](flowgraph::ingest::Builder::time))
//! as a parameter.
//!
//! TradingFlow deliberately re-exports **nothing** from `flowgraph`: the
//! graph-building vocabulary (`Handle`, `ViewPort`, `Segment`, the `segment!`
//! macro, …) comes from adding `flowgraph` as a direct dependency alongside
//! `tradingflow`.
//!
//! Operators ([`operators`]) are implemented directly on
//! [`flowgraph::typed::Operator`] / [`Segment`](flowgraph::typed::Segment) (no
//! TradingFlow-side operator trait) and composed with the `flowgraph::segment!`
//! fusion macro. A data source implements [`flowgraph::ingest::EventSource`],
//! streaming timestamped events into its source cell; time is threaded
//! out-of-band through the driver-advanced
//! [`SharedTime`](flowgraph::ingest::SharedTime) cell ([`operators::Clock`] is
//! its [`Instant`] instantiation).
//!
//! # Modules
//!
//! * [`clock`] — the real TAI [`WallClock`] driving the event loop.
//! * [`data`] — primitive containers: [`Array`] / [`ArrayView`], [`Series`],
//!   [`Instant`] / [`Duration`] (SI nanoseconds since the 1970 TAI epoch),
//!   plus [`Scalar`].
//! * [`sources`] — built-in data sources: `ArraySource`, `IterSource`, the
//!   columnar panel sources (`ParquetPanelSource` / `ParquetFinancialReportPanelSource`), and
//!   the `clock` trigger.
//! * [`operators`] — the operator library. Behind the `python` feature it also
//!   runs Python operators on an embedded interpreter.
//! * [`utils`] — [`Schema`].

pub mod clock;
pub mod data;
pub mod operators;
pub mod sources;
pub mod utils;

pub use clock::WallClock;
pub use data::{
    Array, ArrayView, Duration, Instant, Retention, Scalar, Series, SeriesView, Shape, tai_to_utc,
    utc_to_tai,
};
pub use utils::Schema;

/// The strategy graph builder: [`flowgraph::ingest::Builder`] instantiated at
/// the TAI [`Instant`] + real [`WallClock`]. Sources register via its inherent
/// `add_source` ([`EventSource`](flowgraph::ingest::EventSource) impls),
/// constant cells via the deref'd
/// [`push_source`](flowgraph::typed::Builder::push_source), and every segment
/// via the deref'd [`push`](flowgraph::typed::Builder::push) applied to an
/// [`operators`] constructor. `build()` produces a [`Session`].
pub type Scenario = flowgraph::ingest::Builder<Instant, WallClock>;

/// A live execution: [`flowgraph::ingest::Graph`] instantiated at the TAI
/// [`Instant`] + real [`WallClock`]. `run(|session, ts| …)` drives the event
/// loop to exhaustion (progress via `events()` / `estimated_event_count()`);
/// the deref'd [`ref_view`](flowgraph::typed::Graph::ref_view) reads results
/// back.
pub type Session = flowgraph::ingest::Graph<Instant, WallClock>;
