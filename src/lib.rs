//! `tradingflow` — Rust core for TradingFlow.
//!
//! Performance-critical data structures and the synchronous, parallel
//! computation engine (the [`flow`] module, built on `flowgraph`).
//!
//! # Core data types
//!
//! * [`Array`] — dense N-dimensional array in standard (C-contiguous) layout,
//!   parameterised by a [`Scalar`] element type. [`ArraySlice`] is its borrowed,
//!   zero-copy view (the edge currency of the [`flow`] engine).
//! * [`Series`] — append-only time series with temporal (as-of) lookups; each
//!   element is a uniformly-shaped `Array`-compatible slice.
//! * [`Schema`] — bidirectional name ↔ position mapping for labelling array axes.
//!
//! # Engine
//!
//! [`flow`] is the engine. Operators are implemented directly on
//! [`flowgraph::typed::Operator`] / [`Segment`](flowgraph::typed::Segment) (no
//! TradingFlow-side operator trait), composed with the `flowgraph::segment!`
//! fusion macro, and driven by [`flow::Scenario`] / [`flow::Session`] over an
//! async source event loop. A [`Source`] feeds events into source cells via
//! historical + live channels; time is threaded out-of-band through a shared
//! [`Clock`](flow::Clock).
//!
//! # Modules
//!
//! * [`data`] — primitive containers: [`Array`] / [`ArraySlice`], [`Series`],
//!   [`Instant`] / [`Duration`] (SI nanoseconds since the 1970 TAI epoch),
//!   plus [`Scalar`] and [`PeekableReceiver`].
//! * [`source`] — the [`Source`] trait.
//! * [`sources`] — built-in data sources: `ArraySource`, `CsvSource`,
//!   `IterSource`, the columnar panel sources, and the `clock` trigger.
//! * [`flow`] — the computation engine and operator library. Behind the
//!   `pyflow` feature it also runs Python operators on an embedded interpreter.
//! * [`utils`] — [`Schema`].

pub mod data;
pub mod flow;
pub mod source;
pub mod sources;
pub mod utils;

pub use data::{
    Array, ArraySlice, Duration, Instant, PeekableReceiver, Scalar, Series, tai_to_utc, utc_to_tai,
};
pub use source::Source;
pub use utils::Schema;
