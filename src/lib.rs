#![doc = include_str!("../README.md")]
//!
//! # Core data types
//!
//! * [`Array`] — dense N-dimensional array in standard (C-contiguous) layout,
//!   parameterised by a [`Scalar`] element type. [`ArraySlice`] is its borrowed,
//!   zero-copy view (the edge currency of the computation engine).
//! * [`Series`] — append-only time series with temporal (as-of) lookups; each
//!   element is a uniformly-shaped `Array`-compatible slice.
//! * [`Schema`] — bidirectional name ↔ position mapping for labelling array axes.
//!
//! # Engine
//!
//! Operators ([`operators`]) are implemented directly on
//! [`flowgraph::typed::Operator`] / [`Segment`](flowgraph::typed::Segment) (no
//! TradingFlow-side operator trait), composed with the `flowgraph::segment!`
//! fusion macro, and driven by [`Scenario`] / [`Session`] ([`scenario`]) over an
//! async source event loop. A [`Source`] feeds events into source cells via
//! historical + live channels; time is threaded out-of-band through a shared
//! [`Clock`](operators::Clock).
//!
//! # Modules
//!
//! * [`data`] — primitive containers: [`Array`] / [`ArraySlice`], [`Series`],
//!   [`Instant`] / [`Duration`] (SI nanoseconds since the 1970 TAI epoch),
//!   plus [`Scalar`] and [`PeekableReceiver`].
//! * [`source`] — the [`Source`] trait.
//! * [`sources`] — built-in data sources: `ArraySource`, `CsvSource`,
//!   `IterSource`, the columnar panel sources, and the `clock` trigger.
//! * [`operators`] — the operator library. Behind the `python` feature it also
//!   runs Python operators on an embedded interpreter.
//! * [`scenario`] — the engine: [`Scenario`] (graph builder) and [`Session`]
//!   (the async event-loop driver).
//! * [`utils`] — [`Schema`].

pub mod data;
// [array-view-refactor] operators/scenario/sources are mid-migration to the
// const-rank `ArrayView` currency; temporarily gated so the `data` foundation
// lands + tests green on its own. Re-enabled file-by-file in stages 2–5.
#[cfg(any())]
pub mod operators;
#[cfg(any())]
pub mod scenario;
pub mod source;
#[cfg(any())]
pub mod sources;
pub mod utils;

pub use data::{
    Array, ArrayView, Duration, Instant, PeekableReceiver, Retention, Scalar, Series, Shape,
    tai_to_utc, utc_to_tai,
};
#[cfg(any())]
pub use scenario::{Scenario, Session, ShutdownFlag};
pub use source::Source;
pub use utils::Schema;

// The `flowgraph` vocabulary the operator layer is written in, re-exported for
// downstream graph-building code. (`flowgraph::typed::Id` is NOT re-exported —
// [`operators::Id`] is the structural identity operator; reach the combinator
// via its full path.)
pub use flowgraph::typed::{
    Arena, Handle, Interface, InterfaceHandles, Operator, Port, RefPort, RefPorts, RefSource,
    RefViewPort, RefViewPorts, Scalar as ScalarValue, Segment, SegmentExt, Source as ValueSource,
    SourceHandle, ValueView, ViewPort, ViewSource,
};
