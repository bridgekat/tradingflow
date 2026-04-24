//! `tradingflow` — Rust core for TradingFlow.
//!
//! This crate provides the performance-critical data structures and runtime
//! for the TradingFlow event-driven computation framework.
//!
//! # Core data types
//!
//! * [`Array`] — dense N-dimensional array in standard (C-contiguous) layout.
//!   Parameterised by a [`Scalar`] element type.
//! * [`Series`] — append-only time series with temporal (as-of) lookups.
//!   Each element is a uniformly-shaped `Array`-compatible slice.
//! * [`Schema`] — bidirectional name ↔ position mapping for labelling array
//!   axes.
//!
//! # Traits
//!
//! * [`Source`] — asynchronous data source feeding events into the graph via
//!   historical and live channels. Provides an optional
//!   [`estimated_event_count`](Source::estimated_event_count) method for
//!   progress reporting. Type-erased form: [`ErasedSource`].
//! * [`Operator`] — synchronous computation node that reads typed inputs and
//!   writes a typed output. Type-erased form: [`ErasedOperator`].  The
//!   `compute` method receives two structurally parallel hierarchical trees:
//!   `inputs: <Inputs as InputTypes>::Refs<'_>` and
//!   `produced: <Inputs as InputTypes>::Produced<'_>`.  Slice branches
//!   expose lazy views; sized tuples produce stack-allocated compounds.
//!
//! # Runtime
//!
//! * [`Scenario`] — computation graph with type-erased dispatch. Nodes are
//!   connected via typed [`Handle`](scenario::Handle)s. Execution is driven
//!   by [`Scenario::flush`] (manual) or [`Scenario::run`] (async event loop).
//!
//! # Modules
//!
//! * [`data`] — primitive data types and operator-input trait machinery:
//!   * [`data::array`] — `Array`.
//!   * [`data::series`] — `Series`.
//!   * [`data::time`] — `Instant` (SI nanoseconds since 1970-01-01 00:00:00
//!     TAI) and `Duration` (SI nanoseconds).
//!   * [`data::inputs`] — `InputTypes`, `Input<T>`, `FlatRead` / `BitRead`
//!     cursors, and related machinery.
//!   * Flat: `Scalar`, `PeekableReceiver`.
//! * [`source`] — `Source` trait, `ErasedSource`.
//! * [`operator`] — `Operator` trait, `ErasedOperator`.
//! * [`scenario`] — `Scenario`, `Handle`, `InputTypesHandles`.
//! * [`operators`] — built-in operators: structural (`Const`, `Id`, `Filter`,
//!   `Where`, `Select`, `Concat`, `Stack`, `Cast`), series (`Record`, `Last`,
//!   `Lag`), element-wise numeric ([`operators::num`]), rolling-window
//!   ([`operators::rolling`]), and stock-specific ([`operators::stocks`]).
//! * [`sources`] — built-in data sources: `ArraySource`, `CsvSource`,
//!   `IterSource`, and the `clock` trigger source.  Calendar-aligned
//!   clock schedules live in the Python wrapper.
//! * [`utils`] — `Schema`.
//! * [`bridge`] — PyO3 bindings (behind the `python` feature).
//!
//! When compiled with the `python` feature, the crate also produces a PyO3
//! `cdylib` exposing the runtime to Python.

pub mod data;
pub mod experimental;
pub mod operator;
pub mod operators;
pub mod scenario;
pub mod source;
pub mod sources;
pub mod utils;

pub use data::{
    Array, BitRead, Duration, FlatRead, FlatWrite, Input, InputTypes, Instant, PeekableReceiver,
    Scalar, Series, SliceProduced, SliceRefs, tai_to_utc, utc_to_tai,
};
pub use operator::{ErasedOperator, Operator};
pub use operators::Clocked;
pub use scenario::Scenario;
pub use source::{ErasedSource, Source};
pub use utils::Schema;

#[cfg(feature = "python")]
pub mod bridge;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bridge::register(m)?;
    Ok(())
}
