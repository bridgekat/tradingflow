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
//! * [`Schema`] — bidirectional name↔position mapping for labelling array axes.
//!
//! # Traits
//!
//! * [`Source`] — asynchronous data source feeding events into the graph via
//!   historical and live channels. Type-erased form: [`ErasedSource`].
//! * [`Operator`] — synchronous computation node that reads typed inputs and
//!   writes a typed output. Type-erased form: [`ErasedOperator`].
//!
//! # Runtime
//!
//! * [`Scenario`] — DAG runtime with type-erased dispatch. Nodes are
//!   connected via typed [`Handle`](scenario::Handle)s. Execution is driven
//!   by [`Scenario::flush`] (manual) or [`Scenario::run`] (async POCQ event
//!   loop).
//!
//! # Modules
//!
//! * [`array`] — `Array`, `Scalar`.
//! * [`series`] — `Series`, `SeriesIter`.
//! * [`source`] — `Source` trait, `ErasedSource`, `PeekableReceiver`.
//! * [`operator`] — `Operator` trait, `ErasedOperator`.
//! * [`scenario`] — `Scenario`, `Handle`, `InputTypesHandles`.
//! * [`types`] — `InputTypes` trait and tuple/slice implementations.
//! * [`operators`] — built-in operators: structural (`Const`, `Id`, `Filter`,
//!   `Where`, `Select`, `Concat`, `Stack`, `Cast`), series (`Record`, `Last`,
//!   `Lag`), element-wise numeric ([`operators::num`]), and rolling-window
//!   ([`operators::rolling`]).
//! * [`sources`] — built-in data sources: `ArraySource`, `CsvSource`,
//!   `IterSource`, and clock sources (`clock`, `daily_clock`,
//!   `monthly_clock`).
//! * [`utils`] — `Schema`.
//! * [`bridge`] — PyO3 bindings (behind the `python` feature).
//!
//! When compiled with the `python` feature, the crate also produces a PyO3
//! `cdylib` exposing the runtime to Python.

pub mod array;
pub mod operator;
pub mod operators;
pub mod scenario;
pub mod series;
pub mod source;
pub mod sources;
pub mod types;
pub mod utils;

pub use array::{Array, Scalar};
pub use operator::{ErasedOperator, Operator};
pub use scenario::Scenario;
pub use series::Series;
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
