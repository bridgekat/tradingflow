//! `tradingflow` — Rust core for TradingFlow.
//!
//! This crate provides the performance-critical data structures and runtime
//! for the TradingFlow event-driven trading framework:
//!
//! * [`Store`](store::Store) — unified time-series storage.
//! * [`Operator`](operator::Operator) — trait for pure compute functions
//!   on Store views.
//! * [`Source`](source::Source) — trait for data sources.
//! * [`Scenario`] — DAG runtime with type-erased operator dispatch.
//!
//! When compiled with the `python` feature, the crate also produces a PyO3
//! `cdylib` exposing the runtime to Python.

pub mod operators;
pub mod sources;
pub mod store;

mod operator;
mod scenario;
mod source;
mod types;

// -- Public API --------------------------------------------------------------

pub use operator::Operator;
pub use scenario::{Handle, InputKindsHandles, Scenario};
pub use source::Source;
pub use store::{ElementView, ElementViewMut, SeriesView, Store};
pub use types::{InputKinds, Scalar};

#[cfg(feature = "python")]
mod bridge;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bridge::register(m)?;
    Ok(())
}
