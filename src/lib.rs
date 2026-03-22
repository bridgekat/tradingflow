//! `tradingflow` — Rust core for TradingFlow.
//!
//! This crate provides the performance-critical data structures and runtime
//! for the TradingFlow event-driven trading framework:
//!
//! * [`Array`](array::Array) — dense, dynamically-shaped array.
//! * [`Series`](series::Series) — append-only time series.
//! * [`Operator`](operator::Operator) — trait for compute functions.
//! * [`Source`](source::Source) — trait for data sources.
//! * [`Scenario`](scenario::Scenario) — DAG runtime with type-erased dispatch.
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
