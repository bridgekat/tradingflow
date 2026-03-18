//! `tradingflow` — Rust core for TradingFlow.
//!
//! This crate provides the performance-critical data structures and runtime
//! for the TradingFlow event-driven trading framework:
//!
//! * [`Observable`] — single-value buffer (latest value of a graph node).
//! * [`Series`] — append-only time series with length-doubling growth.
//! * [`Operator`] — trait for pure compute functions.
//! * [`Scenario`] — DAG runtime with type-erased operator dispatch.
//!
//! When compiled with the `python` feature, the crate also produces a PyO3
//! `cdylib` exposing the runtime to Python.

pub mod input;
pub mod observable;
pub mod operator;
pub mod operators;
pub mod scenario;
pub mod series;
pub mod source;

pub use observable::Observable;
pub use operator::Operator;
pub use scenario::Scenario;
pub use series::Series;

#[cfg(feature = "python")]
mod bench;
#[cfg(feature = "python")]
mod bridge;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bench::register(m)?;
    bridge::register(m)?;
    Ok(())
}
