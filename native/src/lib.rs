//! `tradingflow-native` — Rust core for TradingFlow.
//!
//! This crate provides the performance-critical data structures and runtime
//! for the TradingFlow event-driven trading framework:
//!
//! * [`Series`] — append-only time series with manual memory management.
//! * [`Operator`] — trait for pure compute functions over series.
//! * [`Scenario`] — DAG runtime with type-erased operator dispatch.
//!
//! The crate is compiled as a PyO3 `cdylib` and exposes benchmark entry
//! points to Python.  The core types are Rust-only (no Python wrapper yet).

pub mod series;
pub mod operator;
pub mod operators;
pub mod scenario;
mod bench;

use pyo3::prelude::*;

#[pymodule]
fn tradingflow_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bench::register(m)?;
    Ok(())
}
