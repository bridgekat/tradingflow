//! `tradingflow` — Rust core for TradingFlow.
//!
//! This crate provides the performance-critical data structures and runtime
//! for the TradingFlow event-driven trading framework:
//!
//! * [`Array`] — dense, dynamically-shaped array.
//! * [`Series`] — append-only time series.
//! * [`Operator`] — trait for compute functions.
//! * [`Source`] — trait for data sources.
//! * [`Scenario`] — DAG runtime with type-erased dispatch.
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
pub use operator::Operator;
pub use scenario::{ErasedOperator, Scenario};
pub use series::Series;
pub use source::Source;
pub use types::InputTypes;
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
