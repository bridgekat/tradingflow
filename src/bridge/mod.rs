//! PyO3 bridge — exposes the Rust runtime to Python.
//!
//! [`NativeScenario`](scenario::NativeScenario) wraps the Rust
//! [`Scenario`](crate::scenario::Scenario), providing four registration
//! entry points:
//!
//! * `add_native_source` — register a Rust-implemented source by kind string.
//! * `add_native_operator` — register a Rust-implemented operator by kind
//!   string + dtype + params.
//! * `add_py_source` — register a Python source whose `init()` returns
//!   async iterators.  Driver tasks on the tokio runtime iterate them and
//!   feed events into bounded channels consumed by the POCQ event loop.
//! * `add_py_operator` — register a Python-implemented operator whose
//!   `compute()` is called via GIL during flush.
//!
//! # Public types
//!
//! - [`NativeScenario`](scenario::NativeScenario) — the main pyclass.
//! - [`NativeArrayView`] / [`NativeSeriesView`] — read-only Python views into
//!   `Array<T>` and `Series<T>` node values, backed by raw pointers.
//! - [`NativeNotify`] — Python-visible wrapper around the Rust
//!   [`Notify`](crate::data::Notify) context, exposing which inputs
//!   produced new output to Python operators during flush.
//! - [`DoneCallback`](source::DoneCallback) — pyclass used internally to
//!   bridge `concurrent.futures.Future` completion to tokio oneshot channels.
//!
//! # Sub-modules
//!
//! - [`dispatch`] — dtype normalisation and monomorphised dispatch helpers.
//! - [`operator`] — Python operator machinery (`make_py_operator`,
//!   `resolve_type_id`).
//! - [`operators`] — native operator dispatch (`dispatch_native_operator`).
//! - [`scenario`] — [`NativeScenario`](scenario::NativeScenario) pyclass.
//! - [`source`] — Python source machinery (`register_py_source`,
//!   `DoneCallback`).
//! - [`sources`] — native source dispatch (`dispatch_native_source`).
//! - [`views`] — [`NativeArrayView`] and [`NativeSeriesView`] pyclasses.

mod dispatch;
mod operator;
mod operators;
mod scenario;
mod source;
mod sources;
mod views;

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

pub use views::{NativeArrayView, NativeNotify, NativeSeriesView};

use scenario::NativeScenario;

// ---------------------------------------------------------------------------
// Error slot
// ---------------------------------------------------------------------------

type ErrorSlot = Arc<Mutex<Option<PyErr>>>;

fn set_error(slot: &ErrorSlot, err: PyErr) {
    let mut guard = slot.lock().unwrap();
    if guard.is_none() {
        *guard = Some(err);
    }
}

fn set_error_msg(slot: &ErrorSlot, msg: String) {
    set_error(slot, pyo3::exceptions::PyRuntimeError::new_err(msg));
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<NativeScenario>()?;
    m.add_class::<NativeArrayView>()?;
    m.add_class::<NativeSeriesView>()?;
    m.add_class::<NativeNotify>()?;
    m.add_class::<source::DoneCallback>()?;
    m.add_function(pyo3::wrap_pyfunction!(py_utc_to_tai, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_tai_to_utc, m)?)?;
    Ok(())
}

/// Convert a UTC timestamp in nanoseconds to a TAI timestamp in nanoseconds.
#[pyo3::pyfunction(name = "utc_to_tai")]
fn py_utc_to_tai(utc_ns: i64) -> i64 {
    crate::data::utc_to_tai(utc_ns)
}

/// Convert a TAI timestamp in nanoseconds to a UTC timestamp in nanoseconds.
#[pyo3::pyfunction(name = "tai_to_utc")]
fn py_tai_to_utc(tai_ns: i64) -> i64 {
    crate::data::tai_to_utc(tai_ns)
}
