//! PyO3 bridge — exposes the Rust runtime to Python.
//!
//! [`NativeScenario`](scenario::NativeScenario) wraps the Rust
//! [`Scenario`](crate::scenario::Scenario), providing four registration
//! entry points:
//!
//! * `add_native_source` — register a Rust-implemented source by kind string.
//! * `add_native_operator` — register a Rust-implemented operator by kind
//!   string + dtype + params.
//! * `add_py_source` — register a channel-based source driven by Python async
//!   iterators.  Returns a pair of [`EventSender`]s (historical and live).
//! * `add_py_operator` — register a Python-implemented operator whose
//!   `compute()` is called via GIL during flush.
//!
//! Both native and Python operators are registered through
//! [`Scenario::add_erased_operator`], the unified type-erased entry point.
//!
//! # Public types
//!
//! - [`NativeScenario`](scenario::NativeScenario) — the main pyclass.
//! - [`NativeArrayView`] / [`NativeSeriesView`] — read-only Python views into
//!   `Array<T>` and `Series<T>` node values, backed by raw pointers.
//! - [`EventSender`] — pyclass wrapping a typed channel sender for pushing
//!   events from Python into a channel-based source.
//!
//! # Sub-modules
//!
//! - [`dispatch`] — dtype normalisation and monomorphised dispatch helpers.
//! - [`operator`] — Python operator machinery (`make_py_operator`,
//!   `resolve_type_id`).
//! - [`operators`] — native operator dispatch (`dispatch_native_operator`).
//! - [`scenario`] — [`NativeScenario`](scenario::NativeScenario) pyclass.
//! - [`sources`] — source registration (`dispatch_native_source`,
//!   `register_channel_source`).
//! - [`views`] — [`NativeArrayView`] and [`NativeSeriesView`] pyclasses.

mod dispatch;
mod operator;
mod operators;
mod scenario;
mod sources;
mod views;

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

pub use sources::EventSender;
pub use views::{NativeArrayView, NativeSeriesView};

use scenario::NativeScenario;

// ---------------------------------------------------------------------------
// Error slot
// ---------------------------------------------------------------------------

type ErrorSlot = Arc<Mutex<Option<String>>>;

fn set_error(slot: &ErrorSlot, msg: String) {
    let mut guard = slot.lock().unwrap();
    if guard.is_none() {
        *guard = Some(msg);
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<NativeScenario>()?;
    m.add_class::<NativeArrayView>()?;
    m.add_class::<NativeSeriesView>()?;
    m.add_class::<EventSender>()?;
    Ok(())
}
