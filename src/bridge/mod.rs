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
//!   iterators.
//! * `add_py_operator` — register a Python-implemented operator whose
//!   `compute()` is called via GIL during flush.
//!
//! Both native and Python operators are registered through
//! [`Scenario::add_erased_operator`], the unified type-erased entry point.

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
