//! Operator registration for the Python bridge.
//!
//! [`NativeOpHandle`] captures a fully-monomorphised operator registration
//! closure.  Factory pyfunctions (`add`, `negate`, etc.) create handles that
//! are consumed by [`NativeScenario::register_handle_operator`].

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;

use crate::operators;
use crate::scenario::{Handle, Scenario};

use super::dispatch::normalise_dtype;

// ---------------------------------------------------------------------------
// NativeOpHandle
// ---------------------------------------------------------------------------

/// Registration closure: given a Scenario and input indices, registers the
/// operator and returns the output node index.
type RegisterFn = Box<dyn FnOnce(&mut Scenario, &[usize]) -> usize + Send + Sync>;

/// Opaque handle holding a pre-constructed, type-erased Rust operator.
///
/// Created by Python-callable factory functions (`add`, `negate`, etc.) and
/// consumed by [`NativeScenario::register_handle_operator`].  The handle is
/// single-use: the inner closure is consumed on registration.
#[pyclass]
pub struct NativeOpHandle {
    register_fn: Option<RegisterFn>,
    pub(super) dtype_str: String,
}

// SAFETY: The captured operator is Send+Sync (plain data structs).
// NativeOpHandle is only mutated via &mut self in register_handle_operator.
unsafe impl Send for NativeOpHandle {}
unsafe impl Sync for NativeOpHandle {}

impl NativeOpHandle {
    /// Consume the registration closure, returning the output node index.
    pub fn take_and_register(
        &mut self,
        sc: &mut Scenario,
        input_indices: &[usize],
    ) -> PyResult<usize> {
        let f = self
            .register_fn
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("NativeOpHandle has already been consumed"))?;
        Ok(f(sc, input_indices))
    }
}

// ---------------------------------------------------------------------------
// Factory pyfunctions — binary operators
// ---------------------------------------------------------------------------

/// Generate a `#[pyfunction]` that creates a binary `NativeOpHandle`.
///
/// The closure is monomorphised per concrete scalar type, so
/// `add_operator` receives fully-concrete handle tuples.
macro_rules! def_binary_op {
    ($py_name:ident) => {
        #[pyfunction]
        pub fn $py_name(dtype: &str) -> PyResult<NativeOpHandle> {
            let d = normalise_dtype(dtype).to_string();
            match d.as_str() {
                "float64" => Ok(NativeOpHandle {
                    register_fn: Some(Box::new(|sc, inputs| {
                        let h0 = Handle::<f64>::new(inputs[0]);
                        let h1 = Handle::<f64>::new(inputs[1]);
                        sc.add_operator(operators::$py_name::<f64>(), (h0, h1))
                            .index()
                    })),
                    dtype_str: d,
                }),
                "float32" => Ok(NativeOpHandle {
                    register_fn: Some(Box::new(|sc, inputs| {
                        let h0 = Handle::<f32>::new(inputs[0]);
                        let h1 = Handle::<f32>::new(inputs[1]);
                        sc.add_operator(operators::$py_name::<f32>(), (h0, h1))
                            .index()
                    })),
                    dtype_str: d,
                }),
                other => Err(PyTypeError::new_err(format!(
                    "Rust operator '{}' does not support dtype '{other}'",
                    stringify!($py_name),
                ))),
            }
        }
    };
}

/// Generate a `#[pyfunction]` that creates a unary `NativeOpHandle`.
macro_rules! def_unary_op {
    ($py_name:ident) => {
        #[pyfunction]
        pub fn $py_name(dtype: &str) -> PyResult<NativeOpHandle> {
            let d = normalise_dtype(dtype).to_string();
            match d.as_str() {
                "float64" => Ok(NativeOpHandle {
                    register_fn: Some(Box::new(|sc, inputs| {
                        let h0 = Handle::<f64>::new(inputs[0]);
                        sc.add_operator(operators::$py_name::<f64>(), (h0,)).index()
                    })),
                    dtype_str: d,
                }),
                "float32" => Ok(NativeOpHandle {
                    register_fn: Some(Box::new(|sc, inputs| {
                        let h0 = Handle::<f32>::new(inputs[0]);
                        sc.add_operator(operators::$py_name::<f32>(), (h0,)).index()
                    })),
                    dtype_str: d,
                }),
                other => Err(PyTypeError::new_err(format!(
                    "Rust operator '{}' does not support dtype '{other}'",
                    stringify!($py_name),
                ))),
            }
        }
    };
}

def_binary_op!(add);
def_binary_op!(subtract);
def_binary_op!(multiply);
def_binary_op!(divide);
def_unary_op!(negate);

// -- Parameterised operators (hand-written) ----------------------------------

#[pyfunction]
pub fn select(dtype: &str, indices: Vec<usize>) -> PyResult<NativeOpHandle> {
    let d = normalise_dtype(dtype).to_string();
    match d.as_str() {
        "float64" => {
            let op = operators::Select::<f64>::flat(indices);
            Ok(NativeOpHandle {
                register_fn: Some(Box::new(move |sc, inputs| {
                    let h0 = Handle::<f64>::new(inputs[0]);
                    sc.add_operator(op, (h0,)).index()
                })),
                dtype_str: d,
            })
        }
        "float32" => {
            let op = operators::Select::<f32>::flat(indices);
            Ok(NativeOpHandle {
                register_fn: Some(Box::new(move |sc, inputs| {
                    let h0 = Handle::<f32>::new(inputs[0]);
                    sc.add_operator(op, (h0,)).index()
                })),
                dtype_str: d,
            })
        }
        other => Err(PyTypeError::new_err(format!(
            "Rust operator 'select' does not support dtype '{other}'"
        ))),
    }
}

#[pyfunction]
pub fn concat(dtype: &str, input_shape: Vec<usize>, axis: usize) -> PyResult<NativeOpHandle> {
    let d = normalise_dtype(dtype).to_string();
    match d.as_str() {
        "float64" => {
            let op = operators::Concat::<f64>::new(&input_shape, axis);
            Ok(NativeOpHandle {
                register_fn: Some(Box::new(move |sc, inputs| {
                    let handles: Box<[Handle<f64>]> =
                        inputs.iter().map(|&i| Handle::new(i)).collect();
                    sc.add_operator(op, handles).index()
                })),
                dtype_str: d,
            })
        }
        "float32" => {
            let op = operators::Concat::<f32>::new(&input_shape, axis);
            Ok(NativeOpHandle {
                register_fn: Some(Box::new(move |sc, inputs| {
                    let handles: Box<[Handle<f32>]> =
                        inputs.iter().map(|&i| Handle::new(i)).collect();
                    sc.add_operator(op, handles).index()
                })),
                dtype_str: d,
            })
        }
        other => Err(PyTypeError::new_err(format!(
            "Rust operator 'concat' does not support dtype '{other}'"
        ))),
    }
}

#[pyfunction]
pub fn stack(dtype: &str, input_shape: Vec<usize>, axis: usize) -> PyResult<NativeOpHandle> {
    let d = normalise_dtype(dtype).to_string();
    match d.as_str() {
        "float64" => {
            let op = operators::Stack::<f64>::new(&input_shape, axis);
            Ok(NativeOpHandle {
                register_fn: Some(Box::new(move |sc, inputs| {
                    let handles: Box<[Handle<f64>]> =
                        inputs.iter().map(|&i| Handle::new(i)).collect();
                    sc.add_operator(op, handles).index()
                })),
                dtype_str: d,
            })
        }
        "float32" => {
            let op = operators::Stack::<f32>::new(&input_shape, axis);
            Ok(NativeOpHandle {
                register_fn: Some(Box::new(move |sc, inputs| {
                    let handles: Box<[Handle<f32>]> =
                        inputs.iter().map(|&i| Handle::new(i)).collect();
                    sc.add_operator(op, handles).index()
                })),
                dtype_str: d,
            })
        }
        other => Err(PyTypeError::new_err(format!(
            "Rust operator 'stack' does not support dtype '{other}'"
        ))),
    }
}
