//! Native operator dispatch for the Python bridge.
//!
//! [`dispatch_native_operator`] maps a `(kind, dtype)` pair to a
//! monomorphised `Scenario::register_operator_from_indices` call.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::operators;
use crate::scenario::Scenario;

use super::dispatch::{dispatch_dtype, normalise_dtype};

/// Register a Rust-native operator by `(kind, dtype)` and return the output
/// node index.
///
/// If `clock` is `Some(idx)`, only the clock node triggers.
/// If `clock` is `None`, all inputs are trigger edges.
pub fn dispatch_native_operator(
    sc: &mut Scenario,
    kind: &str,
    dtype: &str,
    input_indices: &[usize],
    clock: Option<usize>,
    params: &Bound<'_, PyDict>,
) -> PyResult<usize> {
    let dtype = normalise_dtype(dtype);

    match kind {
        // -- Binary element-wise operators ----------------------------------
        "add" => {
            macro_rules! go { ($T:ty) => {
                sc.register_operator_from_indices(operators::add::<$T>(), input_indices, clock)
            }; }
            Ok(dispatch_dtype!(dtype, go))
        }
        "subtract" => {
            macro_rules! go { ($T:ty) => {
                sc.register_operator_from_indices(operators::subtract::<$T>(), input_indices, clock)
            }; }
            Ok(dispatch_dtype!(dtype, go))
        }
        "multiply" => {
            macro_rules! go { ($T:ty) => {
                sc.register_operator_from_indices(operators::multiply::<$T>(), input_indices, clock)
            }; }
            Ok(dispatch_dtype!(dtype, go))
        }
        "divide" => {
            macro_rules! go { ($T:ty) => {
                sc.register_operator_from_indices(operators::divide::<$T>(), input_indices, clock)
            }; }
            Ok(dispatch_dtype!(dtype, go))
        }

        // -- Unary element-wise operators -----------------------------------
        "negate" => {
            // negate requires T: Neg — not available for unsigned types.
            match dtype {
                "float64" => Ok(sc.register_operator_from_indices(operators::negate::<f64>(), input_indices, clock)),
                "float32" => Ok(sc.register_operator_from_indices(operators::negate::<f32>(), input_indices, clock)),
                "int64" => Ok(sc.register_operator_from_indices(operators::negate::<i64>(), input_indices, clock)),
                "int32" => Ok(sc.register_operator_from_indices(operators::negate::<i32>(), input_indices, clock)),
                other => Err(PyTypeError::new_err(format!(
                    "negate does not support dtype '{other}'"
                ))),
            }
        }

        // -- Parameterised operators ----------------------------------------
        "select" => {
            let indices: Vec<usize> = params
                .get_item("indices")?
                .ok_or_else(|| PyTypeError::new_err("select requires 'indices' param"))?
                .extract()?;
            macro_rules! go { ($T:ty) => {
                sc.register_operator_from_indices(
                    operators::Select::<$T>::flat(indices.clone()),
                    input_indices, clock,
                )
            }; }
            Ok(dispatch_dtype!(dtype, go))
        }

        // -- Variadic (homogeneous) operators -------------------------------
        "concat" => {
            let axis: usize = params
                .get_item("axis")?
                .ok_or_else(|| PyTypeError::new_err("concat requires 'axis' param"))?
                .extract()?;
            macro_rules! go { ($T:ty) => {
                sc.register_operator_from_indices(
                    operators::Concat::<$T>::new(axis),
                    input_indices, clock,
                )
            }; }
            Ok(dispatch_dtype!(dtype, go))
        }
        "stack" => {
            let axis: usize = params
                .get_item("axis")?
                .ok_or_else(|| PyTypeError::new_err("stack requires 'axis' param"))?
                .extract()?;
            macro_rules! go { ($T:ty) => {
                sc.register_operator_from_indices(
                    operators::Stack::<$T>::new(axis),
                    input_indices, clock,
                )
            }; }
            Ok(dispatch_dtype!(dtype, go))
        }

        // -- Record (Array → Series) ----------------------------------------
        "record" => {
            macro_rules! go { ($T:ty) => {{
                use crate::operators::Record;
                sc.register_operator_from_indices(Record::<$T>::new(), input_indices, clock)
            }}; }
            Ok(dispatch_dtype!(dtype, go))
        }

        other => Err(PyTypeError::new_err(format!(
            "unknown native operator kind: {other}"
        ))),
    }
}
