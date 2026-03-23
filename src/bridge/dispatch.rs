//! Dtype dispatch helpers.
//!
//! Provides normalisation of numpy dtype strings and a dispatch macro for
//! calling monomorphised code based on a runtime dtype string.

use pyo3::exceptions::PyTypeError;
use pyo3::PyResult;

/// Normalise a numpy dtype string to a canonical form.
pub fn normalise_dtype(dtype: &str) -> &str {
    match dtype {
        "float64" | "<f8" => "float64",
        "float32" | "<f4" => "float32",
        "int64" | "<i8" => "int64",
        "int32" | "<i4" => "int32",
        "uint64" | "<u8" => "uint64",
        "uint32" | "<u4" => "uint32",
        "bool" | "|b1" => "bool",
        other => other,
    }
}

/// Number of bytes per scalar element for a given dtype.
pub fn dtype_element_bytes(dtype: &str) -> PyResult<usize> {
    match dtype {
        "float64" | "<f8" => Ok(8),
        "float32" | "<f4" => Ok(4),
        "int64" | "<i8" => Ok(8),
        "int32" | "<i4" => Ok(4),
        "uint64" | "<u8" => Ok(8),
        "uint32" | "<u4" => Ok(4),
        "bool" | "|b1" => Ok(1),
        other => Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
}

/// Dispatch on a normalised dtype string, calling a macro-rule with a concrete
/// scalar type.
///
/// Usage: `dispatch_dtype!(dtype_str, my_macro)` where `my_macro` is defined as
/// `macro_rules! my_macro { ($T:ty) => { ... } }`.
///
/// The macro arm is invoked with the concrete type (e.g. `f64`, `i32`).
/// The enclosing function must return `PyResult<_>` (the unsupported-dtype
/// arm uses `return Err(...)`).
macro_rules! dispatch_dtype {
    ($dtype:expr, $macro_name:ident) => {
        match crate::bridge::dispatch::normalise_dtype($dtype) {
            "float64" => $macro_name!(f64),
            "float32" => $macro_name!(f32),
            "int64" => $macro_name!(i64),
            "int32" => $macro_name!(i32),
            "uint64" => $macro_name!(u64),
            "uint32" => $macro_name!(u32),
            "bool" => $macro_name!(u8),
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported dtype: {other}"
                )))
            }
        }
    };
}

pub(crate) use dispatch_dtype;
