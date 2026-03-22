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

/// Dispatch on a normalised dtype string, calling a typed expression.
macro_rules! dispatch_dtype {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            "float64" => { type $T = f64; $body }
            "float32" => { type $T = f32; $body }
            "int64"   => { type $T = i64; $body }
            "int32"   => { type $T = i32; $body }
            "uint64"  => { type $T = u64; $body }
            "uint32"  => { type $T = u32; $body }
            "bool"    => { type $T = u8;  $body }
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    format!("unsupported dtype: {other}"),
                ))
            }
        }
    };
}

pub(crate) use dispatch_dtype;
