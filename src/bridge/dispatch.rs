//! Dtype dispatch helpers.
//!
//! Provides normalisation of numpy dtype strings and a dispatch macro for
//! calling monomorphised code based on a runtime dtype string.

use pyo3::PyResult;
use pyo3::exceptions::PyTypeError;

/// Normalize a numpy dtype string to a canonical form.
pub fn normalize_dtype(dtype: &str) -> &str {
    match dtype {
        "bool" | "|b1" => "bool",
        "int8" | "|i1" => "int8",
        "int16" | "<i2" => "int16",
        "int32" | "<i4" => "int32",
        "int64" | "<i8" => "int64",
        "uint8" | "|u1" => "uint8",
        "uint16" | "<u2" => "uint16",
        "uint32" | "<u4" => "uint32",
        "uint64" | "<u8" => "uint64",
        "float32" | "<f4" => "float32",
        "float64" | "<f8" => "float64",
        other => other,
    }
}

/// Number of bytes per scalar element for a given dtype.
pub fn dtype_element_bytes(dtype: &str) -> PyResult<usize> {
    match normalize_dtype(dtype) {
        "bool" | "int8" | "uint8" => Ok(1),
        "int16" | "uint16" => Ok(2),
        "int32" | "uint32" | "float32" => Ok(4),
        "int64" | "uint64" | "float64" => Ok(8),
        other => Err(PyTypeError::new_err(format!("unsupported dtype: {other}"))),
    }
}

/// Dispatch on a normalized dtype string, calling a macro-rule with a concrete
/// scalar type.
///
/// Usage: `dispatch_dtype!(dtype_str, my_macro)` where `my_macro` is defined as
/// `macro_rules! my_macro { ($T:ty) => { ... } }`.
///
/// The macro arm is invoked with the concrete type (e.g. `f64`, `i32`).
/// The enclosing function must return `PyResult<_>` (the unsupported-dtype
/// arm uses `return Err(...)`).
///
/// An optional third argument selects a type subset:
///
/// * *(omitted)* — all supported dtypes.
/// * `numeric` — integer and floating-point types (excludes `bool`).
/// * `signed` — signed integers and floating-point types.
/// * `float` — floating-point types only (`f32`, `f64`).
macro_rules! dispatch_dtype {
    // Internal: shared match body.
    (@match $dtype:expr, $action:ident, $label:literal,
        $($name:literal => $ty:ty),+ $(,)?) => {
        match crate::bridge::dispatch::normalize_dtype($dtype) {
            $($name => $action!($ty),)+
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    format!(concat!("unsupported dtype", $label, ": {}"), other)
                ))
            }
        }
    };
    // All supported dtypes (default).
    ($dtype:expr, $action:ident) => {
        dispatch_dtype!(@match $dtype, $action, "",
            "bool" => bool,
            "int8" => i8, "int16" => i16, "int32" => i32, "int64" => i64,
            "uint8" => u8, "uint16" => u16, "uint32" => u32, "uint64" => u64,
            "float32" => f32, "float64" => f64,
        )
    };
    // Numeric types only (Add + Sub + Mul + Div).
    ($dtype:expr, $action:ident, numeric) => {
        dispatch_dtype!(@match $dtype, $action, " for numeric operation",
            "int8" => i8, "int16" => i16, "int32" => i32, "int64" => i64,
            "uint8" => u8, "uint16" => u16, "uint32" => u32, "uint64" => u64,
            "float32" => f32, "float64" => f64,
        )
    };
    // Signed numeric types only (Neg).
    ($dtype:expr, $action:ident, signed) => {
        dispatch_dtype!(@match $dtype, $action, " for signed operation",
            "int8" => i8, "int16" => i16, "int32" => i32, "int64" => i64,
            "float32" => f32, "float64" => f64,
        )
    };
    // Floating-point types only (num_traits::Float).
    ($dtype:expr, $action:ident, float) => {
        dispatch_dtype!(@match $dtype, $action, " for floating-point operation",
            "float32" => f32, "float64" => f64,
        )
    };
}

pub(crate) use dispatch_dtype;
