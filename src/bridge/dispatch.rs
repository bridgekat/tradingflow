//! Dtype dispatch helpers.
//!
//! Dispatch macro for calling monomorphised code based on a runtime dtype
//! string.  Python is responsible for sending canonical dtype names (i.e.
//! `numpy.dtype.name` rather than `numpy.dtype.str`); this module performs
//! no aliasing.

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
        $($name:literal => $ty:ty),+ $(,)?) => {{
        // Accept either `&str` or anything that derefs to `str` (e.g. `&String`).
        let __dtype: &str = ::std::convert::AsRef::<str>::as_ref($dtype);
        match __dtype {
            $($name => $action!($ty),)+
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    format!(concat!("unsupported dtype", $label, ": {}"), other)
                ))
            }
        }
    }};
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

use std::any::TypeId;

use pyo3::exceptions::PyValueError;
use pyo3::types::PyAnyMethods;

use crate::{Array, Series};

use super::views::NativeNodeKind;

/// Resolve a `(kind, dtype)` pair to a Rust [`TypeId`].
///
/// The dtype string is ignored for [`NativeNodeKind::Unit`] — unit nodes
/// always map to `TypeId::of::<()>()`.
pub fn resolve_type_id(kind: NativeNodeKind, dtype: &str) -> pyo3::PyResult<TypeId> {
    if kind == NativeNodeKind::Unit {
        return Ok(TypeId::of::<()>());
    }
    macro_rules! resolve {
        ($T:ty) => {
            match kind {
                NativeNodeKind::Array => Ok(TypeId::of::<Array<$T>>()),
                NativeNodeKind::Series => Ok(TypeId::of::<Series<$T>>()),
                NativeNodeKind::Unit => unreachable!(),
            }
        };
    }
    dispatch_dtype!(dtype, resolve)
}

/// Validated C-contiguous numpy array metadata, read from
/// `__array_interface__`.
///
/// Constructed via [`from_bound`](Self::from_bound), which checks that
/// `strides` is `None` (C-contiguous) and extracts the data pointer and
/// shape.  The caller (Python side) is responsible for ensuring contiguous
/// layout before passing the array; this type only validates.
pub struct ContiguousArrayInfo {
    /// Raw data pointer (`__array_interface__["data"][0]`).
    pub ptr: usize,
    /// Array shape (`__array_interface__["shape"]`).
    pub shape: Vec<usize>,
}

impl<'py> TryFrom<&pyo3::Bound<'py, pyo3::types::PyAny>> for ContiguousArrayInfo {
    type Error = pyo3::PyErr;

    /// Parse and validate `__array_interface__` from a Python object.
    ///
    /// Fails if `strides` is not `None`.
    fn try_from(array: &pyo3::Bound<'py, pyo3::types::PyAny>) -> pyo3::PyResult<Self> {
        let interface = array.getattr("__array_interface__")?;

        let strides = interface.get_item("strides")?;
        if !strides.is_none() {
            return Err(PyValueError::new_err(
                "numpy array is not C-contiguous (strides is not None)",
            ));
        }

        let shape: Vec<usize> = interface.get_item("shape")?.extract()?;
        let ptr: usize = interface.get_item("data")?.get_item(0)?.extract()?;

        Ok(Self { ptr, shape })
    }
}

impl ContiguousArrayInfo {
    /// Total number of elements (product of shape dimensions).
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Copy the array data into a `Vec<T>`.
    ///
    /// # Safety
    ///
    /// `T` must be a numeric type where every bit pattern of size
    /// `size_of::<T>()` is a valid value, and the array's dtype must
    /// match `T` in element size.
    pub unsafe fn to_vec<T>(&self) -> Vec<T> {
        let n = self.len();
        let mut result = Vec::with_capacity(n);
        unsafe {
            self.clone_to_slice(std::slice::from_raw_parts_mut(result.as_mut_ptr(), n));
            result.set_len(n);
        }
        result
    }

    /// Copy the array data into an existing slice.
    ///
    /// # Safety
    ///
    /// `T` must be a numeric type where every bit pattern is valid.
    /// `dst` must have exactly `self.len()` elements.
    pub unsafe fn clone_to_slice<T>(&self, dst: &mut [T]) {
        let nbytes = std::mem::size_of_val(dst);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr as *const u8,
                dst.as_mut_ptr() as *mut u8,
                nbytes,
            );
        }
    }
}
