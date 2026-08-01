use numpy::Element;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{from_numpy, from_numpy_into, to_numpy};
use crate::data::{Array, ArrayView, Scalar};
use crate::graph::Interface;
use crate::ports::{ArrayPort, ArrayPorts, SignalPort, SignalPorts};

/// An [`Interface`] whose payloads can cross the FFI boundary by copy.
///
/// An implementation packs both directions of the conversion for one
/// [`Interface`] shape:
///
/// - **Into Python**: [`to_python`](Self::to_python) walks the values tree
///   and appends **one owned NumPy array per leaf, in tree order**, so the
///   Python side receives a single flat `inputs` tuple whose positions mirror
///   the wiring exactly (signals and values share one numbering).
/// - **Out of Python**: [`from_python`](Self::from_python) copies a Python
///   return value into Rust-owned [`Buffers`](Self::Buffers) and borrows the
///   result back as the interface's values. A leaf takes its value directly;
///   a tuple branch takes a sequence with one entry per field. Buffers are
///   allocated on first use, taking their extents from the first returned
///   value; later writes negotiate by element count, as [`from_numpy`] does.
///
/// Only **arrays** cross the boundary for now — [`ArrayPort`] /
/// [`SignalPort`] leaves, their variadic [`ArrayPorts`] / [`SignalPorts`]
/// groups, and tuples thereof. Series stay on the Rust side (window them into
/// arrays, or reduce them with native operators, before wiring into a Python
/// segment).
pub trait PyInterface: Interface {
    /// Rust-owned buffers backing one node's outputs of this shape.
    /// [`Default`] is the unallocated state; the first
    /// [`from_python`](Self::from_python) allocates.
    type Buffers: Default + Send + 'static;

    /// Appends one owned NumPy array per leaf to `args`, in tree order.
    fn to_python<'py>(
        py: Python<'py>,
        values: Self::Values<'_>,
        args: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()>;

    /// Copies a Python return value into `buffers` — allocating them on first
    /// use — and borrows the result back as interface values.
    fn from_python<'a>(
        value: &Bound<'_, PyAny>,
        buffers: &'a mut Self::Buffers,
    ) -> PyResult<Self::Values<'a>>;
}

// ===========================================================================
// Single-port leaves
// ===========================================================================

/// The shared leaf write: allocate on first use, negotiate by count after.
fn leaf_from_python<'a, T: Scalar + Element, const N: usize>(
    value: &Bound<'_, PyAny>,
    buffers: &'a mut Option<Array<T, N>>,
) -> PyResult<ArrayView<'a, T, N>> {
    match buffers {
        Some(array) => from_numpy_into(value, array)?,
        None => *buffers = Some(from_numpy(value)?),
    }
    Ok(buffers.as_ref().expect("just filled").view())
}

impl<T: Scalar + Element, const N: usize> PyInterface for ArrayPort<T, N> {
    type Buffers = Option<Array<T, N>>;

    fn to_python<'py>(
        py: Python<'py>,
        values: Self::Values<'_>,
        args: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        args.push(to_numpy(py, values).into_any());
        Ok(())
    }

    fn from_python<'a>(
        value: &Bound<'_, PyAny>,
        buffers: &'a mut Self::Buffers,
    ) -> PyResult<Self::Values<'a>> {
        leaf_from_python(value, buffers)
    }
}

impl<const N: usize> PyInterface for SignalPort<N> {
    type Buffers = Option<Array<bool, N>>;

    fn to_python<'py>(
        py: Python<'py>,
        values: Self::Values<'_>,
        args: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        args.push(to_numpy(py, values).into_any());
        Ok(())
    }

    fn from_python<'a>(
        value: &Bound<'_, PyAny>,
        buffers: &'a mut Self::Buffers,
    ) -> PyResult<Self::Values<'a>> {
        leaf_from_python(value, buffers)
    }
}

// ===========================================================================
// Variadic groups
// ===========================================================================

/// Buffers for a variadic group: the arrays, plus stable storage for the
/// views the group's `Values<'a> = &'a [ArrayView<'a, T, N>]` slice needs to
/// point at. The views are stored lifetime-erased, exactly like the engine's
/// own `Ports` scratch (see `Interface for Ports<V>`): erasure is
/// storage-only, and the views are rebuilt from `arrays` on every write —
/// after clearing the stale ones, so no erased view is ever read after the
/// array it points into has been touched.
pub struct GroupBuffers<T: Scalar, const N: usize> {
    /// `None` until the first write fixes the group's length.
    arrays: Option<Vec<Array<T, N>>>,
    views: Vec<ArrayView<'static, T, N>>,
}

impl<T: Scalar, const N: usize> Default for GroupBuffers<T, N> {
    fn default() -> Self {
        Self {
            arrays: None,
            views: Vec::new(),
        }
    }
}

/// The shared group write. The group's length is fixed by the first write —
/// the engine requires variadic port groups to keep a static length anyway.
fn group_from_python<'a, T: Scalar + Element, const N: usize>(
    value: &Bound<'_, PyAny>,
    buffers: &'a mut GroupBuffers<T, N>,
) -> PyResult<&'a [ArrayView<'a, T, N>]> {
    let GroupBuffers { arrays, views } = buffers;
    // Drop stale views before writing the arrays they point into.
    views.clear();

    let n = value.len()?;
    match arrays {
        Some(arrays) => {
            if arrays.len() != n {
                return Err(PyValueError::new_err(format!(
                    "group: expected {} values (fixed by the first return), got {n}",
                    arrays.len(),
                )));
            }
            for (i, array) in arrays.iter_mut().enumerate() {
                from_numpy_into(&value.get_item(i)?, array)?;
            }
        }
        None => {
            let mut fresh = Vec::with_capacity(n);
            for i in 0..n {
                fresh.push(from_numpy(&value.get_item(i)?)?);
            }
            *arrays = Some(fresh);
        }
    }

    for array in arrays.as_ref().expect("just filled") {
        // SAFETY: storage-only lifetime erasure. The view points into the
        // array's boxed data, which stays put for as long as `buffers` lives
        // (moving the `Vec<Array>` moves array headers, not their heap data),
        // and every erased view is cleared above before that data is next
        // written.
        views.push(unsafe {
            std::mem::transmute::<ArrayView<'_, T, N>, ArrayView<'static, T, N>>(array.view())
        });
    }
    // SAFETY: shortening the erased views back to the borrow of `buffers`
    // they are in fact valid for; `ArrayView` is covariant in its lifetime.
    Ok(unsafe {
        std::mem::transmute::<&[ArrayView<'static, T, N>], &'a [ArrayView<'a, T, N>]>(
            views.as_slice(),
        )
    })
}

impl<T: Scalar + Element, const N: usize> PyInterface for ArrayPorts<T, N> {
    type Buffers = GroupBuffers<T, N>;

    fn to_python<'py>(
        py: Python<'py>,
        values: Self::Values<'_>,
        args: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        for &value in values {
            args.push(to_numpy(py, value).into_any());
        }
        Ok(())
    }

    fn from_python<'a>(
        value: &Bound<'_, PyAny>,
        buffers: &'a mut Self::Buffers,
    ) -> PyResult<Self::Values<'a>> {
        group_from_python(value, buffers)
    }
}

impl<const N: usize> PyInterface for SignalPorts<N> {
    type Buffers = GroupBuffers<bool, N>;

    fn to_python<'py>(
        py: Python<'py>,
        values: Self::Values<'_>,
        args: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        for &value in values {
            args.push(to_numpy(py, value).into_any());
        }
        Ok(())
    }

    fn from_python<'a>(
        value: &Bound<'_, PyAny>,
        buffers: &'a mut Self::Buffers,
    ) -> PyResult<Self::Values<'a>> {
        group_from_python(value, buffers)
    }
}

// ===========================================================================
// Branches
// ===========================================================================

impl PyInterface for () {
    type Buffers = ();

    fn to_python<'py>(
        _py: Python<'py>,
        _values: Self::Values<'_>,
        _args: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        Ok(())
    }

    fn from_python<'a>(
        value: &Bound<'_, PyAny>,
        _buffers: &'a mut Self::Buffers,
    ) -> PyResult<Self::Values<'a>> {
        if value.is_none() {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "unit interface: expected None as the return value",
            ))
        }
    }
}

macro_rules! impl_py_interface_for_tuple {
    ($len:literal; $($idx:tt: $T:ident),+) => {
        impl<$($T: PyInterface,)+> PyInterface for ($($T,)+) {
            type Buffers = ($($T::Buffers,)+);

            fn to_python<'py>(
                py: Python<'py>,
                values: Self::Values<'_>,
                args: &mut Vec<Bound<'py, PyAny>>,
            ) -> PyResult<()> {
                $( $T::to_python(py, values.$idx, args)?; )+
                Ok(())
            }

            fn from_python<'a>(
                value: &Bound<'_, PyAny>,
                buffers: &'a mut Self::Buffers,
            ) -> PyResult<Self::Values<'a>> {
                let n = value.len()?;
                if n != $len {
                    return Err(PyValueError::new_err(format!(
                        "tuple interface: expected a sequence of {} values, got {n}",
                        $len,
                    )));
                }
                Ok(( $( $T::from_python(&value.get_item($idx)?, &mut buffers.$idx)?, )+ ))
            }
        }
    };
}

impl_py_interface_for_tuple!(1; 0: A);
impl_py_interface_for_tuple!(2; 0: A, 1: B);
impl_py_interface_for_tuple!(3; 0: A, 1: B, 2: C);
impl_py_interface_for_tuple!(4; 0: A, 1: B, 2: C, 3: D);
impl_py_interface_for_tuple!(5; 0: A, 1: B, 2: C, 3: D, 4: E);
impl_py_interface_for_tuple!(6; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_py_interface_for_tuple!(7; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_py_interface_for_tuple!(8; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_py_interface_for_tuple!(9; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_py_interface_for_tuple!(10; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_py_interface_for_tuple!(11; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_py_interface_for_tuple!(12; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
