//! Helper functions for element-wise [`ArrayView`] operations, with built-in
//! fast paths for row-major contiguous layouts.

use super::{Array, ArrayView};
use crate::{Layout, Scalar};

/// Applies a unary function on elements of `a`, returning a new array.
pub fn map<T: Scalar, U: Scalar, const N: usize>(
    a: ArrayView<T, N>,
    f: impl Fn(T) -> U,
) -> Array<U, N> {
    let mut out = Array::zeros(a.extents());
    map_into(out.data_mut(), a, f);
    out
}

/// Applies a unary function on elements of `a`, writing into the row-major
/// buffer `out`. The buffer and `a` are assumed to have the same extents.
pub fn map_into<T: Scalar, U: Scalar, const N: usize>(
    out: &mut [U],
    a: ArrayView<T, N>,
    f: impl Fn(T) -> U,
) {
    if let Some(sa) = a.as_slice() {
        for (dst, v) in out.iter_mut().zip(sa) {
            *dst = f(v.clone());
        }
    } else {
        for (dst, i) in out.iter_mut().zip(a.layout().iter()) {
            *dst = f(a.data()[i].clone());
        }
    }
}

/// Applies a binary function on elements of `a` and `b`, returning a new
/// array.
///
/// # Panics
///
/// Panics if `a.extents() != b.extents()`.
pub fn map_binary<T: Scalar, U: Scalar, const N: usize>(
    a: ArrayView<T, N>,
    b: ArrayView<T, N>,
    f: impl Fn(T, T) -> U,
) -> Array<U, N> {
    assert_eq!(a.extents(), b.extents(), "apply_binary: extents mismatch");
    let mut out = Array::zeros(a.extents());
    map_binary_into(out.data_mut(), a, b, f);
    out
}

/// Applies a binary function on elements of `a` and `b`, writing into the
/// row-major buffer `out`. The buffer, `a` and `b` are assumed to have the
/// same extents.
pub fn map_binary_into<T: Scalar, U: Scalar, const N: usize>(
    out: &mut [U],
    a: ArrayView<T, N>,
    b: ArrayView<T, N>,
    f: impl Fn(T, T) -> U,
) {
    if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        for (dst, (va, vb)) in out.iter_mut().zip(sa.iter().zip(sb)) {
            *dst = f(va.clone(), vb.clone());
        }
    } else {
        let (al, bl) = (a.layout(), b.layout());
        for (dst, (oa, ob)) in out.iter_mut().zip(al.iter().zip(bl.iter())) {
            *dst = f(a.data()[oa].clone(), b.data()[ob].clone());
        }
    }
}
