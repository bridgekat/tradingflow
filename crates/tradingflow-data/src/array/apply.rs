//! Helper functions for element-wise [`ArrayView`] operations, with built-in
//! fast paths for row-major contiguous layouts.

use super::{Array, ArrayView};
use crate::{Layout, Scalar};

/// Applies a unary function on elements of `a`, writing to `out`.
/// The arrays `out` and `a` are assumed to have the same size.
pub fn apply_unary<T: Scalar, U: Scalar, const N: usize>(
    out: &mut Array<U, N>,
    a: ArrayView<T, N>,
    f: impl Fn(T) -> U,
) {
    let o = out.data_mut();
    if let Some(sa) = a.as_slice() {
        for (dst, v) in o.iter_mut().zip(sa) {
            *dst = f(v.clone());
        }
    } else {
        for (dst, i) in o.iter_mut().zip(a.layout().iter()) {
            *dst = f(a.data()[i].clone());
        }
    }
}

/// Applies a binary function on elements of `a` and `b`, writing to `out`.
/// The arrays `out`, `a`, `b` are assumed to have the same size.
pub fn apply_binary<T: Scalar, U: Scalar, const N: usize>(
    out: &mut Array<U, N>,
    a: ArrayView<T, N>,
    b: ArrayView<T, N>,
    f: impl Fn(T, T) -> U,
) {
    let o = out.data_mut();
    if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        for (dst, (va, vb)) in o.iter_mut().zip(sa.iter().zip(sb)) {
            *dst = f(va.clone(), vb.clone());
        }
    } else {
        let (al, bl) = (a.layout(), b.layout());
        for (dst, (oa, ob)) in o.iter_mut().zip(al.iter().zip(bl.iter())) {
            *dst = f(a.data()[oa].clone(), b.data()[ob].clone());
        }
    }
}
