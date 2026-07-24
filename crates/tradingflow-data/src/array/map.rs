//! Helper functions for element-wise [`ArrayView`] operations, with built-in
//! fast paths for row-major contiguous layouts.

use super::{Array, ArrayView};
use crate::layout::Strided;
use crate::{Layout, Scalar};

/// Applies a unary function on elements of `a`, returning a new array.
pub fn map<A: Scalar, T: Scalar, const N: usize>(
    a: ArrayView<A, N>,
    f: impl FnMut(&A) -> T,
) -> Array<T, N> {
    let mut out = Array::zeros(a.extents());
    map_into(out.data_mut(), a, f);
    out
}

/// Applies a unary function on elements of `a`, writing into the row-major
/// buffer `out`. The array view `a` is assumed to have extents matching `out`.
pub fn map_into<A: Scalar, T: Scalar, const N: usize>(
    out: &mut [T],
    a: ArrayView<A, N>,
    f: impl FnMut(&A) -> T,
) {
    let mut f = f;
    for_each(a, |i, v| out[i] = f(v));
}

/// Applies a binary function on elements of `a` and `b`, broadcasting each
/// extent-1 axis, and returning a new array with the broadcast extents.
///
/// # Panics
///
/// Panics if `a` and `b` do not have broadcast-compatible extents.
pub fn binary_map<A: Scalar, B: Scalar, T: Scalar, const N: usize>(
    a: ArrayView<A, N>,
    b: ArrayView<B, N>,
    f: impl FnMut(&A, &B) -> T,
) -> Array<T, N> {
    let extents = broadcast_extents(a.extents(), b.extents());
    let mut out = Array::zeros(extents);
    binary_map_into(out.data_mut(), a, b, f);
    out
}

/// Iterates over the scalars of `a` and `b` in row-major order, broadcasting
/// each extent-1 axis, and writing into the row-major buffer `out`.
/// The array views `a` and `b` are assumed to have broadcast-compatible
/// extents, with the broadcast extents matching `out`.
pub fn binary_map_into<A: Scalar, B: Scalar, T: Scalar, const N: usize>(
    out: &mut [T],
    a: ArrayView<A, N>,
    b: ArrayView<B, N>,
    f: impl FnMut(&A, &B) -> T,
) {
    let mut f = f;
    binary_for_each(a, b, |i, x, y| out[i] = f(x, y));
}

/// Applies a ternary function on elements of `a`, `b` and `c`, broadcasting
/// each extent-1 axis, and returning a new array with the broadcast extents.
///
/// # Panics
///
/// Panics if `a`, `b` and `c` do not have broadcast-compatible extents.
pub fn ternary_map<A: Scalar, B: Scalar, C: Scalar, T: Scalar, const N: usize>(
    a: ArrayView<A, N>,
    b: ArrayView<B, N>,
    c: ArrayView<C, N>,
    f: impl FnMut(&A, &B, &C) -> T,
) -> Array<T, N> {
    let extents = broadcast_extents(a.extents(), broadcast_extents(b.extents(), c.extents()));
    let mut out = Array::zeros(extents);
    ternary_map_into(out.data_mut(), a, b, c, f);
    out
}

/// Iterates over the scalars of `a`, `b`, `c` in row-major order, broadcasting
/// each extent-1 axis, and writing into the row-major buffer `out`.
/// The array views `a`, `b`, `c` are assumed to have broadcast-compatible
/// extents, with the broadcast extents matching `out`.
pub fn ternary_map_into<A: Scalar, B: Scalar, C: Scalar, T: Scalar, const N: usize>(
    out: &mut [T],
    a: ArrayView<A, N>,
    b: ArrayView<B, N>,
    c: ArrayView<C, N>,
    f: impl FnMut(&A, &B, &C) -> T,
) {
    let mut f = f;
    ternary_for_each(a, b, c, |i, x, y, z| out[i] = f(x, y, z));
}

/// Iterates over the scalars of `a` in row-major order.
pub fn for_each<A: Scalar, const N: usize>(a: ArrayView<A, N>, f: impl FnMut(usize, &A)) {
    let mut f = f;
    match a.as_slice() {
        Some(a) => {
            for (i, x) in a.iter().enumerate() {
                f(i, x);
            }
        }
        None => {
            for (i, x) in a.iter().enumerate() {
                f(i, x);
            }
        }
    }
}

/// Iterates over the scalars of `a` and `b` in row-major order, broadcasting
/// each extent-1 axis.
pub fn binary_for_each<A: Scalar, B: Scalar, const N: usize>(
    a: ArrayView<A, N>,
    b: ArrayView<B, N>,
    f: impl FnMut(usize, &A, &B),
) {
    let extents = broadcast_extents(a.extents(), b.extents());
    let a = broadcast_to(a, extents);
    let b = broadcast_to(b, extents);
    let mut f = f;
    match (a.as_slice(), b.as_slice()) {
        (Some(a), Some(b)) => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                f(i, x, y);
            }
        }
        (Some(a), None) => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                f(i, x, y);
            }
        }
        (None, Some(b)) => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                f(i, x, y);
            }
        }
        (None, None) => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                f(i, x, y);
            }
        }
    }
}

/// Iterates over the scalars of `a`, `b`, `c` in row-major order, broadcasting
/// each extent-1 axis.
pub fn ternary_for_each<A: Scalar, B: Scalar, C: Scalar, const N: usize>(
    a: ArrayView<A, N>,
    b: ArrayView<B, N>,
    c: ArrayView<C, N>,
    f: impl FnMut(usize, &A, &B, &C),
) {
    let extents = broadcast_extents(a.extents(), broadcast_extents(b.extents(), c.extents()));
    let a = broadcast_to(a, extents);
    let b = broadcast_to(b, extents);
    let c = broadcast_to(c, extents);
    let mut f = f;
    for (i, (x, (y, z))) in a.iter().zip(b.iter().zip(c.iter())).enumerate() {
        f(i, x, y, z);
    }
}

/// The broadcast extents of `a` and `b`: per axis, the two extents must be
/// equal, or one of them must be `1` and stretches to the other.
///
/// # Panics
///
/// Panics if the extents differ on an axis where neither is `1`.
fn broadcast_extents<const N: usize>(a: [usize; N], b: [usize; N]) -> [usize; N] {
    std::array::from_fn(|d| match (a[d], b[d]) {
        (x, y) if x == y => x,
        (1, y) => y,
        (x, 1) => x,
        _ => panic!("map_broadcast: extents {a:?} and {b:?} are not broadcast-compatible"),
    })
}

/// Stretches the extent-1 axes of `v` to the target `extents` (as produced by
/// [`broadcast_extents`]) by giving them stride 0, so every index along such
/// an axis reads the one element. A view already at `extents` is returned
/// unchanged, preserving contiguity.
fn broadcast_to<'a, T: Scalar, const N: usize>(
    v: ArrayView<'a, T, N>,
    extents: [usize; N],
) -> ArrayView<'a, T, N> {
    let (ve, vs) = (v.extents(), v.layout().strides());
    if ve == extents {
        return v;
    }
    let strides = std::array::from_fn(|d| if ve[d] == extents[d] { vs[d] } else { 0 });
    // SAFETY: stride-0 axes contribute nothing to the span, so the stretched
    // layout spans no more data than the original.
    unsafe { ArrayView::from_parts_unchecked(Strided::new(extents, strides), v.data()) }
}
