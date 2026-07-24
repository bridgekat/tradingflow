//! Helper functions for element-wise [`ArrayView`] operations, with built-in
//! fast paths for row-major contiguous layouts.

use super::{Array, ArrayView};
use crate::layout::Strided;
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
pub fn map_binary<S: Scalar, T: Scalar, U: Scalar, const N: usize>(
    a: ArrayView<S, N>,
    b: ArrayView<T, N>,
    f: impl Fn(S, T) -> U,
) -> Array<U, N> {
    assert_eq!(a.extents(), b.extents(), "apply_binary: extents mismatch");
    let mut out = Array::zeros(a.extents());
    map_binary_into(out.data_mut(), a, b, f);
    out
}

/// Applies a binary function on elements of `a` and `b`, writing into the
/// row-major buffer `out`. The buffer, `a` and `b` are assumed to have the
/// same extents.
pub fn map_binary_into<S: Scalar, T: Scalar, U: Scalar, const N: usize>(
    out: &mut [U],
    a: ArrayView<S, N>,
    b: ArrayView<T, N>,
    f: impl Fn(S, T) -> U,
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

/// Applies a binary function on elements of `a` and `b`, broadcasting each
/// extent-1 axis to the other operand's extent on that axis, and returning a
/// new array with the broadcast extents.
///
/// # Panics
///
/// Panics if `a` and `b` differ in extent on an axis where neither is `1`.
pub fn map_broadcast<S: Scalar, T: Scalar, U: Scalar, const N: usize>(
    a: ArrayView<S, N>,
    b: ArrayView<T, N>,
    f: impl Fn(S, T) -> U,
) -> Array<U, N> {
    let extents = broadcast_extents(a.extents(), b.extents());
    map_binary(broadcast_to(a, extents), broadcast_to(b, extents), f)
}

/// Applies a binary function on elements of `a` and `b`, broadcasting each
/// extent-1 axis to the other operand's extent on that axis, writing into the
/// row-major buffer `out`. The buffer is assumed to have the broadcast
/// extents.
pub fn map_broadcast_into<S: Scalar, T: Scalar, U: Scalar, const N: usize>(
    out: &mut [U],
    a: ArrayView<S, N>,
    b: ArrayView<T, N>,
    f: impl Fn(S, T) -> U,
) {
    let extents = broadcast_extents(a.extents(), b.extents());
    map_binary_into(out, broadcast_to(a, extents), broadcast_to(b, extents), f);
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
