//! Helper functions for folding the scalars of an [`ArrayView`] into an
//! accumulator array, along either its trailing or its leading axes.

use super::{Array, ArrayView, for_each};
use crate::Scalar;

/// Reduces the trailing `M` axes: folds every scalar of each rank-`M`
/// sub-region of [`split_iter`](ArrayView::split_iter) into that sub-region's
/// own accumulator, returning a new rank-`K` array of accumulators over the
/// leading axes, each starting at `initial`.
///
/// `K` is usually inferred from the returned array, but `M` needs to be spelled
/// explicitly: `inner_reduce::<_, _, _, _, M>(a, initial, f)`.
///
/// # Panics
///
/// Panics if `K + M != N`.
pub fn inner_reduce<T: Scalar, U: Scalar, const N: usize, const K: usize, const M: usize>(
    a: ArrayView<'_, T, N>,
    initial: U,
    f: impl FnMut(&mut U, &T),
) -> Array<U, K> {
    assert_eq!(K + M, N, "inner_reduce: K ({K}) + M ({M}) must be N ({N})");
    let extents = a.extents();
    let mut out = Array::full(std::array::from_fn(|d| extents[d]), initial);
    inner_reduce_into::<T, U, N, K, M>(&mut out, a, f);
    out
}

/// Reduces the trailing `M` axes: folds every scalar of each rank-`M`
/// sub-region of [`split_iter`](ArrayView::split_iter) into that sub-region's
/// accumulator in `out`, which is read as well as written and so carries over
/// between calls.
///
/// `K` is taken from `out`, but `M` needs to be spelled explicitly:
/// `inner_reduce_into::<_, _, _, _, M>(out, a, f)`.
///
/// # Panics
///
/// Panics if `K + M != N`, or if the leading `K` extents of `a` do not match
/// `out`.
pub fn inner_reduce_into<T: Scalar, U: Scalar, const N: usize, const K: usize, const M: usize>(
    out: &mut Array<U, K>,
    a: ArrayView<'_, T, N>,
    f: impl FnMut(&mut U, &T),
) {
    assert_eq!(
        K + M,
        N,
        "inner_reduce_into: K ({K}) + M ({M}) must be N ({N})"
    );
    let extents = a.extents();
    assert_eq!(
        std::array::from_fn::<usize, K, _>(|d| extents[d]),
        out.extents(),
        "inner_reduce_into: extents mismatch",
    );
    let mut f = f;
    for (acc, v) in out.data_mut().iter_mut().zip(a.split_iter::<K, M>()) {
        for_each(v, |_, x| f(acc, x));
    }
}

/// Reduces the leading `K` axes: folds every scalar of each rank-`M`
/// sub-region of [`split_iter`](ArrayView::split_iter) into the accumulator at
/// the same position within the sub-region, returning a new rank-`M` array of
/// accumulators over the trailing axes, each starting at `initial`.
///
/// `M` is usually inferred from the returned array, but `K` needs to be spelled
/// explicitly: `outer_reduce::<_, _, _, K, _>(a, initial, f)`.
///
/// # Panics
///
/// Panics if `K + M != N`.
pub fn outer_reduce<T: Scalar, U: Scalar, const N: usize, const K: usize, const M: usize>(
    a: ArrayView<'_, T, N>,
    initial: U,
    f: impl FnMut(&mut U, &T),
) -> Array<U, M> {
    assert_eq!(K + M, N, "outer_reduce: K ({K}) + M ({M}) must be N ({N})");
    let extents = a.extents();
    let mut out = Array::full(std::array::from_fn(|d| extents[K + d]), initial);
    outer_reduce_into::<T, U, N, K, M>(&mut out, a, f);
    out
}

/// Reduces the leading `K` axes: folds every scalar of each rank-`M`
/// sub-region of [`split_iter`](ArrayView::split_iter) into the accumulator of
/// `out` at the same position, which is read as well as written and so carries
/// over between calls.
///
/// `M` is taken from `out`, but `K` needs to be spelled explicitly:
/// `outer_reduce_into::<_, _, _, K, _>(out, a, f)`.
///
/// # Panics
///
/// Panics if `K + M != N`, or if the trailing `M` extents of `a` do not match
/// `out`.
pub fn outer_reduce_into<T: Scalar, U: Scalar, const N: usize, const K: usize, const M: usize>(
    out: &mut Array<U, M>,
    a: ArrayView<'_, T, N>,
    f: impl FnMut(&mut U, &T),
) {
    assert_eq!(
        K + M,
        N,
        "outer_reduce_into: K ({K}) + M ({M}) must be N ({N})"
    );
    let extents = a.extents();
    assert_eq!(
        std::array::from_fn::<usize, M, _>(|d| extents[K + d]),
        out.extents(),
        "outer_reduce_into: extents mismatch",
    );
    let mut f = f;
    let out = out.data_mut();
    for v in a.split_iter::<K, M>() {
        for_each(v, |i, x| f(&mut out[i], x));
    }
}
