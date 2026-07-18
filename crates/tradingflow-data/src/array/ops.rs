//! The element-wise kernels and the row-major materialization primitive.
//!
//! Traversal order comes from [`Shape::offsets`](crate::Shape::offsets); these
//! add only the fast path for layouts that need no stride arithmetic at all.
//! [`apply_unary`] / [`apply_binary`] are `pub` because the operator library
//! (`tradingflow`'s `arith` module) is the caller: the element-wise kernels
//! live with the array they iterate.

use super::{Array, ArrayView};
use crate::Scalar;

/// Element-wise unary `out[i] = f(x[i])`, contiguous-fast / strided-slow. The
/// output scalar `U` may differ from the input's (a predicate maps `T` to
/// `bool`); `U == T` for the arithmetic operators.
pub fn apply_unary<T: Scalar, U: Scalar, const N: usize>(
    out: &mut Array<U, N>,
    x: ArrayView<T, N>,
    f: impl Fn(T) -> U,
) {
    let o = out.data_mut();
    if let Some(s) = x.as_slice() {
        for (dst, v) in o.iter_mut().zip(s) {
            *dst = f(v.clone());
        }
        return;
    }
    for (dst, off) in o.iter_mut().zip(x.shape.offsets()) {
        *dst = f(x.data[off].clone());
    }
}

/// Element-wise binary `out[i] = f(a[i], b[i])`, contiguous-fast / strided-slow.
/// `a` and `b` must share extents (asserted by the caller via output sizing).
/// As with [`apply_unary`], the output scalar `U` may differ from the inputs'.
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
        return;
    }
    // Strided: `a` and `b` share extents, so their offsets pair up row-major.
    for (dst, (oa, ob)) in o.iter_mut().zip(a.shape.offsets().zip(b.shape.offsets())) {
        *dst = f(a.data[oa].clone(), b.data[ob].clone());
    }
}

/// Copy a (possibly strided) view into a contiguous destination in row-major
/// order — the materialization primitive behind [`Array::assign`] and the
/// rank-changers.
pub(crate) fn write_row_major<T: Scalar, const N: usize>(dst: &mut [T], v: ArrayView<T, N>) {
    if let Some(s) = v.as_slice() {
        dst.clone_from_slice(s);
        return;
    }
    for (d, off) in dst.iter_mut().zip(v.shape.offsets()) {
        *d = v.data[off].clone();
    }
}
