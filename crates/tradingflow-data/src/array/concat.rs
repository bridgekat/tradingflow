//! Helper functions for combining multiple [`ArrayView`]s into an owned
//! [`Array`], with built-in fast paths for row-major contiguous layouts.

use super::{Array, ArrayView};
use crate::Scalar;

/// Concatenates `views` along the existing axis `axis`, returning a new array.
///
/// The extents of all views must match on every axis except `axis`; the
/// output extent along `axis` is the sum of the input extents.
///
/// # Panics
///
/// Panics if `views` is empty, `axis >= N`, or the extents mismatch on any
/// axis other than `axis`.
pub fn concat<T: Scalar, const N: usize>(views: &[ArrayView<T, N>], axis: usize) -> Array<T, N> {
    assert!(!views.is_empty(), "concat: requires at least one view");
    assert!(axis < N, "concat: axis {axis} out of bounds for rank {N}");
    let first = views[0].extents();
    let mut total = 0;
    for v in views {
        let e = v.extents();
        assert!(
            (0..N).all(|d| d == axis || e[d] == first[d]),
            "concat: extents {e:?} incompatible with {first:?} along axis {axis}",
        );
        total += e[axis];
    }
    let mut extents = first;
    extents[axis] = total;
    let mut out = Array::zeros(extents);
    concat_into(out.data_mut(), views, axis);
    out
}

/// Concatenates `views` along the existing axis `axis`, writing into the
/// row-major buffer `out`. The buffer is assumed to hold exactly the
/// concatenated scalar count, and the views' extents are assumed to match on
/// every axis other than `axis`.
pub fn concat_into<T: Scalar, const N: usize>(
    out: &mut [T],
    views: &[ArrayView<T, N>],
    axis: usize,
) {
    let Some(first) = views.first().map(|v| v.extents()) else {
        return;
    };
    let outer_count: usize = first[..axis].iter().product();
    let inner_count: usize = first[axis + 1..].iter().product();
    let chunks: Vec<usize> = views
        .iter()
        .map(|v| v.extents()[axis] * inner_count)
        .collect();
    interleave_into(out, views, outer_count, &chunks);
}

/// Stacks `views` along a new axis inserted at position `axis`, returning a
/// new array whose extent along `axis` is `views.len()`.
///
/// All views must have identical extents.
///
/// `M` needs to be spelled explicitly because stable Rust cannot form
/// `N + 1` in a type, so the relation is asserted at runtime.
///
/// # Panics
///
/// Panics if `M != N + 1`, `views` is empty, `axis > N`, or the extents
/// mismatch on any axis.
pub fn stack<T: Scalar, const N: usize, const M: usize>(
    views: &[ArrayView<T, N>],
    axis: usize,
) -> Array<T, M> {
    assert_eq!(M, N + 1, "stack: M ({M}) must be N + 1 ({})", N + 1);
    assert!(!views.is_empty(), "stack: requires at least one view");
    assert!(axis <= N, "stack: axis {axis} out of bounds for rank {N}");
    let first = views[0].extents();
    for v in views {
        assert_eq!(v.extents(), first, "stack: extents mismatch");
    }
    let mut extents = [0; M];
    extents[..axis].copy_from_slice(&first[..axis]);
    extents[axis] = views.len();
    extents[axis + 1..].copy_from_slice(&first[axis..]);
    let mut out = Array::zeros(extents);
    stack_into(out.data_mut(), views, axis);
    out
}

/// Stacks `views` along a new axis inserted at position `axis`, writing into
/// the row-major buffer `out`. The buffer is assumed to hold exactly the
/// stacked scalar count, and the views are assumed to have identical extents.
pub fn stack_into<T: Scalar, const N: usize>(
    out: &mut [T],
    views: &[ArrayView<T, N>],
    axis: usize,
) {
    let Some(first) = views.first().map(|v| v.extents()) else {
        return;
    };
    let outer_count: usize = first[..axis].iter().product();
    let inner_count: usize = first[axis..].iter().product();
    let chunks = vec![inner_count; views.len()];
    interleave_into(out, views, outer_count, &chunks);
}

/// Row-major interleave: for each of the `outer_count` outer indices, writes
/// each view's next `chunks[i]` scalars in view order. Assumes each view holds
/// exactly `outer_count * chunks[i]` scalars and `out` holds their total.
fn interleave_into<T: Scalar, const N: usize>(
    out: &mut [T],
    views: &[ArrayView<T, N>],
    outer_count: usize,
    chunks: &[usize],
) {
    let mats: Vec<_> = views.iter().map(|v| v.to_contiguous()).collect();
    let mut dst = 0;
    for outer in 0..outer_count {
        for (src, &chunk) in mats.iter().zip(chunks) {
            out[dst..dst + chunk].clone_from_slice(&src[outer * chunk..(outer + 1) * chunk]);
            dst += chunk;
        }
    }
}
