//! Helper functions for splitting an [`ArrayView`] into multiple.

use super::ArrayView;
use crate::{Scalar, Slice, SliceReshape};

/// Splits `view` into consecutive chunks of the given `lengths` along the
/// existing axis `axis`, returning array views.
///
/// # Panics
///
/// Panics if `axis >= N` or `lengths` does not sum to the extent along `axis`.
pub fn split<'a, T: Scalar, const N: usize>(
    view: ArrayView<'a, T, N>,
    lengths: &[usize],
    axis: usize,
) -> Vec<ArrayView<'a, T, N>> {
    assert!(axis < N, "split: axis {axis} out of bounds for rank {N}");
    let extent = view.extents()[axis];
    let total: usize = lengths.iter().sum();
    assert_eq!(
        total, extent,
        "split: lengths {lengths:?} sum to {total}, expected extent {extent} along axis {axis}",
    );
    let mut start = 0;
    lengths
        .iter()
        .map(|&len| {
            let slices: [Slice; N] = std::array::from_fn(|d| {
                if d == axis {
                    Slice::new(start, Some(len), 1)
                } else {
                    Slice::new(0, None, 1)
                }
            });
            start += len;
            view.slice(slices)
        })
        .collect()
}

/// Splits `view` into slices of dimensions `M = N - 1` along the existing axis
/// `axis`, returning array views.
///
/// `M` needs to be spelled explicitly because stable Rust cannot form
/// `N - 1` in a type, so the relation is asserted at runtime.
///
/// # Panics
///
/// Panics if `M != N - 1` or `axis >= N`.
pub fn unstack<'a, T: Scalar, const N: usize, const M: usize>(
    view: ArrayView<'a, T, N>,
    axis: usize,
) -> Vec<ArrayView<'a, T, M>> {
    assert_eq!(M + 1, N, "unstack: M ({M}) must be N ({N}) - 1");
    assert!(axis < N, "unstack: axis {axis} out of bounds for rank {N}");
    (0..view.extents()[axis])
        .map(|i| {
            let slices: [SliceReshape; N] = std::array::from_fn(|d| {
                if d == axis {
                    SliceReshape::Index(i)
                } else {
                    SliceReshape::Slice(Slice::new(0, None, 1))
                }
            });
            view.slice_reshape(slices)
        })
        .collect()
}
