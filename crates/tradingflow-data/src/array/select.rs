//! Helper functions for indexing an [`ArrayView`] along an axis by a list of
//! indices or a boolean mask.

use super::{Array, ArrayView};
use crate::{Layout, Scalar, Slice, layout};

/// Selects `indices` in order (repetition allowed) along the existing axis
/// `axis`, returning a new array whose extent along `axis` is `indices.len()`.
///
/// # Panics
///
/// Panics if `axis >= N` or any index is out of bounds along `axis`.
pub fn select<T: Scalar, const N: usize>(
    view: ArrayView<T, N>,
    indices: &[usize],
    axis: usize,
) -> Array<T, N> {
    assert!(axis < N, "select: axis {axis} out of bounds for rank {N}");
    let extent = view.extents()[axis];
    for &i in indices {
        assert!(
            i < extent,
            "select: index {i} out of bounds for extent {extent} along axis {axis}",
        );
    }
    let mut extents = view.extents();
    extents[axis] = indices.len();
    let mut out = Array::zeros(extents);
    select_into(&mut out, view, indices, axis);
    out
}

/// Selects `indices` in order (repetition allowed) along the existing axis
/// `axis`, writing into `out`. Every index is assumed to be in bounds along
/// `axis`.
///
/// # Panics
///
/// Panics if `axis >= N`, or if `out` does not have the extents of `view` with
/// `indices.len()` along `axis`.
pub fn select_into<T: Scalar, const N: usize>(
    out: &mut Array<T, N>,
    view: ArrayView<T, N>,
    indices: &[usize],
    axis: usize,
) {
    assert!(
        axis < N,
        "select_into: axis {axis} out of bounds for rank {N}"
    );
    let mut extents = view.extents();
    extents[axis] = indices.len();
    assert_eq!(out.extents(), extents, "select_into: extents mismatch");
    select_iter_into(
        out.data_mut(),
        view,
        indices.iter().copied(),
        axis,
        indices.len(),
    );
}

/// Selects the indices where `mask` holds `true` along the existing axis
/// `axis`, returning a new array whose extent along `axis` is the number of
/// `true` entries.
///
/// # Panics
///
/// Panics if `axis >= N` or `mask.len()` differs from the extent along
/// `axis`.
pub fn select_mask<T: Scalar, const N: usize>(
    view: ArrayView<T, N>,
    mask: &[bool],
    axis: usize,
) -> Array<T, N> {
    assert!(
        axis < N,
        "select_mask: axis {axis} out of bounds for rank {N}"
    );
    let extent = view.extents()[axis];
    assert_eq!(
        mask.len(),
        extent,
        "select_mask: mask length {} != extent {extent} along axis {axis}",
        mask.len(),
    );
    let mut extents = view.extents();
    extents[axis] = mask.iter().filter(|&&keep| keep).count();
    let mut out = Array::zeros(extents);
    select_mask_into(&mut out, view, mask, axis);
    out
}

/// Selects the indices where `mask` holds `true` along the existing axis
/// `axis`, writing into `out`. The `mask` is assumed to hold one flag per
/// index along `axis`.
///
/// # Panics
///
/// Panics if `axis >= N`, or if `out` does not have the extents of `view` with
/// the number of `true` entries along `axis`.
pub fn select_mask_into<T: Scalar, const N: usize>(
    out: &mut Array<T, N>,
    view: ArrayView<T, N>,
    mask: &[bool],
    axis: usize,
) {
    let count = mask.iter().filter(|&&keep| keep).count();
    assert!(
        axis < N,
        "select_mask_into: axis {axis} out of bounds for rank {N}"
    );
    let mut extents = view.extents();
    extents[axis] = count;
    assert_eq!(out.extents(), extents, "select_mask_into: extents mismatch");
    let indices = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &keep)| keep.then_some(i));
    select_iter_into(out.data_mut(), view, indices, axis, count);
}

fn select_iter_into<T: Scalar, const N: usize>(
    out: &mut [T],
    view: ArrayView<T, N>,
    indices: impl Iterator<Item = usize>,
    axis: usize,
    count: usize,
) {
    let pin = |i: usize| -> [Slice; N] {
        std::array::from_fn(|d| {
            if d == axis {
                Slice::new(i, Some(1), 1)
            } else {
                Slice::new(0, None, 1)
            }
        })
    };
    let mut out_extents = view.extents();
    out_extents[axis] = count;
    let out_layout = layout::RowMajor::new(out_extents);
    let data = view.data();
    for (k, index) in indices.enumerate() {
        let (src_base, src) = view.layout().slice(pin(index));
        let (dst_base, dst) = out_layout.slice(pin(k));
        if src.is_contiguous() && dst.is_contiguous() {
            let len = src.len();
            out[dst_base..dst_base + len].clone_from_slice(&data[src_base..src_base + len]);
        } else {
            for (s, d) in src.iter().zip(dst.iter()) {
                out[dst_base + d] = data[src_base + s].clone();
            }
        }
    }
}
