//! Layout helpers shared by the `NaN`-fill combine family (`StackSync`,
//! `ConcatSync`).
//!
//! Only [`ReshapeState`] is re-exported by the parent module; the copy and
//! extent helpers stay private to `structural`.

use crate::data::{Array, ArrayView, Scalar};

/// Shared runtime state: the outer × chunk combine layout (sized at init from
/// the build-time input views) and the output buffer.
pub struct ReshapeState<T: Scalar, const OUT: usize> {
    pub(super) outer_count: usize,
    pub(super) chunk_size: usize,
    pub(super) n_inputs: usize,
    pub(super) out: Array<T, OUT>,
}

/// Interleave the selected `positions` of `inputs` (each materialized
/// row-major) into `output` along the combine layout, leaving the other
/// positions untouched (the caller pre-fills them, e.g. with `NaN`).
pub(super) fn interleaved_copy_views_selective<T: Scalar, const IN: usize>(
    output: &mut [T],
    inputs: &[ArrayView<T, IN>],
    positions: impl IntoIterator<Item = usize>,
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let stride = n_inputs * chunk_size;
    for pos in positions {
        let src = inputs[pos].to_contiguous();
        for outer in 0..outer_count {
            let src_offset = outer * chunk_size;
            let dst_offset = outer * stride + pos * chunk_size;
            output[dst_offset..dst_offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
        }
    }
}

/// Output extents for a stack-along-new-axis (`OUT == IN + 1`): insert
/// `n_inputs` at `axis`.
pub(super) fn stack_extents<const IN: usize, const OUT: usize>(
    input_extents: [usize; IN],
    axis: usize,
    n_inputs: usize,
) -> [usize; OUT] {
    let mut v = Vec::with_capacity(IN + 1);
    v.extend_from_slice(&input_extents[..axis]);
    v.push(n_inputs);
    v.extend_from_slice(&input_extents[axis..]);
    <[usize; OUT]>::try_from(v.as_slice())
        .unwrap_or_else(|_| panic!("Stack: OUT ({OUT}) must be IN ({IN}) + 1"))
}

pub(super) fn self_axis_ok(axis: usize, rank: usize, allow_equal: bool) -> bool {
    if allow_equal {
        axis <= rank
    } else {
        axis < rank
    }
}
