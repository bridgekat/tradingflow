use super::map::ArrayMap;
use super::view::DeriveView;
use crate::data::{Array, ArrayView, Instant, Scalar, SliceReshape, array};
use crate::graph::Operator;
use crate::ports::ArrayPort;

/// Selects a single index along an axis, squeezing that axis: [`ArrayView::slice_reshape`].
#[allow(clippy::type_complexity)]
pub fn select_at<T: Scalar, const N: usize, const M: usize>(
    index: usize,
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    assert!(
        M + 1 == N,
        "select_at: output ndim ({M}) must be input ndim ({N}) minus one"
    );
    let mut slices = [SliceReshape::from(..); N];
    slices[axis] = SliceReshape::from(index);
    DeriveView::new(move |a| ArrayView::slice_reshape(&a, slices))
}

/// Selects multiple indices along an axis: [`array::select`].
#[allow(clippy::type_complexity)]
pub fn select<T: Scalar, const N: usize>(
    indices: Vec<usize>,
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let init = {
        let indices = indices.clone();
        move |x: ArrayView<'_, T, N>| array::select(x, &indices, axis)
    };
    let update = move |out: &mut Array<T, N>, x: ArrayView<'_, T, N>| {
        array::select_into(out, x, &indices, axis);
    };
    ArrayMap::new(init, update)
}

/// Selects indices by mask along an axis: [`array::select_mask`].
#[allow(clippy::type_complexity)]
pub fn select_mask<T: Scalar, const N: usize>(
    mask: Vec<bool>,
    axis: usize,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let init = {
        let mask = mask.clone();
        move |x: ArrayView<'_, T, N>| array::select_mask(x, &mask, axis)
    };
    let update = move |out: &mut Array<T, N>, x: ArrayView<'_, T, N>| {
        array::select_mask_into(out, x, &mask, axis);
    };
    ArrayMap::new(init, update)
}
