use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Select elements along an axis into an owned, contiguous rank-`OUT` array
/// (`OUT == IN` for a plain selection, `OUT == IN - 1` when squeezing a single
/// index). Accepts a strided input view; the selection is the **materialization
/// point** of a view chain — it retains the last computed selection in owned
/// state, preserving the carry semantics downstream `Stack`-style readers rely
/// on.
pub struct Select<T: Scalar, const IN: usize, const OUT: usize> {
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Clone for Select<T, IN, OUT> {
    fn clone(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            axis: self.axis,
            squeeze: self.squeeze,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const IN: usize, const OUT: usize> Select<T, IN, OUT> {
    pub fn new(indices: Vec<usize>, axis: usize, squeeze: bool) -> Self {
        assert!(
            !squeeze || indices.len() == 1,
            "squeeze requires exactly one index, got {}",
            indices.len(),
        );
        assert!(
            OUT == IN - squeeze as usize,
            "Select: OUT ({OUT}) must be IN ({IN}) minus {} (squeeze={squeeze})",
            squeeze as usize,
        );
        Self {
            indices,
            axis,
            squeeze,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Select`]: the resolved flat index map (computed in
/// `init` once the input shape is known) and the output buffer.
pub struct SelectState<T: Scalar, const OUT: usize> {
    index_map: Vec<usize>,
    out: Array<T, OUT>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Operator for Select<T, IN, OUT> {
    type Inputs = ArrayPort<T, IN>;
    type Outputs = ArrayPort<T, OUT>;
    type Context = Instant;
    type State = SelectState<T, OUT>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, IN>)) -> Self::State {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let input_extents = x.extents();
        let index_map = compute_select_map(&input_extents, &self.indices, self.axis);
        let out_extents =
            select_out_extents::<OUT>(&input_extents, self.indices.len(), self.axis, self.squeeze);
        // Seed the initial output with the actual selection of the build-time
        // input (NOT zeros — a fabricated finite observation leaks through
        // carry-style consumers; the faithful selection of a NaN-initialised
        // panel correctly reads "no data yet").
        let out = Array::from_parts(
            out_extents,
            index_map.iter().map(|&s| src[s].clone()).collect(),
        );
        SelectState { index_map, out }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, IN>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        for (dst_i, &src_i) in state.index_map.iter().enumerate() {
            dst[dst_i] = src[src_i].clone();
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, IN>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        (false, state.out.view())
    }
}

/// The row-major flat index map of selecting `indices` along `axis`.
fn compute_select_map(input_extents: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
    if input_extents.is_empty() {
        return indices.to_vec();
    }
    let outer: usize = input_extents[..axis].iter().product();
    let inner: usize = input_extents[axis + 1..].iter().product();
    let axis_size = input_extents[axis];
    let mut map = Vec::with_capacity(outer * indices.len() * inner);
    for o in 0..outer {
        for &idx in indices {
            for i in 0..inner {
                map.push(o * axis_size * inner + idx * inner + i);
            }
        }
    }
    map
}

/// The output extents of a select, as a static `[usize; OUT]`.
fn select_out_extents<const OUT: usize>(
    input_extents: &[usize],
    n_selected: usize,
    axis: usize,
    squeeze: bool,
) -> [usize; OUT] {
    let mut v = input_extents.to_vec();
    if v.is_empty() {
        v = vec![n_selected];
    } else {
        v[axis] = n_selected;
    }
    if squeeze && n_selected == 1 && v.len() > axis {
        v.remove(axis);
    }
    <[usize; OUT]>::try_from(v.as_slice())
        .unwrap_or_else(|_| panic!("Select: output rank {} != OUT {OUT}", v.len()))
}

/// Gather `indices` along `axis` into an owned output, optionally squeezing a
/// length-1 axis.
pub fn select<T: Scalar, const IN: usize, const OUT: usize>(
    indices: Vec<usize>,
    axis: usize,
    squeeze: bool,
) -> Select<T, IN, OUT> {
    Select::new(indices, axis, squeeze)
}

/// [`select`] at a single index along `axis`, squeezing that axis.
pub fn select_at<T: Scalar, const IN: usize, const OUT: usize>(
    index: usize,
    axis: usize,
) -> Select<T, IN, OUT> {
    Select::new(vec![index], axis, true)
}

/// [`select`] along `axis`, keeping the axis: `select(indices, axis, false)`.
pub fn select_many<T: Scalar, const IN: usize, const OUT: usize>(
    indices: Vec<usize>,
    axis: usize,
) -> Select<T, IN, OUT> {
    Select::new(indices, axis, false)
}
