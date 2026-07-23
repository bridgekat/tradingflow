//! `ConcatSync` — existing-axis combine that NaN-fills silent inputs.

use num_traits::Float;

use super::reshape::{ReshapeState, interleaved_copy_views_selective};
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, ArrayPorts};

/// Concatenate `N` float views along an existing axis, NaN-filling inputs that
/// did not notify this generation.
#[derive(Clone)]
pub struct ConcatSync<T: Scalar + Float, const N: usize> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> ConcatSync<T, N> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Operator for ConcatSync<T, N> {
    type Inputs = ArrayPorts<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ReshapeState<T, N>;

    fn init(self, (_, views): (&[bool], &[ArrayView<'_, T, N>])) -> Self::State {
        assert!(!views.is_empty(), "ConcatSync requires at least one input");
        let mut ext = views[0].extents();
        assert!(self.axis < N, "axis out of bounds");
        let (outer_count, chunk_size) = (
            ext[..self.axis].iter().product(),
            ext[self.axis..].iter().product(),
        );
        ext[self.axis] *= views.len();
        ReshapeState {
            outer_count,
            chunk_size,
            n_inputs: views.len(),
            out: Array::full(ext, T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (flags, views): (&'a [bool], &'a [ArrayView<'a, T, N>]),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        state.out.data_mut().fill(T::nan());
        interleaved_copy_views_selective(
            state.out.data_mut(),
            views,
            (0..flags.len()).filter(|&i| flags[i]),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [ArrayView<'a, T, N>]),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// Like [`concat`](fn@crate::operators::array::concat), but emits `NaN` for
/// inputs that have not notified.
pub fn concat_sync<T: Scalar + Float, const N: usize>(axis: usize) -> ConcatSync<T, N> {
    ConcatSync::new(axis)
}
