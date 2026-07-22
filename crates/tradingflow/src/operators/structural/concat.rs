//! `Concat` — combine N inputs along an **existing** axis (rank-preserving).

use super::reshape::{ReshapeState, interleaved_copy_views};
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, ArrayPorts};

/// Concatenate `N` homogeneous rank-`N` views along an **existing** axis.
#[derive(Clone)]
pub struct Concat<T: Scalar, const N: usize> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const N: usize> Concat<T, N> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Operator for Concat<T, N> {
    type Inputs = ArrayPorts<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ReshapeState<T, N>;

    fn init(self, (_, views): (&[bool], &[ArrayView<'_, T, N>])) -> Self::State {
        assert!(!views.is_empty(), "Concat requires at least one input");
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
            out: Array::zeros(ext),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, views): (&'a [bool], &'a [ArrayView<'a, T, N>]),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        interleaved_copy_views(
            state.out.data_mut(),
            views,
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

/// Concatenate `N` inputs along an existing `axis` (carry semantics).
pub fn concat<T: Scalar, const N: usize>(axis: usize) -> Concat<T, N> {
    Concat::new(axis)
}
