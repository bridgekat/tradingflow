//! `StackSync` — new-axis combine that NaN-fills silent inputs.

use num_traits::Float;

use super::reshape::{ReshapeState, interleaved_copy_views_selective, self_axis_ok, stack_extents};
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, ArrayPorts};

/// Stack `N` float views along a new axis, NaN-filling inputs that did not
/// notify this generation.
#[derive(Clone)]
pub struct StackSync<T: Scalar + Float, const IN: usize, const OUT: usize> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float, const IN: usize, const OUT: usize> StackSync<T, IN, OUT> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float, const IN: usize, const OUT: usize> Operator for StackSync<T, IN, OUT> {
    type Inputs = ArrayPorts<T, IN>;
    type Outputs = ArrayPort<T, OUT>;
    type Context = Instant;
    type State = ReshapeState<T, OUT>;

    fn init(self, (_, views): (&[bool], &[ArrayView<'_, T, IN>])) -> Self::State {
        assert!(!views.is_empty(), "StackSync requires at least one input");
        let first = views[0].extents();
        assert!(self_axis_ok(self.axis, IN, true), "axis out of bounds");
        ReshapeState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: views.len(),
            out: Array::full(
                stack_extents::<IN, OUT>(first, self.axis, views.len()),
                T::nan(),
            ),
        }
    }

    fn compute<'a, 'b: 'a>(
        (flags, views): (&'a [bool], &'a [ArrayView<'a, T, IN>]),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, OUT>) {
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
        _: (&'a [bool], &'a [ArrayView<'a, T, IN>]),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        (false, state.out.view())
    }
}

/// Like [`stack`](crate::operators::array::stack), but emits `NaN` for inputs
/// that have not notified this tick.
pub fn stack_sync<T: Scalar + Float, const IN: usize, const OUT: usize>(
    axis: usize,
) -> StackSync<T, IN, OUT> {
    StackSync::new(axis)
}
