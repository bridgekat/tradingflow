//! `Stack` — combine N inputs along a **new** axis (carry semantics).

use super::reshape::{ReshapeState, interleaved_copy_views, self_axis_ok, stack_extents};
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, ArrayPorts};

/// Stack `N` homogeneous rank-`IN` views along a **new** axis into the owned
/// rank-`OUT` (`= IN + 1`) cross-section. Reads **every** input each generation
/// (the carry join), relying on the no-notify⟹unchanged contract.
#[derive(Clone)]
pub struct Stack<T: Scalar, const IN: usize, const OUT: usize> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Stack<T, IN, OUT> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, const IN: usize, const OUT: usize> Operator for Stack<T, IN, OUT> {
    type Inputs = ArrayPorts<T, IN>;
    type Outputs = ArrayPort<T, OUT>;
    type Context = Instant;
    type State = ReshapeState<T, OUT>;

    fn init(self, (_, views): (&[bool], &[ArrayView<'_, T, IN>])) -> Self::State {
        assert!(!views.is_empty(), "Stack requires at least one input");
        let first = views[0].extents();
        assert!(self_axis_ok(self.axis, IN, true), "axis out of bounds");
        ReshapeState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: views.len(),
            out: Array::zeros(stack_extents::<IN, OUT>(first, self.axis, views.len())),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, views): (&'a [bool], &'a [ArrayView<'a, T, IN>]),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, OUT>) {
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
        _: (&'a [bool], &'a [ArrayView<'a, T, IN>]),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        (false, state.out.view())
    }
}

/// Stack `N` rank-`IN` inputs along a new `axis`, carrying un-notified inputs
/// forward at their last value.
pub fn stack<T: Scalar, const IN: usize, const OUT: usize>(axis: usize) -> Stack<T, IN, OUT> {
    Stack::new(axis)
}
