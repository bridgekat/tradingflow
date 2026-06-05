//! Reshape / combine operators — port of `Stack`/`StackSync`/`Concat`/
//! `ConcatSync`. Axis-based and generic, with the shared interleaved-copy
//! helpers. The `*Sync` variants read the per-input notify tree.

use num_traits::Float;

use flowgraph::typed::{Port, SliceNotify, SliceRefs};

use super::op::Operator;
use crate::{Array, Scalar};

/// Shared runtime state for all four operators (outer × chunk layout).
pub struct ReshapeState {
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
}

#[inline(always)]
fn interleaved_copy<'a, T: Scalar>(
    output: &mut Array<T>,
    inputs: impl IntoIterator<Item = &'a Array<T>>,
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let out = output.as_mut_slice();
    let stride = n_inputs * chunk_size;
    for (input_idx, arr) in inputs.into_iter().enumerate() {
        let src = arr.as_slice();
        for outer in 0..outer_count {
            let src_offset = outer * chunk_size;
            let dst_offset = outer * stride + input_idx * chunk_size;
            out[dst_offset..dst_offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
        }
    }
}

#[inline(always)]
fn interleaved_copy_selective<T: Scalar>(
    output: &mut Array<T>,
    inputs: &SliceRefs<'_, Port<Array<T>>>,
    positions: impl IntoIterator<Item = usize>,
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let out = output.as_mut_slice();
    let stride = n_inputs * chunk_size;
    for pos in positions {
        let src = inputs.get(pos).as_slice();
        for outer in 0..outer_count {
            let src_offset = outer * chunk_size;
            let dst_offset = outer * stride + pos * chunk_size;
            out[dst_offset..dst_offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
        }
    }
}

// ---------------------------------------------------------------------------
// Stack — new axis, time-series semantics (copy all).
// ---------------------------------------------------------------------------

/// Stack N homogeneous arrays along a new axis.
#[derive(Clone)]
pub struct Stack<T: Scalar> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Stack<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> Operator for Stack<T> {
    type State = ReshapeState;
    type Inputs = [Port<Array<T>>];
    type Output = Array<T>;

    fn init(&self, inputs: SliceRefs<'_, Port<Array<T>>>) -> (ReshapeState, Array<T>) {
        assert!(!inputs.is_empty(), "Stack requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis <= first.len(), "axis out of bounds");
        let state = ReshapeState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = Vec::with_capacity(first.len() + 1);
        shape.extend_from_slice(&first[..self.axis]);
        shape.push(inputs.len());
        shape.extend_from_slice(&first[self.axis..]);
        (state, Array::zeros(&shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut ReshapeState,
        inputs: SliceRefs<'_, Port<Array<T>>>,
        output: &mut Array<T>,
        _produced: SliceNotify<'_, Port<Array<T>>>,
    ) -> bool {
        interleaved_copy(
            output,
            inputs.iter(),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}

// ---------------------------------------------------------------------------
// StackSync — new axis, message-passing (NaN-fill non-notified).
// ---------------------------------------------------------------------------

/// Stack N float arrays along a new axis, NaN-filling inputs that did not
/// notify this generation.
#[derive(Clone)]
pub struct StackSync<T: Scalar + Float> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> StackSync<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float> Operator for StackSync<T> {
    type State = ReshapeState;
    type Inputs = [Port<Array<T>>];
    type Output = Array<T>;

    fn init(&self, inputs: SliceRefs<'_, Port<Array<T>>>) -> (ReshapeState, Array<T>) {
        assert!(!inputs.is_empty(), "StackSync requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis <= first.len(), "axis out of bounds");
        let state = ReshapeState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = Vec::with_capacity(first.len() + 1);
        shape.extend_from_slice(&first[..self.axis]);
        shape.push(inputs.len());
        shape.extend_from_slice(&first[self.axis..]);
        let total: usize = shape.iter().product();
        (state, Array::from_vec(&shape, vec![T::nan(); total]))
    }

    #[inline(always)]
    fn compute(
        state: &mut ReshapeState,
        inputs: SliceRefs<'_, Port<Array<T>>>,
        output: &mut Array<T>,
        produced: SliceNotify<'_, Port<Array<T>>>,
    ) -> bool {
        for v in output.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_selective(
            output,
            &inputs,
            (0..produced.len()).filter(|&i| produced.get(i)),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}

// ---------------------------------------------------------------------------
// Concat — existing axis, time-series semantics.
// ---------------------------------------------------------------------------

/// Concatenate N homogeneous arrays along an existing axis.
#[derive(Clone)]
pub struct Concat<T: Scalar> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Concat<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> Operator for Concat<T> {
    type State = ReshapeState;
    type Inputs = [Port<Array<T>>];
    type Output = Array<T>;

    fn init(&self, inputs: SliceRefs<'_, Port<Array<T>>>) -> (ReshapeState, Array<T>) {
        assert!(!inputs.is_empty(), "Concat requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis < first.len(), "axis out of bounds");
        let state = ReshapeState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = first.to_vec();
        shape[self.axis] *= inputs.len();
        (state, Array::zeros(&shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut ReshapeState,
        inputs: SliceRefs<'_, Port<Array<T>>>,
        output: &mut Array<T>,
        _produced: SliceNotify<'_, Port<Array<T>>>,
    ) -> bool {
        interleaved_copy(
            output,
            inputs.iter(),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}

// ---------------------------------------------------------------------------
// ConcatSync — existing axis, message-passing.
// ---------------------------------------------------------------------------

/// Concatenate N float arrays along an existing axis, NaN-filling inputs that
/// did not notify this generation.
#[derive(Clone)]
pub struct ConcatSync<T: Scalar + Float> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> ConcatSync<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float> Operator for ConcatSync<T> {
    type State = ReshapeState;
    type Inputs = [Port<Array<T>>];
    type Output = Array<T>;

    fn init(&self, inputs: SliceRefs<'_, Port<Array<T>>>) -> (ReshapeState, Array<T>) {
        assert!(!inputs.is_empty(), "ConcatSync requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis < first.len(), "axis out of bounds");
        let state = ReshapeState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = first.to_vec();
        shape[self.axis] *= inputs.len();
        let total: usize = shape.iter().product();
        (state, Array::from_vec(&shape, vec![T::nan(); total]))
    }

    #[inline(always)]
    fn compute(
        state: &mut ReshapeState,
        inputs: SliceRefs<'_, Port<Array<T>>>,
        output: &mut Array<T>,
        produced: SliceNotify<'_, Port<Array<T>>>,
    ) -> bool {
        for v in output.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_selective(
            output,
            &inputs,
            (0..produced.len()).filter(|&i| produced.get(i)),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}
