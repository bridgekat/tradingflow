//! Reshape / combine operators — port of `Stack`/`StackSync`/`Concat`/
//! `ConcatSync`. Axis-based and generic, with the shared interleaved-copy
//! helpers. The `*Sync` variants read the per-input notify plane.

use num_traits::Float;

use flowgraph::typed::{Operator, Port, Ports};

use crate::{Array, Scalar};

/// Shared runtime state for all four operators: the axis config, the
/// outer × chunk layout (sized on the `init` build call), and the output
/// buffer.
pub struct ReshapeState<T: Scalar> {
    axis: usize,
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
    out: Array<T>,
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
    inputs: &[&Array<T>],
    positions: impl IntoIterator<Item = usize>,
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let out = output.as_mut_slice();
    let stride = n_inputs * chunk_size;
    for pos in positions {
        let src = inputs[pos].as_slice();
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
    type Inputs = Ports<Array<T>>;
    type Outputs = Port<Array<T>>;
    type State = ReshapeState<T>;

    fn init(self) -> ReshapeState<T> {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, values): (&'a [bool], &'a [&'a Array<T>]),
        state: &'b mut ReshapeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            assert!(!values.is_empty(), "Stack requires at least one input");
            let first = values[0].shape();
            assert!(state.axis <= first.len(), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = values.len();
            let mut shape = Vec::with_capacity(first.len() + 1);
            shape.extend_from_slice(&first[..state.axis]);
            shape.push(values.len());
            shape.extend_from_slice(&first[state.axis..]);
            state.out = Array::zeros(&shape);
            return (false, &state.out);
        }
        interleaved_copy(
            &mut state.out,
            values.iter().copied(),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a Array<T>]),
        state: &'b ReshapeState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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
    type Inputs = Ports<Array<T>>;
    type Outputs = Port<Array<T>>;
    type State = ReshapeState<T>;

    fn init(self) -> ReshapeState<T> {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (flags, values): (&'a [bool], &'a [&'a Array<T>]),
        state: &'b mut ReshapeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            assert!(!values.is_empty(), "StackSync requires at least one input");
            let first = values[0].shape();
            assert!(state.axis <= first.len(), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = values.len();
            let mut shape = Vec::with_capacity(first.len() + 1);
            shape.extend_from_slice(&first[..state.axis]);
            shape.push(values.len());
            shape.extend_from_slice(&first[state.axis..]);
            let total: usize = shape.iter().product();
            state.out = Array::from_vec(&shape, vec![T::nan(); total]);
            return (false, &state.out);
        }
        for v in state.out.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_selective(
            &mut state.out,
            values,
            (0..flags.len()).filter(|&i| flags[i]),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a Array<T>]),
        state: &'b ReshapeState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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
    type Inputs = Ports<Array<T>>;
    type Outputs = Port<Array<T>>;
    type State = ReshapeState<T>;

    fn init(self) -> ReshapeState<T> {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, values): (&'a [bool], &'a [&'a Array<T>]),
        state: &'b mut ReshapeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            assert!(!values.is_empty(), "Concat requires at least one input");
            let first = values[0].shape();
            assert!(state.axis < first.len(), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = values.len();
            let mut shape = first.to_vec();
            shape[state.axis] *= values.len();
            state.out = Array::zeros(&shape);
            return (false, &state.out);
        }
        interleaved_copy(
            &mut state.out,
            values.iter().copied(),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a Array<T>]),
        state: &'b ReshapeState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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
    type Inputs = Ports<Array<T>>;
    type Outputs = Port<Array<T>>;
    type State = ReshapeState<T>;

    fn init(self) -> ReshapeState<T> {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (flags, values): (&'a [bool], &'a [&'a Array<T>]),
        state: &'b mut ReshapeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            assert!(!values.is_empty(), "ConcatSync requires at least one input");
            let first = values[0].shape();
            assert!(state.axis < first.len(), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = values.len();
            let mut shape = first.to_vec();
            shape[state.axis] *= values.len();
            let total: usize = shape.iter().product();
            state.out = Array::from_vec(&shape, vec![T::nan(); total]);
            return (false, &state.out);
        }
        for v in state.out.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_selective(
            &mut state.out,
            values,
            (0..flags.len()).filter(|&i| flags[i]),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a Array<T>]),
        state: &'b ReshapeState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}
