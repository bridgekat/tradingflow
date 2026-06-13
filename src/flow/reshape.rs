//! Reshape / combine operators — `Stack`/`StackSync`/`Concat`/`ConcatSync`
//! (N → 1 combine) and [`Split`] (1 → N fan-out). Axis-based and generic, with
//! the shared interleaved-copy helpers. The `*Sync` variants read the
//! per-input notify plane.

use num_traits::Float;

use flowgraph::typed::{Arena, Interface, Operator, RefPort, RefPorts, RefViewPorts, Segment};

use super::op::ArrayValue;
use crate::{Array, ArraySlice, Scalar};

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

// View-input variants of the interleaved copy: identical layout, but the
// sources are borrowed [`ArraySlice`] views (the zero-copy join inputs) rather
// than owned `Array`s.

#[inline(always)]
fn interleaved_copy_views<T: Scalar>(
    output: &mut Array<T>,
    inputs: &[&ArraySlice<'_, T>],
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let out = output.as_mut_slice();
    let stride = n_inputs * chunk_size;
    for (input_idx, arr) in inputs.iter().enumerate() {
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
fn interleaved_copy_views_selective<T: Scalar>(
    output: &mut Array<T>,
    inputs: &[&ArraySlice<'_, T>],
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
    type Inputs = RefPorts<Array<T>>;
    type Outputs = RefPort<Array<T>>;
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
    type Inputs = RefPorts<Array<T>>;
    type Outputs = RefPort<Array<T>>;
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
// StackView — new axis, time-series semantics, zero-copy view inputs (copy all).
// ---------------------------------------------------------------------------

/// [`Stack`] over borrowed [`ArraySlice`] view inputs
/// (`RefViewPorts<ArrayValue<T>>`): stacks `N` per-stock row views along a new
/// axis into the owned `[N, ...]` cross-section. Like `Stack`, it reads **every**
/// input each generation (the carry join), relying on the no-notify⟹unchanged
/// contract — an un-notified view reads its producer's stable storage, i.e. that
/// stock's last notified value. The materialization into the cross-section is
/// the irreducible panel→cross-section data movement; the per-stock selections
/// upstream stay copy-free ([`SliceView`](super::SliceView)).
#[derive(Clone)]
pub struct StackView<T: Scalar> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> StackView<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> Operator for StackView<T> {
    type Inputs = RefViewPorts<ArrayValue<T>>;
    type Outputs = RefPort<Array<T>>;
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
        (_, views): (&'a [bool], &'a [&'a ArraySlice<'a, T>]),
        state: &'b mut ReshapeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            assert!(!views.is_empty(), "StackView requires at least one input");
            let first = views[0].shape();
            assert!(state.axis <= first.len(), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = views.len();
            let mut shape = Vec::with_capacity(first.len() + 1);
            shape.extend_from_slice(&first[..state.axis]);
            shape.push(views.len());
            shape.extend_from_slice(&first[state.axis..]);
            state.out = Array::zeros(&shape);
            return (false, &state.out);
        }
        interleaved_copy_views(
            &mut state.out,
            views,
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a ArraySlice<'a, T>]),
        state: &'b ReshapeState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// StackSyncView — new axis, message-passing, zero-copy view inputs.
// ---------------------------------------------------------------------------

/// [`StackSync`] over borrowed [`ArraySlice`] view inputs
/// (`RefViewPorts<ArrayValue<T>>`): NaN-fills inputs that did not notify this
/// generation and copies only the notified rows — so it never reads an
/// un-notified view, the view-edge counterpart of `StackSync`.
#[derive(Clone)]
pub struct StackSyncView<T: Scalar + Float> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> StackSyncView<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float> Operator for StackSyncView<T> {
    type Inputs = RefViewPorts<ArrayValue<T>>;
    type Outputs = RefPort<Array<T>>;
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
        (flags, views): (&'a [bool], &'a [&'a ArraySlice<'a, T>]),
        state: &'b mut ReshapeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            assert!(!views.is_empty(), "StackSyncView requires at least one input");
            let first = views[0].shape();
            assert!(state.axis <= first.len(), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = views.len();
            let mut shape = Vec::with_capacity(first.len() + 1);
            shape.extend_from_slice(&first[..state.axis]);
            shape.push(views.len());
            shape.extend_from_slice(&first[state.axis..]);
            let total: usize = shape.iter().product();
            state.out = Array::from_vec(&shape, vec![T::nan(); total]);
            return (false, &state.out);
        }
        for v in state.out.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_views_selective(
            &mut state.out,
            views,
            (0..flags.len()).filter(|&i| flags[i]),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a ArraySlice<'a, T>]),
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
    type Inputs = RefPorts<Array<T>>;
    type Outputs = RefPort<Array<T>>;
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
    type Inputs = RefPorts<Array<T>>;
    type Outputs = RefPort<Array<T>>;
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

// ---------------------------------------------------------------------------
// Split — zero-copy axis-0 fan-out into view ports (the inverse of `Stack`).
// ---------------------------------------------------------------------------

/// Split an `[N, ...]` array along axis 0 into `N` per-row output ports — the
/// `1 → N` inverse of [`Stack`], replacing `N` separate row-`Select` nodes
/// with one scheduling unit. The port count is declared explicitly at
/// construction (`axis_size`), making the static graph structure independent
/// of the build-time input value; the build call asserts the input's axis-0
/// size matches.
///
/// **Zero-copy**: each output is a borrowed [`ArraySlice`] view of the input's
/// row, re-derived from the fresh input on every invocation and lent through
/// the per-generation [`Arena`] — no row data is copied. All rows notify
/// exactly when the input notifies.
///
/// Implements [`Segment`] directly (not the gated `Operator`): views cannot be
/// re-lent through `&State`, so every invocation rebuilds the planes and gates
/// manually, expressing the gate's verdict in the notify flags.
pub struct Split<T: Scalar> {
    axis_size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Split<T> {
    /// `axis_size` is the declared input axis-0 size = the output port count.
    pub fn new(axis_size: usize) -> Self {
        assert!(axis_size > 0, "Split requires at least one output port");
        Self {
            axis_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`Split`]: the declared port count and the per-generation
/// arena backing the notify/value planes.
pub struct SplitState {
    axis_size: usize,
    arena: Arena,
}

impl<T: Scalar> Segment for Split<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefViewPorts<ArrayValue<T>>;
    type State = SplitState;

    fn init(self) -> SplitState {
        SplitState {
            axis_size: self.axis_size,
            arena: Arena::new(),
        }
    }

    fn compute<'a, 'b: 'a>(
        (notified, x): (bool, &'a Array<T>),
        state: &'b mut SplitState,
        init: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let n = state.axis_size;
        let shape = x.shape();
        if init {
            assert!(
                shape.first() == Some(&n),
                "Split: input shape {shape:?} does not have declared axis-0 size {n}",
            );
        }
        let row_shape = &shape[1..];
        let chunk: usize = row_shape.iter().product();
        let src = x.as_slice();
        let alloc = state.arena.reset();
        let flags = alloc.slice(std::iter::repeat_n(notified && !init, n));
        let views = alloc.slice((0..n).map(|i| {
            &*alloc.alloc(ArraySlice::new(&src[i * chunk..(i + 1) * chunk], row_shape))
        }));
        (flags, views)
    }
}
