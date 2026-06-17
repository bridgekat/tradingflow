//! Reshape / combine operators — `Stack`/`StackSync` (N → 1 along a **new**
//! axis, `OUT == IN + 1`), `Concat`/`ConcatSync` (N → 1 along an **existing**
//! axis, rank-preserving), and [`Split`] (1 → N row fan-out, `OUT == IN - 1`).
//!
//! In the view currency every multi-input combine takes
//! `RefViewPorts<ArrayValue<T, IN>>` (a slice of strided views), so the old
//! owned/`*View` split collapses: `StackView`/`StackSyncView` are now type
//! aliases of `Stack`/`StackSync`. The combine into the output cross-section is
//! the irreducible panel→cross-section data movement (each input materialized
//! via `to_contiguous`); the per-stock selections upstream stay copy-free
//! ([`SliceView`](super::SliceView)).

use num_traits::Float;

use flowgraph::typed::{Arena, Interface, Operator, RefViewPorts, Segment, ViewPort};

use super::op::ArrayValue;
use crate::data::array::Shape;
use crate::{Array, ArrayView, Scalar};

/// Shared runtime state: the axis config, the outer × chunk layout (sized on the
/// build call), and the output buffer.
pub struct ReshapeState<T: Scalar, const OUT: usize> {
    axis: usize,
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
    out: Array<T, OUT>,
}

/// Interleave `inputs` (each materialized row-major) into `output` along the
/// combine layout.
#[inline(always)]
fn interleaved_copy_views<T: Scalar, const IN: usize>(
    output: &mut [T],
    inputs: &[&ArrayView<T, IN>],
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let stride = n_inputs * chunk_size;
    for (input_idx, arr) in inputs.iter().enumerate() {
        let src = arr.to_contiguous();
        for outer in 0..outer_count {
            let src_offset = outer * chunk_size;
            let dst_offset = outer * stride + input_idx * chunk_size;
            output[dst_offset..dst_offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
        }
    }
}

#[inline(always)]
fn interleaved_copy_views_selective<T: Scalar, const IN: usize>(
    output: &mut [T],
    inputs: &[&ArrayView<T, IN>],
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
fn stack_extents<const IN: usize, const OUT: usize>(
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

// ---------------------------------------------------------------------------
// Stack / StackSync — new axis. (StackView/StackSyncView are aliases.)
// ---------------------------------------------------------------------------

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
    type Inputs = RefViewPorts<ArrayValue<T, IN>>;
    type Outputs = ViewPort<ArrayValue<T, OUT>>;
    type State = ReshapeState<T, OUT>;

    fn init(self) -> Self::State {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros([0; OUT]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, views): (&'a [bool], &'a [&'a ArrayView<'a, T, IN>]),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        if init {
            assert!(!views.is_empty(), "Stack requires at least one input");
            let first = views[0].extents();
            assert!(self_axis_ok(state.axis, IN, true), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = views.len();
            state.out = Array::zeros(stack_extents::<IN, OUT>(first, state.axis, views.len()));
            return (false, state.out.view());
        }
        interleaved_copy_views(
            state.out.as_mut_slice(),
            views,
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a ArrayView<'a, T, IN>]),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        (false, state.out.view())
    }
}

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
    type Inputs = RefViewPorts<ArrayValue<T, IN>>;
    type Outputs = ViewPort<ArrayValue<T, OUT>>;
    type State = ReshapeState<T, OUT>;

    fn init(self) -> Self::State {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros([0; OUT]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (flags, views): (&'a [bool], &'a [&'a ArrayView<'a, T, IN>]),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        if init {
            assert!(!views.is_empty(), "StackSync requires at least one input");
            let first = views[0].extents();
            assert!(self_axis_ok(state.axis, IN, true), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = views.len();
            let mut out = Array::zeros(stack_extents::<IN, OUT>(first, state.axis, views.len()));
            for v in out.as_mut_slice().iter_mut() {
                *v = T::nan();
            }
            state.out = out;
            return (false, state.out.view());
        }
        for v in state.out.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_views_selective(
            state.out.as_mut_slice(),
            views,
            (0..flags.len()).filter(|&i| flags[i]),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a ArrayView<'a, T, IN>]),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        (false, state.out.view())
    }
}

/// [`Stack`] over view inputs — now the same operator (the currency is already
/// views); retained for source compatibility.
pub type StackView<T, const IN: usize, const OUT: usize> = Stack<T, IN, OUT>;
/// [`StackSync`] over view inputs — now the same operator.
pub type StackSyncView<T, const IN: usize, const OUT: usize> = StackSync<T, IN, OUT>;

// ---------------------------------------------------------------------------
// Concat / ConcatSync — existing axis (rank-preserving).
// ---------------------------------------------------------------------------

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
    type Inputs = RefViewPorts<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = ReshapeState<T, N>;

    fn init(self) -> Self::State {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, views): (&'a [bool], &'a [&'a ArrayView<'a, T, N>]),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            assert!(!views.is_empty(), "Concat requires at least one input");
            let mut ext = views[0].extents();
            assert!(state.axis < N, "axis out of bounds");
            state.outer_count = ext[..state.axis].iter().product();
            state.chunk_size = ext[state.axis..].iter().product();
            state.n_inputs = views.len();
            ext[state.axis] *= views.len();
            state.out = Array::zeros(ext);
            return (false, state.out.view());
        }
        interleaved_copy_views(
            state.out.as_mut_slice(),
            views,
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a ArrayView<'a, T, N>]),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

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
    type Inputs = RefViewPorts<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = ReshapeState<T, N>;

    fn init(self) -> Self::State {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (flags, views): (&'a [bool], &'a [&'a ArrayView<'a, T, N>]),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            assert!(!views.is_empty(), "ConcatSync requires at least one input");
            let mut ext = views[0].extents();
            assert!(state.axis < N, "axis out of bounds");
            state.outer_count = ext[..state.axis].iter().product();
            state.chunk_size = ext[state.axis..].iter().product();
            state.n_inputs = views.len();
            ext[state.axis] *= views.len();
            let mut out = Array::zeros(ext);
            for v in out.as_mut_slice().iter_mut() {
                *v = T::nan();
            }
            state.out = out;
            return (false, state.out.view());
        }
        for v in state.out.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_views_selective(
            state.out.as_mut_slice(),
            views,
            (0..flags.len()).filter(|&i| flags[i]),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a ArrayView<'a, T, N>]),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

#[inline(always)]
fn self_axis_ok(axis: usize, rank: usize, allow_equal: bool) -> bool {
    if allow_equal { axis <= rank } else { axis < rank }
}

// ---------------------------------------------------------------------------
// Split — zero-copy axis-0 fan-out into view ports (the inverse of `Stack`).
// ---------------------------------------------------------------------------

/// Split a rank-`IN` array along axis 0 into `N` per-row rank-`OUT` (`= IN - 1`)
/// views — the `1 → N` inverse of [`Stack`]. The port count is declared at
/// construction (`axis_size`); the build call asserts the input's axis-0 size
/// matches.
///
/// **Zero-copy**: each output is a strided [`ArrayView`] of the input's row,
/// re-derived from the fresh input every invocation and lent through the
/// per-generation [`Arena`] (the multi-output kind needs by-reference homing) —
/// no row data is copied. All rows notify exactly when the input notifies.
///
/// Implements [`Segment`] directly: views cannot be re-lent through `&State`, so
/// every invocation rebuilds the planes and expresses the gate in the flags.
pub struct Split<T: Scalar, const IN: usize, const OUT: usize> {
    axis_size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Split<T, IN, OUT> {
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

impl<T: Scalar, const IN: usize, const OUT: usize> Segment for Split<T, IN, OUT> {
    type Inputs = ViewPort<ArrayValue<T, IN>>;
    type Outputs = RefViewPorts<ArrayValue<T, OUT>>;
    type State = SplitState;

    fn init(self) -> SplitState {
        SplitState {
            axis_size: self.axis_size,
            arena: Arena::new(),
        }
    }

    fn compute<'a, 'b: 'a>(
        (notified, x): (bool, ArrayView<'a, T, IN>),
        state: &'b mut SplitState,
        init: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let n = state.axis_size;
        let (data, base) = x.buffer();
        let shape = x.shape();
        let (ext, strd) = (shape.extents(), shape.strides());
        if init {
            assert!(IN >= 1, "Split requires IN >= 1");
            assert!(OUT == IN - 1, "Split: OUT ({OUT}) must be IN ({IN}) - 1");
            assert!(
                ext[0] == n,
                "Split: input axis-0 size {} != declared {n}",
                ext[0],
            );
        }
        // Each row drops axis 0, keeping the inner axes' extents/strides.
        let mut inner_ext = [0usize; OUT];
        let mut inner_str = [0usize; OUT];
        for d in 0..OUT {
            inner_ext[d] = ext[d + 1];
            inner_str[d] = strd[d + 1];
        }
        let row_shape = Shape::strided(inner_ext, inner_str);
        let alloc = state.arena.reset();
        let flags = alloc.slice(std::iter::repeat_n(notified && !init, n));
        let views = alloc.slice((0..n).map(|i| {
            &*alloc.alloc(ArrayView::from_parts(data, base + i * strd[0], row_shape))
        }));
        (flags, views)
    }
}
