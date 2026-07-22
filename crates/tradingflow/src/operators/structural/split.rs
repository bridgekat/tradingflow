//! `Split` — zero-copy axis-0 fan-out into view ports (the inverse of `Stack`).

use bumpalo::Bump;

use crate::data::layout::Strided;
use crate::data::{ArrayView, Instant, Layout, Scalar};
use crate::graph::typed::{Interface, Operator};
use crate::ports::{ArrayPort, ArrayPorts};

/// Split a rank-`IN` array along axis 0 into `N` per-row rank-`OUT` (`= IN - 1`)
/// views — the `1 → N` inverse of [`Stack`](super::Stack). The port count is
/// declared at construction (`axis_size`); the build call asserts the input's
/// axis-0 size matches.
///
/// **Zero-copy**: each output is a strided [`ArrayView`] of the input's row,
/// re-derived from the fresh input every invocation, by value; only the
/// per-generation notify/view *planes* live in the [`Bump`]
/// arena — no row data is copied. All rows notify exactly when the input
/// notifies, and each row handle is an ordinary [`ArrayPort`] producer.
///
/// An [`Operator`] whose two branches both rebuild the planes (borrowing the
/// fresh input, so nothing is re-lent through state): [`compute`](Operator::compute)
/// on a notified input (rows notify), [`passthrough`](Operator::passthrough) on a
/// silent one (rows don't). Both reset the arena first — possible because
/// `passthrough` takes `&mut State`.
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
    arena: Bump,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Operator for Split<T, IN, OUT> {
    type Inputs = ArrayPort<T, IN>;
    type Outputs = ArrayPorts<T, OUT>;
    type Context = Instant;
    type State = SplitState;

    fn init(self, (_, x): (bool, ArrayView<'_, T, IN>)) -> SplitState {
        assert!(IN >= 1, "Split requires IN >= 1");
        assert!(OUT == IN - 1, "Split: OUT ({OUT}) must be IN ({IN}) - 1");
        assert!(
            x.extents()[0] == self.axis_size,
            "Split: input axis-0 size {} != declared {}",
            x.extents()[0],
            self.axis_size,
        );
        SplitState {
            axis_size: self.axis_size,
            arena: Bump::new(),
        }
    }

    fn passthrough<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, IN>),
        state: &'b mut SplitState,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        // Silent input (also the build call): rebuild the row planes over the
        // carried input, none notifying.
        state.arena.reset();
        split_rows::<T, IN, OUT>(x, state.axis_size, &state.arena, false)
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, IN>),
        state: &'b mut SplitState,
        _: &Instant,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        // Reached only when the input notified, so every row notifies.
        state.arena.reset();
        split_rows::<T, IN, OUT>(x, state.axis_size, &state.arena, true)
    }
}

/// Build the per-row notify/view planes for [`Split`] in `arena`.
fn split_rows<'a, T: Scalar, const IN: usize, const OUT: usize>(
    x: ArrayView<'a, T, IN>,
    n: usize,
    arena: &'a Bump,
    notify: bool,
) -> (&'a [bool], &'a [ArrayView<'a, T, OUT>]) {
    let data = x.data();
    let layout = x.layout();
    let (ext, strd) = (layout.extents(), layout.strides());
    // Each row drops axis 0, keeping the inner axes' extents/strides.
    let mut inner_ext = [0usize; OUT];
    let mut inner_str = [0usize; OUT];
    inner_ext.copy_from_slice(&ext[1..]);
    inner_str.copy_from_slice(&strd[1..]);
    let row_shape = Strided::new(inner_ext, inner_str);
    let flags = arena.alloc_slice_fill_iter(std::iter::repeat_n(notify, n));
    let views = arena.alloc_slice_fill_iter(
        (0..n).map(|i| ArrayView::from_parts(row_shape, &data[i * strd[0]..])),
    );
    (flags, views)
}

/// Split a rank-`IN` array into `axis_size` rank-`OUT` by-value view rows.
pub fn split<T: Scalar, const IN: usize, const OUT: usize>(axis_size: usize) -> Split<T, IN, OUT> {
    Split::new(axis_size)
}
