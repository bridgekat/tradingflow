use std::marker::PhantomData;

use crate::data::layout::{IntoSliceReshapes, IntoSlices};
use crate::data::{Instant, Scalar, SeriesView};
use crate::graph::Operator;
use crate::ports::SeriesPort;

/// Operator signature for [`derive_view`] etc.
pub struct DeriveView<T: Scalar, const N: usize, U: Scalar, const M: usize, F>
where
    F: FnMut(SeriesView<'_, T, N>) -> SeriesView<'_, U, M> + Send + 'static,
{
    derive: F,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, F> DeriveView<T, N, U, M, F>
where
    F: FnMut(SeriesView<'_, T, N>) -> SeriesView<'_, U, M> + Send + 'static,
{
    pub fn new(derive: F) -> Self {
        Self {
            derive,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, F> Operator for DeriveView<T, N, U, M, F>
where
    F: FnMut(SeriesView<'_, T, N>) -> SeriesView<'_, U, M> + Send + 'static,
{
    type Inputs = SeriesPort<T, N>;
    type Outputs = SeriesPort<U, M>;
    type Context = Instant;
    type State = F;

    fn init(self, _: SeriesView<'_, T, N>) -> F {
        self.derive
    }

    fn reset<'a, 'b: 'a>(a: SeriesView<'a, T, N>, derive: &'b mut F) -> SeriesView<'a, U, M> {
        derive(a)
    }

    fn compute<'a, 'b: 'a>(
        a: SeriesView<'a, T, N>,
        derive: &'b mut F,
        _: &Instant,
    ) -> SeriesView<'a, U, M> {
        derive(a)
    }
}

/// A closure applied to a series view and producing a series view.
pub fn derive_view<T: Scalar, const N: usize, U: Scalar, const M: usize>(
    f: impl FnMut(SeriesView<'_, T, N>) -> SeriesView<'_, U, M> + Send + 'static,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<U, M>, Context = Instant> {
    DeriveView::new(f)
}

/// Takes a slice of a series view: [`SeriesView::slice`].
pub fn slice<T: Scalar, const N: usize>(
    slices: impl IntoSlices<N>,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    let slices = slices.into_slices();
    DeriveView::new(move |a| SeriesView::slice(&a, slices))
}

/// Takes a slice of a series view: [`SeriesView::slice_reshape`].
pub fn slice_reshape<T: Scalar, const N: usize, const M: usize, const K: usize>(
    slices: impl IntoSliceReshapes<K>,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, M>, Context = Instant> {
    let slices = slices.into_slice_reshapes();
    DeriveView::new(move |a| SeriesView::slice_reshape(&a, slices))
}

/// Pads leading new axes to a series view: [`SeriesView::pad_ndim`].
pub fn pad_ndim<T: Scalar, const N: usize, const M: usize>()
-> impl Operator<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, M>, Context = Instant> {
    DeriveView::new(move |a| SeriesView::pad_ndim(&a))
}

/// Swaps the axes of a series view: [`SeriesView::swap_axes`].
pub fn swap_axes<T: Scalar, const N: usize>(
    c: usize,
    d: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    DeriveView::new(move |a| SeriesView::swap_axes(&a, c, d))
}

/// Permutes the axes of a series view: [`SeriesView::permute_axes`].
pub fn permute_axes<T: Scalar, const N: usize>(
    perm: [usize; N],
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    DeriveView::new(move |a| SeriesView::permute_axes(&a, perm))
}

/// Moves one element axis of a series view: [`SeriesView::move_axis`].
pub fn move_axis<T: Scalar, const N: usize>(
    from: usize,
    to: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    DeriveView::new(move |a| SeriesView::move_axis(&a, from, to))
}
