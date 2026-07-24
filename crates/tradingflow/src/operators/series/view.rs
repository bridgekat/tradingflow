use std::marker::PhantomData;

use crate::data::layout::{IntoSliceReshapes, IntoSlices};
use crate::data::{Instant, Scalar, SeriesView};
use crate::graph::{Operator, Segment};
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

    fn init(self, _: (bool, SeriesView<'_, T, N>)) -> F {
        self.derive
    }

    fn passthrough<'a, 'b: 'a>(
        (_, a): (bool, SeriesView<'a, T, N>),
        derive: &'b mut F,
    ) -> (bool, SeriesView<'a, U, M>) {
        (false, derive(a))
    }

    fn compute<'a, 'b: 'a>(
        (_, a): (bool, SeriesView<'a, T, N>),
        derive: &'b mut F,
        _: &Instant,
    ) -> (bool, SeriesView<'a, U, M>) {
        (true, derive(a))
    }
}

/// A closure applied to a series view and producing a series view.
pub fn derive_view<T: Scalar, const N: usize, U: Scalar, const M: usize>(
    f: impl FnMut(SeriesView<'_, T, N>) -> SeriesView<'_, U, M> + Send + 'static,
) -> impl Segment<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<U, M>, Context = Instant> {
    DeriveView::new(f)
}

/// Takes a slice of a series view: [`SeriesView::slice`].
pub fn slice<T: Scalar, const N: usize>(
    slices: impl IntoSlices<N>,
) -> impl Segment<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    let slices = slices.into_slices();
    DeriveView::new(move |a| SeriesView::slice(&a, slices))
}

/// Takes a slice of a series view: [`SeriesView::slice_reshape`].
pub fn slice_reshape<T: Scalar, const N: usize, const M: usize>(
    slices: impl IntoSliceReshapes<N>,
) -> impl Segment<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, M>, Context = Instant> {
    let slices = slices.into_slice_reshapes();
    DeriveView::new(move |a| SeriesView::slice_reshape(&a, slices))
}

/// Pads leading new axes to a series view: [`SeriesView::pad_ndim`].
pub fn pad_ndim<T: Scalar, const N: usize>()
-> impl Segment<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    DeriveView::new(move |a| SeriesView::pad_ndim(&a))
}

/// Permutes the axes of a series view: [`SeriesView::transpose`].
pub fn transpose<T: Scalar, const N: usize>(
    perm: [usize; N],
) -> impl Segment<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    DeriveView::new(move |a| SeriesView::transpose(&a, perm))
}
