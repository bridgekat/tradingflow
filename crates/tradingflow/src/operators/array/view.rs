use std::marker::PhantomData;

use crate::data::layout::{IntoSliceReshapes, IntoSlices};
use crate::data::{ArrayView, Instant, Scalar};
use crate::graph::{Operator, Segment};
use crate::ports::ArrayPort;

/// Operator signature for [`derive_view`] etc.
pub struct DeriveView<T: Scalar, const N: usize, U: Scalar, const M: usize, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> ArrayView<'_, U, M> + Send + 'static,
{
    derive: F,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, F> DeriveView<T, N, U, M, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> ArrayView<'_, U, M> + Send + 'static,
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
    F: FnMut(ArrayView<'_, T, N>) -> ArrayView<'_, U, M> + Send + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<U, M>;
    type Context = Instant;
    type State = F;

    fn init(self, _: (bool, ArrayView<'_, T, N>)) -> F {
        self.derive
    }

    fn passthrough<'a, 'b: 'a>(
        (_, a): (bool, ArrayView<'a, T, N>),
        derive: &'b mut F,
    ) -> (bool, ArrayView<'a, U, M>) {
        (false, derive(a))
    }

    fn compute<'a, 'b: 'a>(
        (_, a): (bool, ArrayView<'a, T, N>),
        derive: &'b mut F,
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, M>) {
        (true, derive(a))
    }
}

/// A closure applied to an array view and producing an array view.
pub fn derive_view<T: Scalar, const N: usize, U: Scalar, const M: usize>(
    f: impl FnMut(ArrayView<'_, T, N>) -> ArrayView<'_, U, M> + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<U, M>, Context = Instant> {
    DeriveView::new(f)
}

/// Takes a slice of an array view: [`ArrayView::slice`].
pub fn slice<T: Scalar, const N: usize>(
    slices: impl IntoSlices<N>,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let slices = slices.into_slices();
    DeriveView::new(move |a| ArrayView::slice(&a, slices))
}

/// Takes a slice of an array view: [`ArrayView::slice_reshape`].
pub fn slice_reshape<T: Scalar, const N: usize, const M: usize, const K: usize>(
    slices: impl IntoSliceReshapes<K>,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    let slices = slices.into_slice_reshapes();
    DeriveView::new(move |a| ArrayView::slice_reshape(&a, slices))
}

/// Pads leading new axes to an array view: [`ArrayView::pad_ndim`].
pub fn pad_ndim<T: Scalar, const N: usize, const M: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    DeriveView::new(move |a| ArrayView::pad_ndim(&a))
}

/// Permutes the axes of an array view: [`ArrayView::transpose`].
pub fn transpose<T: Scalar, const N: usize>(
    perm: [usize; N],
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    DeriveView::new(move |a| ArrayView::transpose(&a, perm))
}
