use bumpalo::Bump;
use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Scalar, array};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ArrayPorts};

/// Operator signature for [`split`], [`unstack`] etc.
pub struct SplitView<T: Scalar, const N: usize, U: Scalar, const M: usize, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> Vec<ArrayView<'_, U, M>> + Send + 'static,
{
    derive: F,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, F> SplitView<T, N, U, M, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> Vec<ArrayView<'_, U, M>> + Send + 'static,
{
    pub fn new(derive: F) -> Self {
        Self {
            derive,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, F> Segment for SplitView<T, N, U, M, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> Vec<ArrayView<'_, U, M>> + Send + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPorts<U, M>;
    type Context = Instant;
    type State = (F, Bump);

    fn init(self, _: ArrayView<'_, T, N>) -> (F, Bump) {
        (self.derive, Bump::new())
    }

    fn reset<'a, 'b: 'a>(
        a: ArrayView<'a, T, N>,
        (derive, arena): &'b mut (F, Bump),
    ) -> &'a [ArrayView<'a, U, M>] {
        arena.reset();
        arena.alloc_slice_fill_iter(derive(a))
    }

    fn compute<'a, 'b: 'a>(
        a: ArrayView<'a, T, N>,
        (derive, arena): &'b mut (F, Bump),
        _: &Instant,
    ) -> &'a [ArrayView<'a, U, M>] {
        arena.reset();
        arena.alloc_slice_fill_iter(derive(a))
    }
}

/// Splits the input into consecutive chunks of the given `lengths` along the
/// existing axis `axis`: [`array::split`].
pub fn split<T: Scalar, const N: usize>(
    lengths: Vec<usize>,
    axis: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPorts<T, N>, Context = Instant> {
    SplitView::new(move |a: ArrayView<'_, T, N>| array::split(a, &lengths, axis))
}

/// Splits the input into slices along the existing axis `axis`, squeezing that
/// axis: [`array::unstack`].
pub fn unstack<T: Scalar, const N: usize, const M: usize>(
    axis: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPorts<T, M>, Context = Instant> {
    assert!(
        M + 1 == N,
        "unstack: output ndim ({M}) must be input ndim ({N}) minus one"
    );
    SplitView::new(move |a: ArrayView<'_, T, N>| array::unstack(a, axis))
}
