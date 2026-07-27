use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar, array};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ArrayPorts};

/// Operator signature for [`concat`](fn@concat), [`stack`] etc.
pub struct Combine<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F>
where
    I: FnOnce(&[ArrayView<'_, T, N>]) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, &[ArrayView<'_, T, N>]) + Send + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> Combine<T, N, U, M, I, F>
where
    I: FnOnce(&[ArrayView<'_, T, N>]) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, &[ArrayView<'_, T, N>]) + Send + 'static,
{
    pub fn new(init: I, update: F) -> Self {
        Self {
            init,
            update,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> Segment
    for Combine<T, N, U, M, I, F>
where
    I: FnOnce(&[ArrayView<'_, T, N>]) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, &[ArrayView<'_, T, N>]) + Send + 'static,
{
    type Inputs = ArrayPorts<T, N>;
    type Outputs = ArrayPort<U, M>;
    type Context = Instant;
    type State = (F, Array<U, M>);

    fn init(self, views: &[ArrayView<'_, T, N>]) -> (F, Array<U, M>) {
        (self.update, (self.init)(views))
    }

    fn reset<'a, 'b: 'a>(
        _: &'a [ArrayView<'a, T, N>],
        (_, out): &'b mut (F, Array<U, M>),
    ) -> ArrayView<'a, U, M> {
        out.view()
    }

    fn compute<'a, 'b: 'a>(
        views: &'a [ArrayView<'a, T, N>],
        (update, out): &'b mut (F, Array<U, M>),
        _: &Instant,
    ) -> ArrayView<'a, U, M> {
        update(out, views);
        out.view()
    }
}

/// Concatenates the inputs along the existing axis `axis`: [`array::concat`].
pub fn concat<T: Scalar, const N: usize>(
    axis: usize,
) -> impl Segment<Inputs = ArrayPorts<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let init = move |views: &[ArrayView<'_, T, N>]| array::concat(views, axis);
    let update = move |out: &mut Array<T, N>, views: &[ArrayView<'_, T, N>]| {
        array::concat_into(out.data_mut(), views, axis);
    };
    Combine::new(init, update)
}

/// Stacks the inputs along a new axis inserted at `axis`: [`array::stack`].
pub fn stack<T: Scalar, const N: usize, const M: usize>(
    axis: usize,
) -> impl Segment<Inputs = ArrayPorts<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    assert!(
        M == N + 1,
        "stack: output ndim ({M}) must be input ndim ({N}) plus one"
    );
    let init = move |views: &[ArrayView<'_, T, N>]| array::stack(views, axis);
    let update = move |out: &mut Array<T, M>, views: &[ArrayView<'_, T, N>]| {
        array::stack_into(out.data_mut(), views, axis);
    };
    Combine::new(init, update)
}
