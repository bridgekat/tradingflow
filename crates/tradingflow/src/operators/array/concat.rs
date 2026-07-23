use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar, array};
use crate::graph::Operator;
use crate::ports::{ArrayPort, ArrayPorts};

/// Operator signature for [`concat`](self::concat), [`stack`] etc.
pub struct CombineArray<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F>
where
    I: FnOnce(&[ArrayView<'_, T, N>]) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, &[ArrayView<'_, T, N>]) + Send + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> CombineArray<T, N, U, M, I, F>
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

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> Operator
    for CombineArray<T, N, U, M, I, F>
where
    I: FnOnce(&[ArrayView<'_, T, N>]) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, &[ArrayView<'_, T, N>]) + Send + 'static,
{
    type Inputs = ArrayPorts<T, N>;
    type Outputs = ArrayPort<U, M>;
    type Context = Instant;
    type State = (F, Array<U, M>);

    fn init(self, (_, views): (&[bool], &[ArrayView<'_, T, N>])) -> (F, Array<U, M>) {
        (self.update, (self.init)(views))
    }

    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [ArrayView<'a, T, N>]),
        (_, out): &'b mut (F, Array<U, M>),
    ) -> (bool, ArrayView<'a, U, M>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        (_, views): (&'a [bool], &'a [ArrayView<'a, T, N>]),
        (update, out): &'b mut (F, Array<U, M>),
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, M>) {
        update(out, views);
        (true, out.view())
    }
}

/// Concatenates the inputs along the existing axis `axis`: [`array::concat`].
#[allow(clippy::type_complexity)]
pub fn concat<T: Scalar, const N: usize>(
    axis: usize,
) -> CombineArray<
    T,
    N,
    T,
    N,
    impl FnOnce(&[ArrayView<'_, T, N>]) -> Array<T, N> + Send + 'static,
    impl FnMut(&mut Array<T, N>, &[ArrayView<'_, T, N>]) + Send + 'static,
> {
    let init = move |views: &[ArrayView<'_, T, N>]| array::concat(views, axis);
    let update = move |out: &mut Array<T, N>, views: &[ArrayView<'_, T, N>]| {
        array::concat_into(out.data_mut(), views, axis);
    };
    CombineArray::new(init, update)
}

/// Stacks the inputs along a new axis inserted at `axis`: [`array::stack`].
#[allow(clippy::type_complexity)]
pub fn stack<T: Scalar, const N: usize, const M: usize>(
    axis: usize,
) -> CombineArray<
    T,
    N,
    T,
    M,
    impl FnOnce(&[ArrayView<'_, T, N>]) -> Array<T, M> + Send + 'static,
    impl FnMut(&mut Array<T, M>, &[ArrayView<'_, T, N>]) + Send + 'static,
> {
    assert!(
        M == N + 1,
        "stack: output ndim ({M}) must be input ndim ({N}) plus one"
    );
    let init = move |views: &[ArrayView<'_, T, N>]| array::stack(views, axis);
    let update = move |out: &mut Array<T, M>, views: &[ArrayView<'_, T, N>]| {
        array::stack_into(out.data_mut(), views, axis);
    };
    CombineArray::new(init, update)
}
