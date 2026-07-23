use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar, array};
use crate::graph::Operator;
use crate::ports::ArrayPort;

/// Operator signature for [`map`], [`map_array`], [`map_array_inplace`] etc.
pub struct MapArray<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F>
where
    I: FnOnce(ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, T, N>) + Send + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> MapArray<T, N, U, M, I, F>
where
    I: FnOnce(ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, T, N>) + Send + 'static,
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
    for MapArray<T, N, U, M, I, F>
where
    I: FnOnce(ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, T, N>) + Send + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<U, M>;
    type Context = Instant;
    type State = (F, Array<U, M>);

    fn init(self, (_, a): (bool, ArrayView<'_, T, N>)) -> (F, Array<U, M>) {
        (self.update, (self.init)(a))
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        (_, out): &'b mut (F, Array<U, M>),
    ) -> (bool, ArrayView<'a, U, M>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        (_, a): (bool, ArrayView<'a, T, N>),
        (update, out): &'b mut (F, Array<U, M>),
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, M>) {
        update(out, a);
        (true, out.view())
    }
}

/// Operator signature for [`map_array_binary`] and [`map_binary`].
pub struct MapArrayBinary<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F>
where
    I: FnOnce(ArrayView<'_, T, N>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, T, N>, ArrayView<'_, T, N>) + Send + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> MapArrayBinary<T, N, U, M, I, F>
where
    I: FnOnce(ArrayView<'_, T, N>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, T, N>, ArrayView<'_, T, N>) + Send + 'static,
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
    for MapArrayBinary<T, N, U, M, I, F>
where
    I: FnOnce(ArrayView<'_, T, N>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, T, N>, ArrayView<'_, T, N>) + Send + 'static,
{
    type Inputs = (ArrayPort<T, N>, ArrayPort<T, N>);
    type Outputs = ArrayPort<U, M>;
    type Context = Instant;
    type State = (F, Array<U, M>);

    fn init(
        self,
        ((_, a), (_, b)): ((bool, ArrayView<'_, T, N>), (bool, ArrayView<'_, T, N>)),
    ) -> (F, Array<U, M>) {
        (self.update, (self.init)(a, b))
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        (_, out): &'b mut (F, Array<U, M>),
    ) -> (bool, ArrayView<'a, U, M>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        (update, out): &'b mut (F, Array<U, M>),
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, M>) {
        update(out, a, b);
        (true, out.view())
    }
}

/// A closure applied to an array view and producing a new array.
pub fn map_array_inplace<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F>(
    init: I,
    update: F,
) -> MapArray<T, N, U, M, I, F>
where
    I: FnOnce(ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, T, N>) + Send + 'static,
{
    MapArray::new(init, update)
}

/// A closure applied to an array view and producing a new array (reallocating).
#[allow(clippy::type_complexity)]
pub fn map_array<T: Scalar, const N: usize, U: Scalar, const M: usize, F>(
    f: F,
) -> MapArray<
    T,
    N,
    U,
    M,
    impl FnOnce(ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    impl FnMut(&mut Array<U, M>, ArrayView<'_, T, N>) + Send + 'static,
>
where
    F: Fn(ArrayView<'_, T, N>) -> Array<U, M> + Clone + Send + 'static,
{
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, T, N>| f(a)
    };
    let update = move |out: &mut Array<U, M>, a: ArrayView<'_, T, N>| {
        *out = f(a);
    };
    MapArray::new(init, update)
}

/// A binary closure applied to two array views and producing a new array.
pub fn map_array_binary_inplace<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F>(
    init: I,
    update: F,
) -> MapArrayBinary<T, N, U, M, I, F>
where
    I: FnOnce(ArrayView<'_, T, N>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, T, N>, ArrayView<'_, T, N>) + Send + 'static,
{
    MapArrayBinary::new(init, update)
}

/// A binary closure applied to two array views and producing a new array
/// (reallocating).
#[allow(clippy::type_complexity)]
pub fn map_array_binary<T: Scalar, const N: usize, U: Scalar, const M: usize, F>(
    f: F,
) -> MapArrayBinary<
    T,
    N,
    U,
    M,
    impl FnOnce(ArrayView<'_, T, N>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    impl FnMut(&mut Array<U, M>, ArrayView<'_, T, N>, ArrayView<'_, T, N>) + Send + 'static,
>
where
    F: Fn(ArrayView<'_, T, N>, ArrayView<'_, T, N>) -> Array<U, M> + Clone + Send + 'static,
{
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, T, N>, b: ArrayView<'_, T, N>| f(a, b)
    };
    let update = move |out: &mut Array<U, M>, a: ArrayView<'_, T, N>, b: ArrayView<'_, T, N>| {
        *out = f(a, b);
    };
    MapArrayBinary::new(init, update)
}

/// A closure applied elementwise: [`array::map`].
#[allow(clippy::type_complexity)]
pub fn map<T: Scalar, U: Scalar, const N: usize, F>(
    f: F,
) -> MapArray<
    T,
    N,
    U,
    N,
    impl FnOnce(ArrayView<'_, T, N>) -> Array<U, N> + Send + 'static,
    impl FnMut(&mut Array<U, N>, ArrayView<'_, T, N>) + Send + 'static,
>
where
    F: Fn(T) -> U + Clone + Send + 'static,
{
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, T, N>| array::map(a, &f)
    };
    let update = move |out: &mut Array<U, N>, a: ArrayView<'_, T, N>| {
        array::map_into(out.data_mut(), a, &f);
    };
    MapArray::new(init, update)
}

/// A binary closure applied elementwise: [`array::map_binary`].
#[allow(clippy::type_complexity)]
pub fn map_binary<T: Scalar, U: Scalar, const N: usize, F>(
    f: F,
) -> MapArrayBinary<
    T,
    N,
    U,
    N,
    impl FnOnce(ArrayView<'_, T, N>, ArrayView<'_, T, N>) -> Array<U, N> + Send + 'static,
    impl FnMut(&mut Array<U, N>, ArrayView<'_, T, N>, ArrayView<'_, T, N>) + Send + 'static,
>
where
    F: Fn(T, T) -> U + Clone + Send + 'static,
{
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, T, N>, b: ArrayView<'_, T, N>| array::map_binary(a, b, &f)
    };
    let update = move |out: &mut Array<U, N>, a: ArrayView<'_, T, N>, b: ArrayView<'_, T, N>| {
        array::map_binary_into(out.data_mut(), a, b, &f);
    };
    MapArrayBinary::new(init, update)
}
