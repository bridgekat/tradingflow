use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar, array};
use crate::graph::{Operator, Segment};
use crate::ports::ArrayPort;

/// Operator signature for [`map`], [`map_array`], [`map_array_inplace`] etc.
#[allow(clippy::type_complexity)]
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

/// Operator signature for [`map_array_binary`], [`map_binary`] etc.
#[allow(clippy::type_complexity)]
pub struct MapArrayBinary<
    S: Scalar,
    const L: usize,
    T: Scalar,
    const N: usize,
    U: Scalar,
    const M: usize,
    I,
    F,
> where
    I: FnOnce(ArrayView<'_, S, L>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, S, L>, ArrayView<'_, T, N>) + Send + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (S, T, U)>,
}

impl<S: Scalar, const L: usize, T: Scalar, const N: usize, U: Scalar, const M: usize, I, F>
    MapArrayBinary<S, L, T, N, U, M, I, F>
where
    I: FnOnce(ArrayView<'_, S, L>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, S, L>, ArrayView<'_, T, N>) + Send + 'static,
{
    pub fn new(init: I, update: F) -> Self {
        Self {
            init,
            update,
            _marker: PhantomData,
        }
    }
}

impl<S: Scalar, const L: usize, T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> Operator
    for MapArrayBinary<S, L, T, N, U, M, I, F>
where
    I: FnOnce(ArrayView<'_, S, L>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, ArrayView<'_, S, L>, ArrayView<'_, T, N>) + Send + 'static,
{
    type Inputs = (ArrayPort<S, L>, ArrayPort<T, N>);
    type Outputs = ArrayPort<U, M>;
    type Context = Instant;
    type State = (F, Array<U, M>);

    fn init(
        self,
        ((_, a), (_, b)): ((bool, ArrayView<'_, S, L>), (bool, ArrayView<'_, T, N>)),
    ) -> (F, Array<U, M>) {
        (self.update, (self.init)(a, b))
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ArrayView<'a, S, L>), (bool, ArrayView<'a, T, N>)),
        (_, out): &'b mut (F, Array<U, M>),
    ) -> (bool, ArrayView<'a, U, M>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, ArrayView<'a, S, L>), (bool, ArrayView<'a, T, N>)),
        (update, out): &'b mut (F, Array<U, M>),
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, M>) {
        update(out, a, b);
        (true, out.view())
    }
}

/// A closure applied to an array view and producing a new array.
#[allow(clippy::type_complexity)]
pub fn map_array_inplace<T: Scalar, const N: usize, U: Scalar, const M: usize>(
    init: impl FnOnce(ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    update: impl FnMut(&mut Array<U, M>, ArrayView<'_, T, N>) + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<U, M>, Context = Instant> {
    MapArray::new(init, update)
}

/// A closure applied to an array view and producing a new array (reallocating).
#[allow(clippy::type_complexity)]
pub fn map_array<T: Scalar, const N: usize, U: Scalar, const M: usize>(
    f: impl FnMut(ArrayView<'_, T, N>) -> Array<U, M> + Clone + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<U, M>, Context = Instant> {
    let init = {
        let mut f = f.clone();
        move |a: ArrayView<'_, T, N>| f(a)
    };
    let update = {
        let mut f = f;
        move |out: &mut Array<U, M>, a: ArrayView<'_, T, N>| {
            *out = f(a);
        }
    };
    MapArray::new(init, update)
}

/// A binary closure applied to two array views and producing a new array.
#[allow(clippy::type_complexity)]
pub fn map_array_binary_inplace<
    S: Scalar,
    const L: usize,
    T: Scalar,
    const N: usize,
    U: Scalar,
    const M: usize,
>(
    init: impl FnOnce(ArrayView<'_, S, L>, ArrayView<'_, T, N>) -> Array<U, M> + Send + 'static,
    update: impl FnMut(&mut Array<U, M>, ArrayView<'_, S, L>, ArrayView<'_, T, N>) + Send + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<S, L>, ArrayPort<T, N>),
    Outputs = ArrayPort<U, M>,
    Context = Instant,
> {
    MapArrayBinary::new(init, update)
}

/// A binary closure applied to two array views and producing a new array
/// (reallocating).
#[allow(clippy::type_complexity)]
pub fn map_array_binary<
    S: Scalar,
    const L: usize,
    T: Scalar,
    const N: usize,
    U: Scalar,
    const M: usize,
>(
    f: impl FnMut(ArrayView<'_, S, L>, ArrayView<'_, T, N>) -> Array<U, M> + Clone + Send + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<S, L>, ArrayPort<T, N>),
    Outputs = ArrayPort<U, M>,
    Context = Instant,
> {
    let init = {
        let mut f = f.clone();
        move |a: ArrayView<'_, S, L>, b: ArrayView<'_, T, N>| f(a, b)
    };
    let update = {
        let mut f = f;
        move |out: &mut Array<U, M>, a: ArrayView<'_, S, L>, b: ArrayView<'_, T, N>| {
            *out = f(a, b);
        }
    };
    MapArrayBinary::new(init, update)
}

/// A closure applied elementwise: [`array::map`].
#[allow(clippy::type_complexity)]
pub fn map<T: Scalar, U: Scalar, const N: usize>(
    f: impl Fn(T) -> U + Clone + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<U, N>, Context = Instant> {
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, T, N>| array::map(a, f)
    };
    let update = move |out: &mut Array<U, N>, a: ArrayView<'_, T, N>| {
        array::map_into(out.data_mut(), a, &f);
    };
    MapArray::new(init, update)
}

/// A binary closure applied elementwise: [`array::map_binary`].
#[allow(clippy::type_complexity)]
pub fn map_binary<S: Scalar, T: Scalar, U: Scalar, const N: usize>(
    f: impl Fn(S, T) -> U + Clone + Send + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<S, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<U, N>,
    Context = Instant,
> {
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, S, N>, b: ArrayView<'_, T, N>| array::map_binary(a, b, &f)
    };
    let update = move |out: &mut Array<U, N>, a: ArrayView<'_, S, N>, b: ArrayView<'_, T, N>| {
        array::map_binary_into(out.data_mut(), a, b, &f);
    };
    MapArrayBinary::new(init, update)
}

/// A binary closure applied elementwise with broadcasting: [`array::map_broadcast`].
#[allow(clippy::type_complexity)]
pub fn map_broadcast<S: Scalar, T: Scalar, U: Scalar, const N: usize>(
    f: impl Fn(S, T) -> U + Clone + Send + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<S, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<U, N>,
    Context = Instant,
> {
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, S, N>, b: ArrayView<'_, T, N>| array::map_broadcast(a, b, &f)
    };
    let update = move |out: &mut Array<U, N>, a: ArrayView<'_, S, N>, b: ArrayView<'_, T, N>| {
        array::map_broadcast_into(out.data_mut(), a, b, &f);
    };
    MapArrayBinary::new(init, update)
}
