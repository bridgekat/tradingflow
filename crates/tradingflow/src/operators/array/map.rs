use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar, array};
use crate::graph::{Operator, Segment};
use crate::ports::ArrayPort;

/// Operator signature for [`array_map`] etc.
#[allow(clippy::type_complexity)]
pub struct ArrayMap<A: Scalar, const N: usize, T: Scalar, const M: usize, I, F>
where
    I: FnOnce(ArrayView<'_, A, N>) -> Array<T, M> + Send + 'static,
    F: FnMut(&mut Array<T, M>, ArrayView<'_, A, N>) + Send + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (A, T)>,
}

impl<A: Scalar, const N: usize, T: Scalar, const M: usize, I, F> ArrayMap<A, N, T, M, I, F>
where
    I: FnOnce(ArrayView<'_, A, N>) -> Array<T, M> + Send + 'static,
    F: FnMut(&mut Array<T, M>, ArrayView<'_, A, N>) + Send + 'static,
{
    pub fn new(init: I, update: F) -> Self {
        Self {
            init,
            update,
            _marker: PhantomData,
        }
    }
}

impl<A: Scalar, const N: usize, T: Scalar, const M: usize, I, F> Operator
    for ArrayMap<A, N, T, M, I, F>
where
    I: FnOnce(ArrayView<'_, A, N>) -> Array<T, M> + Send + 'static,
    F: FnMut(&mut Array<T, M>, ArrayView<'_, A, N>) + Send + 'static,
{
    type Inputs = ArrayPort<A, N>;
    type Outputs = ArrayPort<T, M>;
    type Context = Instant;
    type State = (F, Array<T, M>);

    fn init(self, (_, a): (bool, ArrayView<'_, A, N>)) -> (F, Array<T, M>) {
        (self.update, (self.init)(a))
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, A, N>),
        (_, out): &'b mut (F, Array<T, M>),
    ) -> (bool, ArrayView<'a, T, M>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        (_, a): (bool, ArrayView<'a, A, N>),
        (update, out): &'b mut (F, Array<T, M>),
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, M>) {
        update(out, a);
        (true, out.view())
    }
}

/// Operator signature for [`array_binary_map`] etc.
#[allow(clippy::type_complexity)]
pub struct ArrayBinaryMap<
    A: Scalar,
    const L: usize,
    B: Scalar,
    const N: usize,
    T: Scalar,
    const M: usize,
    I,
    F,
> where
    I: FnOnce(ArrayView<'_, A, L>, ArrayView<'_, B, N>) -> Array<T, M> + Send + 'static,
    F: FnMut(&mut Array<T, M>, ArrayView<'_, A, L>, ArrayView<'_, B, N>) + Send + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (A, B, T)>,
}

impl<A: Scalar, const L: usize, B: Scalar, const N: usize, T: Scalar, const M: usize, I, F>
    ArrayBinaryMap<A, L, B, N, T, M, I, F>
where
    I: FnOnce(ArrayView<'_, A, L>, ArrayView<'_, B, N>) -> Array<T, M> + Send + 'static,
    F: FnMut(&mut Array<T, M>, ArrayView<'_, A, L>, ArrayView<'_, B, N>) + Send + 'static,
{
    pub fn new(init: I, update: F) -> Self {
        Self {
            init,
            update,
            _marker: PhantomData,
        }
    }
}

impl<A: Scalar, const L: usize, B: Scalar, const N: usize, T: Scalar, const M: usize, I, F> Operator
    for ArrayBinaryMap<A, L, B, N, T, M, I, F>
where
    I: FnOnce(ArrayView<'_, A, L>, ArrayView<'_, B, N>) -> Array<T, M> + Send + 'static,
    F: FnMut(&mut Array<T, M>, ArrayView<'_, A, L>, ArrayView<'_, B, N>) + Send + 'static,
{
    type Inputs = (ArrayPort<A, L>, ArrayPort<B, N>);
    type Outputs = ArrayPort<T, M>;
    type Context = Instant;
    type State = (F, Array<T, M>);

    fn init(
        self,
        ((_, a), (_, b)): ((bool, ArrayView<'_, A, L>), (bool, ArrayView<'_, B, N>)),
    ) -> (F, Array<T, M>) {
        (self.update, (self.init)(a, b))
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ArrayView<'a, A, L>), (bool, ArrayView<'a, B, N>)),
        (_, out): &'b mut (F, Array<T, M>),
    ) -> (bool, ArrayView<'a, T, M>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, ArrayView<'a, A, L>), (bool, ArrayView<'a, B, N>)),
        (update, out): &'b mut (F, Array<T, M>),
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, M>) {
        update(out, a, b);
        (true, out.view())
    }
}

/// Operator signature for [`array_ternary_map`] etc.
#[allow(clippy::type_complexity)]
pub struct ArrayTernaryMap<
    A: Scalar,
    const K: usize,
    B: Scalar,
    const L: usize,
    C: Scalar,
    const M: usize,
    T: Scalar,
    const N: usize,
    I,
    F,
> where
    I: FnOnce(ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>) -> Array<T, N>
        + Send
        + 'static,
    F: FnMut(&mut Array<T, N>, ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>)
        + Send
        + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (A, B, C, T)>,
}

impl<
    A: Scalar,
    const K: usize,
    B: Scalar,
    const L: usize,
    C: Scalar,
    const M: usize,
    T: Scalar,
    const N: usize,
    I,
    F,
> ArrayTernaryMap<A, K, B, L, C, M, T, N, I, F>
where
    I: FnOnce(ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>) -> Array<T, N>
        + Send
        + 'static,
    F: FnMut(&mut Array<T, N>, ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>)
        + Send
        + 'static,
{
    pub fn new(init: I, update: F) -> Self {
        Self {
            init,
            update,
            _marker: PhantomData,
        }
    }
}

impl<
    A: Scalar,
    const K: usize,
    B: Scalar,
    const L: usize,
    C: Scalar,
    const M: usize,
    T: Scalar,
    const N: usize,
    I,
    F,
> Operator for ArrayTernaryMap<A, K, B, L, C, M, T, N, I, F>
where
    I: FnOnce(ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>) -> Array<T, N>
        + Send
        + 'static,
    F: FnMut(&mut Array<T, N>, ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>)
        + Send
        + 'static,
{
    type Inputs = (ArrayPort<A, K>, ArrayPort<B, L>, ArrayPort<C, M>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = (F, Array<T, N>);

    fn init(
        self,
        ((_, a), (_, b), (_, c)): (
            (bool, ArrayView<'_, A, K>),
            (bool, ArrayView<'_, B, L>),
            (bool, ArrayView<'_, C, M>),
        ),
    ) -> (F, Array<T, N>) {
        (self.update, (self.init)(a, b, c))
    }

    fn passthrough<'a, 'b: 'a>(
        _: (
            (bool, ArrayView<'a, A, K>),
            (bool, ArrayView<'a, B, L>),
            (bool, ArrayView<'a, C, M>),
        ),
        (_, out): &'b mut (F, Array<T, N>),
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b), (_, c)): (
            (bool, ArrayView<'a, A, K>),
            (bool, ArrayView<'a, B, L>),
            (bool, ArrayView<'a, C, M>),
        ),
        (update, out): &'b mut (F, Array<T, N>),
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        update(out, a, b, c);
        (true, out.view())
    }
}

/// A closure applied to an array view and producing a new array.
#[allow(clippy::type_complexity)]
pub fn array_map_inplace<A: Scalar, const N: usize, T: Scalar, const M: usize>(
    init: impl FnOnce(ArrayView<'_, A, N>) -> Array<T, M> + Send + 'static,
    update: impl FnMut(&mut Array<T, M>, ArrayView<'_, A, N>) + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<A, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    ArrayMap::new(init, update)
}

/// A closure applied to an array view and producing a new array.
#[allow(clippy::type_complexity)]
pub fn array_map<A: Scalar, const N: usize, T: Scalar, const M: usize>(
    f: impl FnMut(ArrayView<'_, A, N>) -> Array<T, M> + Clone + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<A, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    let mut g = f.clone();
    ArrayMap::new(f, move |out, a| *out = g(a))
}

/// A binary closure applied to two array views and producing a new array.
#[allow(clippy::type_complexity)]
pub fn array_binary_map_inplace<
    A: Scalar,
    const L: usize,
    B: Scalar,
    const N: usize,
    T: Scalar,
    const M: usize,
>(
    init: impl FnOnce(ArrayView<'_, A, L>, ArrayView<'_, B, N>) -> Array<T, M> + Send + 'static,
    update: impl FnMut(&mut Array<T, M>, ArrayView<'_, A, L>, ArrayView<'_, B, N>) + Send + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<A, L>, ArrayPort<B, N>),
    Outputs = ArrayPort<T, M>,
    Context = Instant,
> {
    ArrayBinaryMap::new(init, update)
}

/// A binary closure applied to two array views and producing a new array.
#[allow(clippy::type_complexity)]
pub fn array_binary_map<
    A: Scalar,
    const L: usize,
    B: Scalar,
    const N: usize,
    T: Scalar,
    const M: usize,
>(
    f: impl FnMut(ArrayView<'_, A, L>, ArrayView<'_, B, N>) -> Array<T, M> + Clone + Send + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<A, L>, ArrayPort<B, N>),
    Outputs = ArrayPort<T, M>,
    Context = Instant,
> {
    let mut g = f.clone();
    ArrayBinaryMap::new(f, move |out, a, b| *out = g(a, b))
}

/// A ternary closure applied to three array views and producing a new array.
#[allow(clippy::type_complexity)]
pub fn array_ternary_map_inplace<
    A: Scalar,
    const K: usize,
    B: Scalar,
    const L: usize,
    C: Scalar,
    const M: usize,
    T: Scalar,
    const N: usize,
>(
    init: impl FnOnce(ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>) -> Array<T, N>
    + Send
    + 'static,
    update: impl FnMut(&mut Array<T, N>, ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>)
    + Send
    + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<A, K>, ArrayPort<B, L>, ArrayPort<C, M>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    ArrayTernaryMap::new(init, update)
}

/// A ternary closure applied to three array views and producing a new array.
#[allow(clippy::type_complexity)]
pub fn array_ternary_map<
    A: Scalar,
    const K: usize,
    B: Scalar,
    const L: usize,
    C: Scalar,
    const M: usize,
    T: Scalar,
    const N: usize,
>(
    f: impl FnMut(ArrayView<'_, A, K>, ArrayView<'_, B, L>, ArrayView<'_, C, M>) -> Array<T, N>
    + Clone
    + Send
    + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<A, K>, ArrayPort<B, L>, ArrayPort<C, M>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    let mut g = f.clone();
    ArrayTernaryMap::new(f, move |out, a, b, c| *out = g(a, b, c))
}

/// A closure applied elementwise: [`array::map`].
#[allow(clippy::type_complexity)]
pub fn map<A: Scalar, T: Scalar, const N: usize>(
    f: impl FnMut(&A) -> T + Clone + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<A, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, A, N>| array::map(a, f)
    };
    let update = {
        let mut f = f;
        move |out: &mut Array<T, N>, a: ArrayView<'_, A, N>| {
            array::map_into(out.data_mut(), a, &mut f);
        }
    };
    ArrayMap::new(init, update)
}

/// A binary closure applied elementwise: [`array::binary_map`].
#[allow(clippy::type_complexity)]
pub fn binary_map<A: Scalar, B: Scalar, T: Scalar, const N: usize>(
    f: impl FnMut(&A, &B) -> T + Clone + Send + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<A, N>, ArrayPort<B, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, A, N>, b: ArrayView<'_, B, N>| array::binary_map(a, b, f)
    };
    let update = {
        let mut f = f;
        move |out: &mut Array<T, N>, a: ArrayView<'_, A, N>, b: ArrayView<'_, B, N>| {
            array::binary_map_into(out.data_mut(), a, b, &mut f);
        }
    };
    ArrayBinaryMap::new(init, update)
}

/// A ternary closure applied elementwise: [`array::ternary_map`].
#[allow(clippy::type_complexity)]
pub fn ternary_map<A: Scalar, B: Scalar, C: Scalar, T: Scalar, const N: usize>(
    f: impl FnMut(&A, &B, &C) -> T + Clone + Send + 'static,
) -> impl Segment<
    Inputs = (ArrayPort<A, N>, ArrayPort<B, N>, ArrayPort<C, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    let init = {
        let f = f.clone();
        move |a: ArrayView<'_, A, N>, b: ArrayView<'_, B, N>, c: ArrayView<'_, C, N>| {
            array::ternary_map(a, b, c, f)
        }
    };
    let update = {
        let mut f = f;
        move |out: &mut Array<T, N>,
              a: ArrayView<'_, A, N>,
              b: ArrayView<'_, B, N>,
              c: ArrayView<'_, C, N>| {
            array::ternary_map_into(out.data_mut(), a, b, c, &mut f);
        }
    };
    ArrayTernaryMap::new(init, update)
}
