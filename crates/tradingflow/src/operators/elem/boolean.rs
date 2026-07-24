use std::marker::PhantomData;

use crate::data::array::{Array, ArrayView};
use crate::data::{Instant, Scalar};
use crate::graph::typed::{Operator, Segment};
use crate::operators::array;
use crate::ports::ArrayPort;

/// Operator signature for [`choose`].
pub struct Choose<T: Scalar, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, const N: usize> Choose<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for Choose<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Operator for Choose<T, N> {
    type Inputs = (ArrayPort<bool, N>, ArrayPort<T, N>, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(
        self,
        ((_, cond), (_, a), (_, b)): (
            (bool, ArrayView<'_, bool, N>),
            (bool, ArrayView<'_, T, N>),
            (bool, ArrayView<'_, T, N>),
        ),
    ) -> Self::State {
        let mut out = Array::zeros(cond.extents());
        choose_into(&mut out, cond, a, b);
        out
    }

    fn passthrough<'a, 'b: 'a>(
        _: (
            (bool, ArrayView<'a, bool, N>),
            (bool, ArrayView<'a, T, N>),
            (bool, ArrayView<'a, T, N>),
        ),
        out: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        ((_, cond), (_, a), (_, b)): (
            (bool, ArrayView<'a, bool, N>),
            (bool, ArrayView<'a, T, N>),
            (bool, ArrayView<'a, T, N>),
        ),
        out: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        choose_into(out, cond, a, b);
        (true, out.view())
    }
}

fn choose_into<T: Scalar, const N: usize>(
    out: &mut Array<T, N>,
    cond: ArrayView<'_, bool, N>,
    a: ArrayView<'_, T, N>,
    b: ArrayView<'_, T, N>,
) {
    let (cs, as_, bs) = (cond.to_contiguous(), a.to_contiguous(), b.to_contiguous());
    let dst = out.data_mut();
    for i in 0..dst.len() {
        dst[i] = if cs[i] { as_[i].clone() } else { bs[i].clone() };
    }
}

/// Elementwise AND.
pub fn and<const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<bool, N>, ArrayPort<bool, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::map_broadcast(|a, b| a && b)
}

/// Elementwise OR.
pub fn or<const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<bool, N>, ArrayPort<bool, N>),
    Outputs = ArrayPort<bool, N>,
    Context = Instant,
> {
    array::map_broadcast(|a, b| a || b)
}

/// Elementwise choice: select between two arrays under a boolean mask.
pub fn choose<T: Scalar, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<bool, N>, ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    Choose::new()
}

/// Elementwise indicator: select between two constants under a boolean mask.
pub fn indicator<T: Scalar, const N: usize>(
    a: T,
    b: T,
) -> impl Segment<Inputs = ArrayPort<bool, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |cond| if cond { a.clone() } else { b.clone() })
}
