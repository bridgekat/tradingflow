//! [`Choose`] — the three-input element-wise selector.

use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Element-wise `if cond[i] { a[i] } else { b[i] }` — the three-input selector.
pub struct Choose<T: Scalar, const N: usize> {
    _p: PhantomData<T>,
}

impl<T: Scalar, const N: usize> Choose<T, N> {
    pub fn new() -> Self {
        Self { _p: PhantomData }
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
}

/// The per-call body of [`Choose`]: select element-wise between `a` and `b`
/// under the mask `cond` into `out`.
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

/// Element-wise `if cond[i] { a[i] } else { b[i] }` — the three-input selector.
pub fn choose<T: Scalar, const N: usize>() -> Choose<T, N> {
    Choose::new()
}
