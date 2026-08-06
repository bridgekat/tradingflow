use super::map::ArrayMap;
use crate::data::{self, Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

/// A closure folding the trailing `M` axes of the input into one accumulator
/// each, producing a rank-`K` array over the leading axes:
/// [`data::array::inner_reduce`].
#[allow(clippy::type_complexity)]
pub fn inner_reduce<A: Scalar, const N: usize, T: Scalar, const K: usize, const M: usize>(
    initial: T,
    f: impl FnMut(&mut T, &A) + Clone + Send + 'static,
) -> impl Operator<Inputs = ArrayPort<A, N>, Outputs = ArrayPort<T, K>, Context = Instant> {
    let init = {
        let (initial, f) = (initial.clone(), f.clone());
        move |a: ArrayView<'_, A, N>| data::array::inner_reduce::<A, T, N, K, M>(a, initial, f)
    };
    let update = {
        let mut f = f;
        move |out: &mut Array<T, K>, a: ArrayView<'_, A, N>| {
            out.data_mut().fill(initial.clone());
            data::array::inner_reduce_into::<A, T, N, K, M>(out, a, &mut f);
        }
    };
    ArrayMap::new(init, update)
}

/// A closure folding the leading `K` axes of the input into the accumulator at
/// the matching position, producing a rank-`M` array over the trailing axes:
/// [`data::array::outer_reduce`].
#[allow(clippy::type_complexity)]
pub fn outer_reduce<A: Scalar, const N: usize, T: Scalar, const K: usize, const M: usize>(
    initial: T,
    f: impl FnMut(&mut T, &A) + Clone + Send + 'static,
) -> impl Operator<Inputs = ArrayPort<A, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    let init = {
        let (initial, f) = (initial.clone(), f.clone());
        move |a: ArrayView<'_, A, N>| data::array::outer_reduce::<A, T, N, K, M>(a, initial, f)
    };
    let update = {
        let mut f = f;
        move |out: &mut Array<T, M>, a: ArrayView<'_, A, N>| {
            out.data_mut().fill(initial.clone());
            data::array::outer_reduce_into::<A, T, N, K, M>(out, a, &mut f);
        }
    };
    ArrayMap::new(init, update)
}
