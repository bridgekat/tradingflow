use super::map::ArrayMap;
use crate::data::{self, Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::ArrayPort;

/// Reshapes an array: [`Array::reshape`].
pub fn reshape<T: Scalar, const N: usize, const M: usize>(
    extents: [usize; M],
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    let init = move |a: ArrayView<'_, T, N>| a.to_array().reshape(extents);
    let update = |out: &mut Array<T, M>, a: ArrayView<'_, T, N>| {
        let o = out.data_mut();
        data::array::for_each(a, |j, x| o[j] = x.clone());
    };
    ArrayMap::new(init, update)
}
