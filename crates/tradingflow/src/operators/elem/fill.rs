use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise replacement of selected values.
pub fn fill_where<T: Scalar, const N: usize>(
    pred: impl Fn(&T) -> bool + Clone + Send + 'static,
    fill: T,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |x: &T| if pred(x) { fill.clone() } else { x.clone() })
}

/// Elementwise forward fill of selected values.
pub fn forward_fill_where<T: Scalar, const N: usize>(
    pred: impl Fn(&T) -> bool + Clone + Send + 'static,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let init = |a: ArrayView<'_, T, N>| a.to_array();
    let update = move |out: &mut Array<T, N>, a: ArrayView<'_, T, N>| {
        let out = out.data_mut();
        crate::data::array::for_each(a, |j, x| {
            if !pred(x) {
                out[j] = x.clone();
            }
        });
    };
    array::array_map_inplace(init, update)
}

/// Elementwise replacement of NaN values.
pub fn fill_nan<T: Scalar + Float, const N: usize>(
    fill: T,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    fill_where(|x: &T| x.is_nan(), fill)
}

/// Elementwise forward fill of NaN values.
pub fn forward_fill_nan<T: Scalar + Float, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    forward_fill_where(|x: &T| x.is_nan())
}
