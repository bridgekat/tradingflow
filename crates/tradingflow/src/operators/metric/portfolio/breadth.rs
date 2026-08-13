use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::operators::{array, signal};
use crate::ports::{ArrayPort, SignalPort};

fn breadth_into<T: Scalar + Float>(out: &mut Array<T, 0>, weights: ArrayView<'_, T, 1>) {
    let mut sum_sq = T::zero();
    for &w in weights.iter() {
        assert!(w.is_finite(), "breadth: weights must be finite");
        sum_sq = sum_sq + w * w;
    }
    **out = match sum_sq > T::zero() {
        true => sum_sq.recip(),
        false => T::nan(),
    };
}

/// The effective number of holdings of a weight vector: `1 / Σ wᵢ²`.
/// For a fully-invested portfolio (i.e. `Σ wᵢ = 1`) it takes value in `[1, n]`.
/// Produces `NaN` if all weights are zero.
///
/// Inputs and outputs: see [module-level docs](super).
pub fn breadth<T: Scalar + Float>()
-> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 1>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    signal::on_signal(array::array_map_inplace(
        |x: ArrayView<'_, T, 1>| {
            let mut out = Array::zeros([]);
            breadth_into(&mut out, x);
            out
        },
        breadth_into,
    ))
}
