use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::{Operator, OperatorExt, cb::Arr};
use crate::operators::{array, signal};
use crate::ports::{ArrayPort, SignalPort};

fn period_return_into<T: Scalar + Float>(
    out: &mut Array<T, 0>,
    weights: ArrayView<'_, T, 1>,
    returns: ArrayView<'_, T, 1>,
) {
    assert_eq!(
        weights.extents(),
        returns.extents(),
        "period_return: input extents mismatch"
    );
    let mut sum = T::zero();
    for (&w, &r) in weights.iter().zip(returns.iter()) {
        assert!(w.is_finite(), "period_return: weights must be finite");
        if r.is_finite() {
            sum = sum + w * r;
        }
    }
    **out = sum;
}

/// The realized return of a weight vector over one period: `Σ wᵢ rᵢ` over the
/// elements where `rᵢ` is finite, so holding an asset with no available return
/// data for this period contributes zero.
///
/// # Inputs
///
/// - `signal`: a scalar signal indicating a single period of interest.
/// - `weights`: a one-dimensional array of portfolio weights, all finite.
/// - `returns`: a one-dimensional array of per-asset returns realized over
///   the period when `weights` of assets are held, or `NaN` if unobserved.
///
/// # Outputs
///
/// - `metric`: the performance metric computed from the input weights,
///   updated each period at `signal == true`.
pub fn period_return<T: Scalar + Float>() -> impl Operator<
    Inputs = (SignalPort<0>, ArrayPort<T, 1>, ArrayPort<T, 1>),
    Outputs = ArrayPort<T, 0>,
    Context = Instant,
> {
    Arr::new(|(signal, weights, returns), _| (signal, (weights, returns))).then(signal::on_signal(
        array::array_binary_map_inplace(
            |w: ArrayView<'_, T, 1>, r: ArrayView<'_, T, 1>| {
                let mut out = Array::zeros([]);
                period_return_into(&mut out, w, r);
                out
            },
            period_return_into,
        ),
    ))
}
