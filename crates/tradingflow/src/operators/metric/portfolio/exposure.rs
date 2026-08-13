use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::operators::{array, signal};
use crate::ports::{ArrayPort, SignalPort};

fn gross_exposure_into<T: Scalar + Float>(out: &mut Array<T, 0>, weights: ArrayView<'_, T, 1>) {
    let mut sum = T::zero();
    for &w in weights.iter() {
        assert!(w.is_finite(), "gross_exposure: weights must be finite");
        sum = sum + w.abs();
    }
    **out = sum;
}

fn long_exposure_into<T: Scalar + Float>(out: &mut Array<T, 0>, weights: ArrayView<'_, T, 1>) {
    let mut sum = T::zero();
    for &w in weights.iter() {
        assert!(w.is_finite(), "long_exposure: weights must be finite");
        sum = sum + w.max(T::zero());
    }
    **out = sum;
}

fn short_exposure_into<T: Scalar + Float>(out: &mut Array<T, 0>, weights: ArrayView<'_, T, 1>) {
    let mut sum = T::zero();
    for &w in weights.iter() {
        assert!(w.is_finite(), "short_exposure: weights must be finite");
        sum = sum - w.min(T::zero());
    }
    **out = sum;
}

fn net_exposure_into<T: Scalar + Float>(out: &mut Array<T, 0>, weights: ArrayView<'_, T, 1>) {
    let mut sum = T::zero();
    for &w in weights.iter() {
        assert!(w.is_finite(), "net_exposure: weights must be finite");
        sum = sum + w;
    }
    **out = sum;
}

/// The gross exposure of a portfolio: `Σ |wᵢ|`.
///
/// Inputs and outputs: see [module-level docs](super).
pub fn gross_exposure<T: Scalar + Float>()
-> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 1>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    signal::on_signal(array::array_map_inplace(
        |x: ArrayView<'_, T, 1>| {
            let mut out = Array::zeros([]);
            gross_exposure_into(&mut out, x);
            out
        },
        gross_exposure_into,
    ))
}

/// The long exposure of a portfolio: `Σ max(wᵢ, 0)`.
///
/// Inputs and outputs: see [module-level docs](super).
pub fn long_exposure<T: Scalar + Float>()
-> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 1>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    signal::on_signal(array::array_map_inplace(
        |x: ArrayView<'_, T, 1>| {
            let mut out = Array::zeros([]);
            long_exposure_into(&mut out, x);
            out
        },
        long_exposure_into,
    ))
}

/// The short exposure of a portfolio: `Σ max(-wᵢ, 0)`.
///
/// Inputs and outputs: see [module-level docs](super).
pub fn short_exposure<T: Scalar + Float>()
-> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 1>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    signal::on_signal(array::array_map_inplace(
        |x: ArrayView<'_, T, 1>| {
            let mut out = Array::zeros([]);
            short_exposure_into(&mut out, x);
            out
        },
        short_exposure_into,
    ))
}

/// The net exposure of a portfolio: `Σ wᵢ`.
///
/// Inputs and outputs: see [module-level docs](super).
pub fn net_exposure<T: Scalar + Float>()
-> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 1>), Outputs = ArrayPort<T, 0>, Context = Instant>
{
    signal::on_signal(array::array_map_inplace(
        |x: ArrayView<'_, T, 1>| {
            let mut out = Array::zeros([]);
            net_exposure_into(&mut out, x);
            out
        },
        net_exposure_into,
    ))
}
