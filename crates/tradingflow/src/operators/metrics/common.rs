//! Small helpers shared by several metric operators. Private: nothing here is
//! re-exported from the parent module.

use num_traits::Float;

use crate::data::{ArrayView, Scalar};

/// The leading element of a clock-gated data view, or `None` when the clock did
/// not tick or the value is NaN — the two cases in which a scalar metric holds
/// its previous output and reports `notify = false`.
pub(super) fn ticked_value<T: Scalar + Float, const N: usize>(
    produced_clock: bool,
    data: ArrayView<'_, T, N>,
) -> Option<T> {
    if !produced_clock {
        return None;
    }
    let current = data.to_contiguous()[0];
    (!current.is_nan()).then_some(current)
}

/// Population mean and variance of `count` samples, recovered from their
/// running sum and running sum of squares.
pub(super) fn mean_var<T: Scalar + Float>(sum: T, sum_sq: T, count: usize) -> (T, T) {
    let n = T::from(count).unwrap();
    let mean = sum / n;
    (mean, sum_sq / n - mean * mean)
}
