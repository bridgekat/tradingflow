use num_traits::Float;

use super::var::{series_var, var};
use crate::data::{Instant, Retention, Scalar};
use crate::graph::{Operator, OperatorExt};
use crate::operators::elem::sqrt;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// [`std_dev`] over an explicitly recorded series.
pub fn series_std_dev<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    series_var(window.into(), min_count).then(sqrt())
}

/// Elementwise rolling sample standard deviation (the square root of the
/// unbiased `n − 1` variance) over a specified window, ingesting one sample
/// per signal. Non-finite values are skipped.
pub fn std_dev<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    // Built on `var` rather than `buffer(..).then(series_std_dev(..))` so that
    // the variance accumulator's own (possibly zero) buffer retention applies.
    var(window, min_count).then(sqrt())
}
