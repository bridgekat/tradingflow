use num_traits::Float;

use super::var::series_var;
use crate::data::{Instant, Retention, Scalar};
use crate::graph::{Segment, SegmentExt};
use crate::operators::{elem::sqrt, series::buffer};
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// [`std_dev`] over an explicitly recorded series.
pub fn series_std_dev<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Segment<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    series_var(window.into(), min_count).then(sqrt())
}

/// Elementwise rolling standard deviation over a specified window, ingesting
/// one sample per signal. Non-finite values are skipped.
pub fn std_dev<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Segment<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_std_dev(window, min_count))
}
