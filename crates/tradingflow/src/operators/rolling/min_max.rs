use num_traits::Float;

use super::base::{Rolling, Scanning};
use crate::data::{Instant, Retention, Scalar};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// [`max`] over an explicitly recorded series.
pub fn series_max<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], _: usize| {
        let mut best = pairs[0].1;
        for &(_, x) in &pairs[1..] {
            if x > best {
                best = x;
            }
        }
        best
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(1), f))
}

/// Elementwise rolling maximum over a specified window, ingesting one sample
/// per signal. Non-finite values are skipped.
pub fn max<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_max(window, min_count))
}

/// [`min`] over an explicitly recorded series.
pub fn series_min<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], _: usize| {
        let mut best = pairs[0].1;
        for &(_, x) in &pairs[1..] {
            if x < best {
                best = x;
            }
        }
        best
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(1), f))
}

/// Elementwise rolling minimum over a specified window, ingesting one sample
/// per signal. Non-finite values are skipped.
pub fn min<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_min(window, min_count))
}

/// [`idx_max`] over an explicitly recorded series.
pub fn series_idx_max<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], _: usize| {
        let mut best = pairs[0];
        for &p in &pairs[1..] {
            if p.1 > best.1 {
                best = p;
            }
        }
        T::from(best.0 + 1).unwrap()
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(1), f))
}

/// Elementwise 1-based window position (oldest = 1) of the rolling maximum,
/// ingesting one sample per signal; the first occurrence wins ties. Positions
/// count every window row, including skipped non-finite ones.
pub fn idx_max<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_idx_max(window, min_count))
}

/// [`idx_min`] over an explicitly recorded series.
pub fn series_idx_min<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], _: usize| {
        let mut best = pairs[0];
        for &p in &pairs[1..] {
            if p.1 < best.1 {
                best = p;
            }
        }
        T::from(best.0 + 1).unwrap()
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(1), f))
}

/// Elementwise 1-based window position (oldest = 1) of the rolling minimum,
/// ingesting one sample per signal; the first occurrence wins ties. Positions
/// count every window row, including skipped non-finite ones.
pub fn idx_min<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_idx_min(window, min_count))
}
