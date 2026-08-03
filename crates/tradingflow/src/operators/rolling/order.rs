use num_traits::Float;

use super::base::{Rolling, Scanning};
use crate::data::{Instant, Retention, Scalar};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// [`quantile`] over an explicitly recorded series.
pub fn series_quantile<T: Scalar + Float, const N: usize>(
    q: f64,
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    assert!((0.0..=1.0).contains(&q), "quantile must be in [0, 1]");
    let mut scratch: Vec<T> = Vec::new();
    let f = move |pairs: &[(usize, T)], _: usize| {
        scratch.clear();
        scratch.extend(pairs.iter().map(|&(_, x)| x));
        scratch.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = scratch.len();
        let h = q * (n - 1) as f64;
        let lo = h.floor() as usize;
        let v = scratch[lo];
        if lo + 1 < n {
            v + T::from(h - lo as f64).unwrap() * (scratch[lo + 1] - v)
        } else {
            v
        }
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(1), f))
}

/// Elementwise rolling quantile (linear interpolation between order
/// statistics, like `numpy.quantile`) over a specified window, ingesting one
/// sample per signal. Non-finite values are skipped.
pub fn quantile<T: Scalar + Float, const N: usize>(
    q: f64,
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_quantile(q, window, min_count))
}

/// [`median`] over an explicitly recorded series.
pub fn series_median<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    series_quantile(0.5, window, min_count)
}

/// Elementwise rolling median over a specified window, ingesting one sample
/// per signal. Non-finite values are skipped.
pub fn median<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    quantile(0.5, window, min_count)
}

/// [`rank`] over an explicitly recorded series.
pub fn series_rank<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], len: usize| {
        let &(pos, v) = pairs.last().unwrap();
        if pos + 1 != len {
            return T::nan(); // The newest sample itself is not finite.
        }
        let (mut strict, mut weak) = (0usize, 0usize);
        for &(_, x) in pairs {
            if x < v {
                strict += 1;
            }
            if x <= v {
                weak += 1;
            }
        }
        T::from(strict + weak).unwrap() / T::from(2 * pairs.len()).unwrap()
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(1), f))
}

/// Elementwise rolling percentile rank in `[0, 1]` of the newest sample among
/// the window's finite samples (the mean of the strict-below and weak-below
/// fractions, as in `scipy` `percentileofscore(kind="mean")`), ingesting one
/// sample per signal. `NaN` when the newest sample is itself non-finite.
pub fn rank<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_rank(window, min_count))
}

/// [`mad`] over an explicitly recorded series.
pub fn series_mad<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], _: usize| {
        let n = T::from(pairs.len()).unwrap();
        let mean = pairs.iter().fold(T::zero(), |s, &(_, x)| s + x) / n;
        pairs
            .iter()
            .fold(T::zero(), |s, &(_, x)| s + (x - mean).abs())
            / n
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(1), f))
}

/// Elementwise rolling mean absolute deviation around the window mean over a
/// specified window, ingesting one sample per signal. Non-finite values are
/// skipped.
pub fn mad<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_mad(window, min_count))
}
