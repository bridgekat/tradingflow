use num_traits::Float;

use super::base::{Rolling, Scanning};
use crate::data::{Instant, Retention, Scalar};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// The OLS sufficient statistics `(mx, my, sxx, sxy, syy)` of a lane's finite
/// samples regressed against their window positions.
fn ols_sums<T: Scalar + Float>(pairs: &[(usize, T)]) -> (T, T, T, T, T) {
    let n = T::from(pairs.len()).unwrap();
    let (mut sx, mut sy) = (T::zero(), T::zero());
    for &(pos, y) in pairs {
        sx = sx + T::from(pos).unwrap();
        sy = sy + y;
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut sxx, mut sxy, mut syy) = (T::zero(), T::zero(), T::zero());
    for &(pos, y) in pairs {
        let dx = T::from(pos).unwrap() - mx;
        let dy = y - my;
        sxx = sxx + dx * dx;
        sxy = sxy + dx * dy;
        syy = syy + dy * dy;
    }
    (mx, my, sxx, sxy, syy)
}

/// [`slope`] over an explicitly recorded series.
pub fn series_slope<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], _: usize| {
        let (_, _, sxx, sxy, _) = ols_sums(pairs);
        sxy / sxx // `sxx > 0`: positions are distinct and `n >= 2`.
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(2), f))
}

/// Elementwise rolling regression slope of the samples against their window
/// positions (per tick), ingesting one sample per signal. Non-finite values
/// are skipped; needs at least 2 finite samples.
pub fn slope<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_slope(window, min_count))
}

/// [`rsquare`] over an explicitly recorded series.
pub fn series_rsquare<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], _: usize| {
        let (_, _, sxx, sxy, syy) = ols_sums(pairs);
        if syy <= T::zero() {
            return T::nan();
        }
        sxy * sxy / (sxx * syy)
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(2), f))
}

/// Elementwise rolling R² of the regression of the samples against their
/// window positions, ingesting one sample per signal. Non-finite values are
/// skipped; needs at least 2 finite samples, and constant samples yield `NaN`.
pub fn rsquare<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_rsquare(window, min_count))
}

/// [`resi`] over an explicitly recorded series.
pub fn series_resi<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let f = |pairs: &[(usize, T)], _: usize| {
        let (mx, my, sxx, sxy, _) = ols_sums(pairs);
        let slope = sxy / sxx;
        let &(pos, y) = pairs.last().unwrap();
        y - (my + slope * (T::from(pos).unwrap() - mx))
    };
    Rolling::new(window.into(), Scanning::new(min_count.max(2), f))
}

/// Elementwise rolling regression residual of the newest finite sample, from
/// the regression of the samples against their window positions, ingesting one
/// sample per signal. Non-finite values are skipped; needs at least 2 finite
/// samples.
pub fn resi<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_resi(window, min_count))
}
