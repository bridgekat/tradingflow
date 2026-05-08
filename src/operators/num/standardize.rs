//! Cross-sectional z-score standardization.

use std::marker::PhantomData;

use num_traits::Float;

use crate::Instant;
use crate::{Array, Input, InputTypes, Operator, Scalar};

/// Cross-sectional z-score: subtracts the mean of non-NaN entries and
/// divides by their (population) standard deviation, in place per tick.
///
/// Same shape and NaN convention as [`Gaussianize`](super::Gaussianize)
/// and [`Percentile`](super::Percentile): NaN inputs propagate to NaN
/// outputs, the denominator is `n_valid` (not `n`), and the operator
/// requires a 1-D float input.  Unlike `Gaussianize`, this is a
/// distribution-preserving linear rescale rather than a rank-to-Gaussian
/// remap - it is the right choice when downstream code consumes
/// magnitudes (linear regression coefficients, mean-variance objective)
/// and the inputs are already approximately Gaussian, where mapping
/// through the inverse CDF would distort tail information.
///
/// Population std (`σ² = Σ(x - μ)² / n_valid`) is used so the result has
/// unit cross-sectional variance by construction.  When the input
/// cross-section has fewer than two finite values or all finite values
/// are equal (`σ = 0`), the operator emits NaN at every finite slot:
/// the z-score is undefined and emitting NaN signals "no information"
/// to downstream `is_finite` masks.
///
/// Implementation is a numerically stable two-pass: first pass
/// accumulates count and sum (for the mean), second pass accumulates
/// the sum of squared deviations (for the variance), third pass
/// writes the standardized output.  All arithmetic is performed in `T`.
#[derive(Clone)]
pub struct Standardize<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Standardize<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Standardize<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for Standardize<T> {
    type State = ();
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _timestamp: Instant) -> ((), Array<T>) {
        ((), Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        let n = src.len();
        let nan = T::nan();

        // Pass 1: count and sum over finite entries.
        let mut n_valid = 0usize;
        let mut sum = T::zero();
        for i in 0..n {
            let v = src[i];
            if !v.is_nan() {
                n_valid += 1;
                sum = sum + v;
            }
        }

        // Below 2 finite values, std is undefined or zero - emit NaN
        // everywhere.  Out-of-universe slots stay NaN regardless.
        if n_valid < 2 {
            for slot in dst.iter_mut() {
                *slot = nan;
            }
            return true;
        }

        let mean = sum / T::from(n_valid).unwrap();

        // Pass 2: sum of squared deviations from the mean.
        let mut ssd = T::zero();
        for i in 0..n {
            let v = src[i];
            if !v.is_nan() {
                let d = v - mean;
                ssd = ssd + d * d;
            }
        }
        let variance = ssd / T::from(n_valid).unwrap();
        let std = variance.sqrt();

        // Constant cross-section (all finite values equal) - z-score is
        // undefined, emit NaN.
        if std <= T::zero() {
            for slot in dst.iter_mut() {
                *slot = nan;
            }
            return true;
        }

        // Pass 3: write `(x - mean) / std` for finite slots, NaN
        // elsewhere.
        for i in 0..n {
            let v = src[i];
            if v.is_nan() {
                dst[i] = nan;
            } else {
                dst[i] = (v - mean) / std;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardize_unit_variance_zero_mean() {
        let a = Array::from_vec(&[5], vec![10.0, 20.0, 30.0, 40.0, 50.0_f64]);
        let (mut s, mut o) = Standardize::<f64>::new().init(&a, Instant::from_nanos(0));
        Standardize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        let out = o.as_slice();

        // Mean of output is zero.
        let mean: f64 = out.iter().sum::<f64>() / out.len() as f64;
        assert!(mean.abs() < 1e-12);

        // Population variance of output is one.
        let var: f64 = out.iter().map(|&x| x * x).sum::<f64>() / out.len() as f64;
        assert!((var - 1.0).abs() < 1e-12);

        // Order is preserved (monotone affine transform).
        assert!(out[0] < out[1] && out[1] < out[2] && out[2] < out[3] && out[3] < out[4]);
    }

    #[test]
    fn standardize_nan_preserved() {
        let a = Array::from_vec(&[5], vec![f64::NAN, 10.0, 20.0, 30.0, f64::NAN]);
        let (mut s, mut o) = Standardize::<f64>::new().init(&a, Instant::from_nanos(0));
        Standardize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        let out = o.as_slice();
        assert!(out[0].is_nan());
        assert!(out[4].is_nan());
        // Three valid values 10, 20, 30: mean=20, pop var=200/3, std=sqrt(200/3).
        // Standardized: (-10, 0, 10) / sqrt(200/3) = (-1.224..., 0, 1.224...).
        let expected = (200.0_f64 / 3.0).sqrt();
        assert!((out[1] - (-10.0 / expected)).abs() < 1e-12);
        assert!(out[2].abs() < 1e-12);
        assert!((out[3] - (10.0 / expected)).abs() < 1e-12);
    }

    #[test]
    fn standardize_all_nan() {
        let a = Array::from_vec(&[3], vec![f64::NAN, f64::NAN, f64::NAN]);
        let (mut s, mut o) = Standardize::<f64>::new().init(&a, Instant::from_nanos(0));
        Standardize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        for &v in o.as_slice() {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn standardize_single_valid_emits_nan() {
        let a = Array::from_vec(&[3], vec![f64::NAN, 5.0, f64::NAN]);
        let (mut s, mut o) = Standardize::<f64>::new().init(&a, Instant::from_nanos(0));
        Standardize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        for &v in o.as_slice() {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn standardize_constant_cross_section_emits_nan() {
        let a = Array::from_vec(&[4], vec![3.0, 3.0, 3.0, 3.0_f64]);
        let (mut s, mut o) = Standardize::<f64>::new().init(&a, Instant::from_nanos(0));
        Standardize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        for &v in o.as_slice() {
            assert!(v.is_nan());
        }
    }
}
