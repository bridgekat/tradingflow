//! Cross-sectional rank-to-Gaussian transform.

use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::Float;

use crate::Instant;
use crate::{Array, Input, InputTypes, Operator, Scalar};

/// Applies a cross-sectional rank-to-Gaussian transform to a 1-D array.
///
/// Non-NaN elements are ranked ascending and mapped through the inverse
/// standard-normal CDF: rank `r` of `n_valid` non-NaN elements becomes
/// `Φ⁻¹((r + 0.5) / n_valid)`.  NaN inputs propagate to NaN outputs,
/// so they do not skew the regression that typically consumes the
/// result.
///
/// Output has the same dtype and shape as the input.  The inverse CDF
/// is computed via Acklam's rational approximation (max rel error
/// ~1.15e-9), always in `f64` and cast back to `T` at the end.
pub struct Gaussianize<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Gaussianize<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Gaussianize<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for Gaussianize<T> {
    type State = Vec<usize>;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (Vec<usize>, Array<T>) {
        let n = inputs.as_slice().len();
        (vec![0; n], Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut Vec<usize>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let n = src.len();

        // Pack non-NaN indices into the prefix of `state`.
        let mut n_valid = 0usize;
        for i in 0..n {
            if !src[i].is_nan() {
                state[n_valid] = i;
                n_valid += 1;
            }
        }
        state[..n_valid].sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

        let dst = output.as_mut_slice();
        let nan = T::nan();
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        if n_valid > 0 {
            let denom = n_valid as f64;
            for rank in 0..n_valid {
                let p = (rank as f64 + 0.5) / denom;
                let z = norm_inv(p);
                dst[state[rank]] = T::from(z).unwrap_or(nan);
            }
        }
        true
    }
}

/// Inverse standard-normal CDF via Acklam's rational approximation.
///
/// Valid for `p ∈ (0, 1)`; callers must guarantee `p` stays strictly
/// inside this open interval (the operator above only ever produces
/// `p = (r + 0.5) / n_valid`, so this is satisfied).
#[inline]
fn norm_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const PLOW: f64 = 0.02425;
    const PHIGH: f64 = 1.0 - PLOW;

    if p < PLOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= PHIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_inv_reference_points() {
        // Known quantiles of the standard normal.
        assert!((norm_inv(0.5) - 0.0).abs() < 1e-9);
        assert!((norm_inv(0.975) - 1.959963984540054).abs() < 1e-7);
        assert!((norm_inv(0.025) + 1.959963984540054).abs() < 1e-7);
        assert!((norm_inv(0.01) + 2.326347874040841).abs() < 1e-6);
    }

    #[test]
    fn gaussianize_five() {
        let a = Array::from_vec(&[5], vec![30.0, 10.0, 50.0, 20.0, 40.0_f64]);
        let (mut s, mut o) = Gaussianize::<f64>::new().init(&a, Instant::from_nanos(0));
        Gaussianize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        let out = o.as_slice();
        // Order: 10(1) < 20(3) < 30(0) < 40(4) < 50(2).
        // 30 is the middle rank → Φ⁻¹(0.5) = 0.
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!(out[1] < out[3] && out[3] < out[0] && out[0] < out[4] && out[4] < out[2]);
        // Ranks {0..4} are symmetric around 2, so outputs sum to 0.
        assert!(out.iter().sum::<f64>().abs() < 1e-9);
    }

    #[test]
    fn gaussianize_nan_preserved() {
        let a = Array::from_vec(&[5], vec![f64::NAN, 10.0, 20.0, 30.0, f64::NAN]);
        let (mut s, mut o) = Gaussianize::<f64>::new().init(&a, Instant::from_nanos(0));
        Gaussianize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        let out = o.as_slice();
        assert!(out[0].is_nan());
        assert!(out[4].is_nan());
        // Three valid: 10 < 20 < 30. 20 is middle → 0.
        assert!((out[2] - 0.0).abs() < 1e-9);
        assert!(out[1] < out[2] && out[2] < out[3]);
    }

    #[test]
    fn gaussianize_all_nan() {
        let a = Array::from_vec(&[3], vec![f64::NAN, f64::NAN, f64::NAN]);
        let (mut s, mut o) = Gaussianize::<f64>::new().init(&a, Instant::from_nanos(0));
        Gaussianize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        for &v in o.as_slice() {
            assert!(v.is_nan());
        }
    }
}
