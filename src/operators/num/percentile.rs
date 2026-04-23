//! Cross-sectional rank-to-percentile transform.

use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::Float;

use crate::Instant;
use crate::{Array, Input, InputTypes, Operator, Scalar};

/// Applies a cross-sectional rank-to-percentile transform to a 1-D array.
///
/// Non-NaN elements are ranked ascending and mapped to
/// `(rank + 0.5) / n_valid ∈ (0, 1)`.  NaN inputs propagate to NaN
/// outputs — they do not occupy ranks, so the denominator only counts
/// finite values and the percentile distribution is not squeezed by
/// missing entries.
///
/// Identical sort / NaN logic to [`Gaussianize`](super::Gaussianize),
/// which additionally applies `Φ⁻¹` to the percentile.
///
/// Output has the same dtype and shape as the input.
pub struct Percentile<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Percentile<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Percentile<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for Percentile<T> {
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
            let denom = T::from(n_valid as f64).unwrap_or(nan);
            let half = T::from(0.5).unwrap_or(nan);
            for rank in 0..n_valid {
                let p = (T::from(rank as f64).unwrap_or(nan) + half) / denom;
                dst[state[rank]] = p;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_five() {
        let a = Array::from_vec(&[5], vec![30.0, 10.0, 50.0, 20.0, 40.0_f64]);
        let (mut s, mut o) = Percentile::<f64>::new().init(&a, Instant::from_nanos(0));
        Percentile::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        let out = o.as_slice();
        // Order: 10(1) < 20(3) < 30(0) < 40(4) < 50(2).  Percentiles are
        // (0.5, 1.5, 2.5, 3.5, 4.5) / 5 = 0.1, 0.3, 0.5, 0.7, 0.9.
        assert!((out[1] - 0.1).abs() < 1e-12);
        assert!((out[3] - 0.3).abs() < 1e-12);
        assert!((out[0] - 0.5).abs() < 1e-12);
        assert!((out[4] - 0.7).abs() < 1e-12);
        assert!((out[2] - 0.9).abs() < 1e-12);
    }

    #[test]
    fn percentile_nan_preserved() {
        let a = Array::from_vec(&[5], vec![f64::NAN, 10.0, 20.0, 30.0, f64::NAN]);
        let (mut s, mut o) = Percentile::<f64>::new().init(&a, Instant::from_nanos(0));
        Percentile::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        let out = o.as_slice();
        assert!(out[0].is_nan());
        assert!(out[4].is_nan());
        // Three valid: 10 < 20 < 30 → percentiles 1/6, 3/6, 5/6.
        assert!((out[1] - 1.0 / 6.0).abs() < 1e-12);
        assert!((out[2] - 0.5).abs() < 1e-12);
        assert!((out[3] - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_all_nan() {
        let a = Array::from_vec(&[3], vec![f64::NAN, f64::NAN, f64::NAN]);
        let (mut s, mut o) = Percentile::<f64>::new().init(&a, Instant::from_nanos(0));
        Percentile::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        for &v in o.as_slice() {
            assert!(v.is_nan());
        }
    }
}
