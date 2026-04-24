//! Cross-sectional percentile clipping (winsorization).

use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::{Float, ToPrimitive};

use crate::Instant;
use crate::{Array, Input, InputTypes, Operator, Scalar};

/// Cross-sectional winsorization: clip each non-NaN value to the
/// `[p, 1-p]`-quantile range of the input cross-section.
///
/// Non-NaN entries are sorted; values below the p-quantile are replaced
/// by the p-quantile itself, and values above the (1-p)-quantile are
/// replaced by the (1-p)-quantile.  NaN inputs propagate to NaN outputs.
/// Identical sort / NaN logic to [`Percentile`](super::Percentile) —
/// the difference is that `Winsorize` preserves magnitudes (merely
/// clipped at the tails) rather than mapping to ranks.
///
/// With `p` fixed cross-sectionally, the clip bounds automatically
/// adapt to per-period volatility — a high-vol day winsorizes at wider
/// absolute bounds than a quiet day.  This is the Barra/AQR-style
/// per-period winsorization typically applied to factor inputs or
/// daily returns before pooled OLS; `p = 0.01` (1st / 99th percentile)
/// and `p = 0.025` (2.5 / 97.5) are common defaults.
///
/// `k = floor(p · n_valid)` defines the clip index: the lower bound
/// is `sorted[k]` and the upper bound is `sorted[n_valid - 1 - k]`.
/// `k = 0` (e.g. `p = 0`, or `p` too small for the current `n_valid`)
/// is a no-op.
pub struct Winsorize<T: Scalar + Float> {
    p: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Winsorize<T> {
    pub fn new(p: T) -> Self {
        assert!(p >= T::zero(), "Winsorize requires p >= 0");
        assert!(
            p < T::from(0.5).unwrap(),
            "Winsorize requires p < 0.5"
        );
        Self {
            p,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state: the winsorization quantile and a scratch buffer for
/// the per-tick sort of finite inputs.
pub struct WinsorizeState<T: Scalar + Float> {
    p: T,
    sort_buf: Vec<T>,
}

impl<T: Scalar + Float> Operator for Winsorize<T> {
    type State = WinsorizeState<T>;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (WinsorizeState<T>, Array<T>) {
        let n = inputs.as_slice().len();
        (
            WinsorizeState {
                p: self.p,
                sort_buf: vec![T::zero(); n],
            },
            Array::zeros(inputs.shape()),
        )
    }

    #[inline(always)]
    fn compute(
        state: &mut WinsorizeState<T>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let n = src.len();

        // Pack non-NaN values into the prefix of sort_buf, then sort.
        let mut n_valid = 0usize;
        for i in 0..n {
            if !src[i].is_nan() {
                state.sort_buf[n_valid] = src[i];
                n_valid += 1;
            }
        }
        state.sort_buf[..n_valid]
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let dst = output.as_mut_slice();
        let nan = T::nan();

        if n_valid == 0 {
            for slot in dst.iter_mut() {
                *slot = nan;
            }
            return true;
        }

        // k = floor(p · n_valid); lo = sorted[k], hi = sorted[n_valid - 1 - k].
        let p_f = state.p.to_f64().unwrap_or(0.0);
        let k = ((p_f * n_valid as f64).floor() as usize).min(n_valid - 1);
        let lo = state.sort_buf[k];
        let hi = state.sort_buf[n_valid - 1 - k];

        for i in 0..n {
            let v = src[i];
            if v.is_nan() {
                dst[i] = nan;
            } else if v < lo {
                dst[i] = lo;
            } else if v > hi {
                dst[i] = hi;
            } else {
                dst[i] = v;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winsorize_clips_tails_10pct() {
        // 10 values [0..9].  p = 0.1 → k = 1, clip to [sorted[1], sorted[8]]
        // = [1, 8].
        let vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let a = Array::from_vec(&[10], vals);
        let (mut s, mut o) = Winsorize::<f64>::new(0.1).init(&a, Instant::from_nanos(0));
        Winsorize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        assert_eq!(o.as_slice(), &[1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]);
    }

    #[test]
    fn winsorize_p_zero_is_noop() {
        let a = Array::from_vec(&[5], vec![-100.0_f64, 1.0, 2.0, 3.0, 100.0]);
        let (mut s, mut o) = Winsorize::<f64>::new(0.0).init(&a, Instant::from_nanos(0));
        Winsorize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        assert_eq!(o.as_slice(), &[-100.0, 1.0, 2.0, 3.0, 100.0]);
    }

    #[test]
    fn winsorize_nan_preserved() {
        // 3 valid values + 2 NaN.  p = 0.01 → k = 0 (no clipping), NaN passes through.
        let a = Array::from_vec(&[5], vec![f64::NAN, 10.0, 20.0, 30.0, f64::NAN]);
        let (mut s, mut o) = Winsorize::<f64>::new(0.01).init(&a, Instant::from_nanos(0));
        Winsorize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        let out = o.as_slice();
        assert!(out[0].is_nan());
        assert!(out[4].is_nan());
        assert_eq!(out[1], 10.0);
        assert_eq!(out[2], 20.0);
        assert_eq!(out[3], 30.0);
    }

    #[test]
    fn winsorize_all_nan() {
        let a = Array::from_vec(&[3], vec![f64::NAN, f64::NAN, f64::NAN]);
        let (mut s, mut o) = Winsorize::<f64>::new(0.01).init(&a, Instant::from_nanos(0));
        Winsorize::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        for &v in o.as_slice() {
            assert!(v.is_nan());
        }
    }
}
