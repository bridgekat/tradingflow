//! Exponential moving average operator with window-normalized weights.
//!
//! O(1) per element per tick. Weight sum is computed analytically (not
//! tracked incrementally), and eviction uses a precomputed decay factor.

use num_traits::Float;

use crate::{Array, Notify, Operator, Scalar, Series};

/// Exponential moving average.
///
/// For each element position, maintains a running weighted average:
/// ```text
/// ema_t = sum(w_i * x_{t-i}, i=0..window-1) / sum(w_i, i=0..window-1)
/// ```
/// where `w_i = alpha * (1 - alpha)^i`.
///
/// NaN handling: if any value in the window is NaN for an element, the
/// output for that element is NaN (same as the other rolling operators).
pub struct Ema<T: Scalar + Float> {
    alpha: T,
    window: usize,
}

impl<T: Scalar + Float> Ema<T> {
    /// Create with explicit smoothing factor and window size.
    pub fn new(alpha: T, window: usize) -> Self {
        assert!(
            alpha > T::zero() && alpha <= T::one(),
            "alpha must be in (0, 1]"
        );
        assert!(window >= 1, "window must be >= 1");
        Self { alpha, window }
    }

    /// Create from span (like pandas `ewm(span=N)`).
    /// `alpha = 2 / (span + 1)`.
    pub fn with_span(span: usize, window: usize) -> Self {
        assert!(span >= 1, "span must be >= 1");
        let alpha = T::from(2.0).unwrap() / T::from(span + 1).unwrap();
        Self::new(alpha, window)
    }

    /// Create from half-life.
    /// `alpha = 1 - exp(-ln(2) / half_life)`.
    pub fn with_half_life(half_life: T, window: usize) -> Self {
        assert!(half_life > T::zero(), "half_life must be > 0");
        let alpha = T::one() - (-T::from(2.0).unwrap().ln() / half_life).exp();
        Self::new(alpha, window)
    }
}

/// Runtime state for EMA.
///
/// `weight_sum` is not stored — it equals `1 - (1-α)^min(len, window)`
/// and is computed each tick via `fill_decay`.
pub struct EmaState<T: Scalar + Float> {
    alpha: T,
    one_minus_alpha: T,
    /// `(1 - alpha)^window`, precomputed for O(1) eviction.
    decay_factor: T,
    window: usize,
    /// Per-element weighted sum of values in the window.
    weighted_sum: Vec<T>,
    /// Count of NaN values in the window, per element position.
    nan_count: Vec<u32>,
    /// Tracks `(1-α)^len` during fill-up for computing weight_sum.
    fill_decay: T,
}

impl<T: Scalar + Float> Operator for Ema<T> {
    type State = EmaState<T>;
    type Inputs = (Series<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (EmaState<T>, Array<T>) {
        let stride = inputs.0.stride();
        let one_minus_alpha = T::one() - self.alpha;
        let mut decay_factor = T::one();
        for _ in 0..self.window {
            decay_factor = decay_factor * one_minus_alpha;
        }
        let state = EmaState {
            alpha: self.alpha,
            one_minus_alpha,
            decay_factor,
            window: self.window,
            weighted_sum: vec![T::zero(); stride],
            nan_count: vec![0; stride],
            fill_decay: T::one(),
        };
        let shape = inputs.0.shape();
        let stride = shape.iter().product::<usize>();
        (state, Array::from_vec(shape, vec![T::nan(); stride]))
    }

    fn compute(
        state: &mut EmaState<T>,
        inputs: (&Series<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        let series = inputs.0;
        let len = series.len();
        let row = series.at(len - 1);
        let stride = row.len();
        let alpha = state.alpha;
        let one_minus_alpha = state.one_minus_alpha;

        // Compute weight_sum analytically: 1 - (1-α)^min(len, window).
        state.fill_decay = state.fill_decay * one_minus_alpha;
        let weight_sum = T::one()
            - if len >= state.window {
                state.decay_factor
            } else {
                state.fill_decay
            };

        for i in 0..stride {
            let x = row[i];

            // 1. Decay existing accumulator.
            state.weighted_sum[i] = state.weighted_sum[i] * one_minus_alpha;

            // 2. Add new value (NaN contributes zero weight).
            if x.is_nan() {
                state.nan_count[i] += 1;
            } else {
                state.weighted_sum[i] = state.weighted_sum[i] + alpha * x;
            }

            // 3. Evict oldest value if window is full.
            if len > state.window {
                let x_old = series.at(len - 1 - state.window)[i];
                if x_old.is_nan() {
                    state.nan_count[i] -= 1;
                } else {
                    let evict_weight = alpha * state.decay_factor;
                    state.weighted_sum[i] = state.weighted_sum[i] - evict_weight * x_old;
                }
            }
        }

        // Produce output only when window is full.
        if len < state.window {
            false
        } else {
            let out = output.as_mut_slice();
            for i in 0..stride {
                out[i] = if state.nan_count[i] == 0 && weight_sum > T::zero() {
                    state.weighted_sum[i] / weight_sum
                } else {
                    T::nan()
                };
            }
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut EmaState<f64>,
        out: &mut Array<f64>,
        ts: i64,
        val: f64,
    ) -> bool {
        s.push(ts, &[val]);
        Ema::compute(state, (s,), out, ts, &Notify::new(&[], 0))
    }

    #[test]
    fn ema_constant_input() {
        // Use window=3 so that we can reach it with 10 values.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 3).init((&s,), i64::MIN);

        for i in 1..=10 {
            push_compute(&mut s, &mut state, &mut out, i, 10.0);
        }
        // EMA of constant should converge to that constant
        let last = out.as_slice()[0];
        assert!((last - 10.0).abs() < 1e-6, "expected ~10.0, got {last}");
    }

    #[test]
    fn ema_first_value_with_window_1() {
        // With window=1, the first tick is immediately the full window.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 1).init((&s,), i64::MIN);

        assert!(push_compute(&mut s, &mut state, &mut out, 1, 100.0));
        assert_eq!(out.as_slice()[0], 100.0);
    }

    #[test]
    fn ema_warmup() {
        // With window=3, first 2 ticks return false.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 3).init((&s,), i64::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 10.0));
        assert!(out.as_slice()[0].is_nan());

        assert!(!push_compute(&mut s, &mut state, &mut out, 2, 20.0));
        assert!(out.as_slice()[0].is_nan());

        assert!(push_compute(&mut s, &mut state, &mut out, 3, 30.0));
        assert!(!out.as_slice()[0].is_nan());
    }

    #[test]
    fn ema_two_values() {
        // Use window=2 so output is produced at tick 2.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 2).init((&s,), i64::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 10.0));
        assert!(push_compute(&mut s, &mut state, &mut out, 2, 20.0));

        // w0 = 0.5 (for x=20), w1 = 0.5*0.5=0.25 (for x=10)
        // EMA = (0.5*20 + 0.25*10) / (0.5 + 0.25) = 12.5 / 0.75 = 16.667
        let last = out.as_slice()[0];
        let expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
        assert!(
            (last - expected).abs() < 1e-10,
            "expected {expected}, got {last}"
        );
    }

    #[test]
    fn ema_nan_propagation() {
        // NaN in window → NaN output, until NaN is evicted.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 3).init((&s,), i64::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 10.0));
        assert!(out.as_slice()[0].is_nan());

        assert!(!push_compute(&mut s, &mut state, &mut out, 2, f64::NAN));
        assert!(out.as_slice()[0].is_nan());

        assert!(push_compute(&mut s, &mut state, &mut out, 3, 20.0));
        // NaN still in window → NaN
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 30.0);
        // NaN still in window [NaN, 20, 30] → NaN
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 5, 40.0);
        // NaN evicted, window [20, 30, 40] → valid
        let val = out.as_slice()[0];
        assert!(!val.is_nan(), "expected valid output after NaN eviction");
        // weighted: 0.5*40 + 0.25*30 + 0.125*20 = 30
        // weight_sum: 1 - 0.5^3 = 0.875
        let expected = 30.0 / 0.875;
        assert!(
            (val - expected).abs() < 1e-10,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn ema_window_bounds() {
        // With window=2, only last 2 values should matter
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 2).init((&s,), i64::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 100.0));
        assert!(push_compute(&mut s, &mut state, &mut out, 2, 100.0));
        push_compute(&mut s, &mut state, &mut out, 3, 0.0);
        push_compute(&mut s, &mut state, &mut out, 4, 0.0);

        // After two 0.0 values with window=2, the 100.0s should be evicted
        let last = out.as_slice()[0];
        assert!(
            last.abs() < 1e-10,
            "expected ~0.0 after window eviction, got {last}"
        );
    }

    #[test]
    fn ema_with_span() {
        // Use window=2 so output is produced at tick 2.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::with_span(3, 2).init((&s,), i64::MIN);

        // alpha = 2/(3+1) = 0.5
        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 10.0));
        assert!(push_compute(&mut s, &mut state, &mut out, 2, 20.0));
        let last = out.as_slice()[0];
        let expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
        assert!((last - expected).abs() < 1e-10);
    }

    #[test]
    fn ema_vector() {
        // Use window=2 so output is produced at tick 2.
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 2).init((&s,), i64::MIN);

        s.push(1, &[10.0, 100.0]);
        assert!(!Ema::compute(&mut state, (&s,), &mut out, 1, &Notify::new(&[], 0)));
        // Output stays NaN during warmup.
        assert!(out.as_slice()[0].is_nan());
        assert!(out.as_slice()[1].is_nan());

        s.push(2, &[20.0, 200.0]);
        assert!(Ema::compute(&mut state, (&s,), &mut out, 2, &Notify::new(&[], 0)));
        let row = out.as_slice();
        let expected_0 = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
        let expected_1 = (0.5 * 200.0 + 0.25 * 100.0) / (0.5 + 0.25);
        assert!((row[0] - expected_0).abs() < 1e-10);
        assert!((row[1] - expected_1).abs() < 1e-10);
    }

    #[test]
    fn ema_nan_at_start() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 2).init((&s,), i64::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, f64::NAN));
        assert!(out.as_slice()[0].is_nan());

        assert!(push_compute(&mut s, &mut state, &mut out, 2, 10.0));
        // NaN still in window → NaN
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 3, 20.0);
        // NaN evicted, window [10, 20]
        let val = out.as_slice()[0];
        assert!(!val.is_nan());
    }

    #[test]
    fn ema_multiple_nans() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 3).init((&s,), i64::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, f64::NAN));
        assert!(!push_compute(&mut s, &mut state, &mut out, 2, f64::NAN));
        assert!(push_compute(&mut s, &mut state, &mut out, 3, 10.0));
        // Two NaNs in window → NaN
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 20.0);
        // One NaN remains → NaN
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 5, 30.0);
        // Both NaNs evicted → valid
        assert!(!out.as_slice()[0].is_nan());
    }

    #[test]
    fn ema_nan_vector_independent() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 2).init((&s,), i64::MIN);

        // NaN only in element 0
        s.push(1, &[f64::NAN, 10.0]);
        assert!(!Ema::compute(&mut state, (&s,), &mut out, 1, &Notify::new(&[], 0)));
        // Output stays NaN during warmup.
        assert!(out.as_slice()[0].is_nan());
        assert!(out.as_slice()[1].is_nan());

        s.push(2, &[5.0, 20.0]);
        assert!(Ema::compute(&mut state, (&s,), &mut out, 2, &Notify::new(&[], 0)));
        // NaN still in window for element 0
        assert!(out.as_slice()[0].is_nan());
        assert!(!out.as_slice()[1].is_nan());

        s.push(3, &[15.0, 30.0]);
        Ema::compute(&mut state, (&s,), &mut out, 3, &Notify::new(&[], 0));
        // NaN evicted for element 0
        assert!(!out.as_slice()[0].is_nan());
        assert!(!out.as_slice()[1].is_nan());
    }

    #[test]
    fn ema_nan_eviction_restores_correct_value() {
        // After NaN exits, the weighted average uses only valid values.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 2).init((&s,), i64::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, f64::NAN));
        assert!(push_compute(&mut s, &mut state, &mut out, 2, 10.0));
        assert!(out.as_slice()[0].is_nan()); // NaN still in window

        push_compute(&mut s, &mut state, &mut out, 3, 10.0);
        // NaN evicted, window [10, 10] → EMA of constant = 10
        let val = out.as_slice()[0];
        assert!(
            (val - 10.0).abs() < 1e-10,
            "expected 10.0 after NaN eviction, got {val}"
        );
    }
}
