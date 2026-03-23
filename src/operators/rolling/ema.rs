//! Exponential moving average operator with window-normalized weights.
//!
//! On each tick, reads the latest value from the input Series and updates
//! a running EMA. The EMA is normalized within a finite window so that
//! weights sum to ~1 regardless of truncation.

use num_traits::Float;

use crate::{Operator, Scalar, Series};

/// Exponential moving average.
///
/// For each element position, maintains a running weighted average:
/// ```text
/// ema_t = sum(w_i * x_{t-i}, i=0..window-1) / sum(w_i, i=0..window-1)
/// ```
/// where `w_i = alpha * (1 - alpha)^i`.
///
/// NaN handling: if the input element is NaN, the output for that element
/// is NaN. Use [`ForwardFill`](super::ForwardFill) upstream if you want
/// carry-forward semantics.
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

/// Runtime state for EMA: alpha, window, and per-element running buffers.
pub struct EmaState<T: Scalar + Float> {
    alpha: T,
    window: usize,
    /// Per-element: weighted sum of values in the window.
    weighted_sum: Vec<T>,
    /// Per-element: sum of weights in the window.
    weight_sum: Vec<T>,
    /// Per-element: individual weights for the last `window` ticks.
    /// Ring buffer: weights[tick % window] is the oldest weight.
    weights: Vec<Vec<T>>,
    /// Per-element: individual weighted values for the last `window` ticks.
    weighted_vals: Vec<Vec<T>>,
    /// Current tick count (for ring buffer indexing).
    tick: usize,
}

impl<T: Scalar + Float> Operator for Ema<T> {
    type State = EmaState<T>;
    type Inputs = (Series<T>,);
    type Output = Series<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (EmaState<T>, Series<T>) {
        let stride = inputs.0.stride();
        let state = EmaState {
            alpha: self.alpha,
            window: self.window,
            weighted_sum: vec![T::zero(); stride],
            weight_sum: vec![T::zero(); stride],
            weights: vec![vec![T::zero(); self.window]; stride],
            weighted_vals: vec![vec![T::zero(); self.window]; stride],
            tick: 0,
        };
        (state, Series::new(inputs.0.shape()))
    }

    fn compute(
        state: &mut EmaState<T>,
        inputs: (&Series<T>,),
        output: &mut Series<T>,
        timestamp: i64,
    ) -> bool {
        let series = inputs.0;
        let row = series.last().unwrap();
        let stride = row.len();
        let alpha = state.alpha;
        let one_minus_alpha = T::one() - alpha;
        let window = state.window;
        let slot = state.tick % window;

        let mut buf = vec![T::nan(); stride];

        for i in 0..stride {
            let x = row[i];
            if x.is_nan() {
                // NaN input → NaN output, but still need to evict old slot
                let old_w = state.weights[i][slot];
                let old_wv = state.weighted_vals[i][slot];
                state.weight_sum[i] = state.weight_sum[i] - old_w;
                state.weighted_sum[i] = state.weighted_sum[i] - old_wv;
                state.weights[i][slot] = T::zero();
                state.weighted_vals[i][slot] = T::zero();

                // Decay all existing weights
                state.weight_sum[i] = state.weight_sum[i] * one_minus_alpha;
                state.weighted_sum[i] = state.weighted_sum[i] * one_minus_alpha;
                for j in 0..window {
                    state.weights[i][j] = state.weights[i][j] * one_minus_alpha;
                    state.weighted_vals[i][j] = state.weighted_vals[i][j] * one_minus_alpha;
                }
                // buf[i] stays NaN
            } else {
                // Evict oldest value from the window
                let old_w = state.weights[i][slot];
                let old_wv = state.weighted_vals[i][slot];
                state.weight_sum[i] = state.weight_sum[i] - old_w;
                state.weighted_sum[i] = state.weighted_sum[i] - old_wv;

                // Decay all existing weights
                state.weight_sum[i] = state.weight_sum[i] * one_minus_alpha;
                state.weighted_sum[i] = state.weighted_sum[i] * one_minus_alpha;
                for j in 0..window {
                    state.weights[i][j] = state.weights[i][j] * one_minus_alpha;
                    state.weighted_vals[i][j] = state.weighted_vals[i][j] * one_minus_alpha;
                }

                // Add new value with weight alpha
                let w = alpha;
                state.weights[i][slot] = w;
                state.weighted_vals[i][slot] = w * x;
                state.weight_sum[i] = state.weight_sum[i] + w;
                state.weighted_sum[i] = state.weighted_sum[i] + w * x;

                if state.weight_sum[i] > T::zero() {
                    buf[i] = state.weighted_sum[i] / state.weight_sum[i];
                }
            }
        }

        state.tick += 1;
        output.push(timestamp, &buf);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut EmaState<f64>,
        out: &mut Series<f64>,
        ts: i64,
        val: f64,
    ) {
        s.push(ts, &[val]);
        Ema::compute(state, (s,), out, ts);
    }

    #[test]
    fn ema_constant_input() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 100).init((&s,), i64::MIN);

        for i in 1..=10 {
            push_compute(&mut s, &mut state, &mut out, i, 10.0);
        }
        // EMA of constant should converge to that constant
        let last = out.last().unwrap()[0];
        assert!((last - 10.0).abs() < 1e-6, "expected ~10.0, got {last}");
    }

    #[test]
    fn ema_first_value() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 10).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 100.0);
        // First value: only one weight, so EMA = value itself
        assert_eq!(out.last().unwrap()[0], 100.0);
    }

    #[test]
    fn ema_two_values() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 10).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 10.0);
        push_compute(&mut s, &mut state, &mut out, 2, 20.0);

        // w0 = 0.5 (for x=20), w1 = 0.5*0.5=0.25 (for x=10)
        // EMA = (0.5*20 + 0.25*10) / (0.5 + 0.25) = 12.5 / 0.75 = 16.667
        let last = out.last().unwrap()[0];
        let expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
        assert!(
            (last - expected).abs() < 1e-10,
            "expected {expected}, got {last}"
        );
    }

    #[test]
    fn ema_nan_output() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 10).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 10.0);
        assert_eq!(out.last().unwrap()[0], 10.0);

        // NaN input → NaN output
        s.push(2, &[f64::NAN]);
        Ema::compute(&mut state, (&s,), &mut out, 2);
        assert!(out.last().unwrap()[0].is_nan());

        // Valid input resumes
        push_compute(&mut s, &mut state, &mut out, 3, 20.0);
        assert!(!out.last().unwrap()[0].is_nan());
    }

    #[test]
    fn ema_window_bounds() {
        // With window=2, only last 2 values should matter
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 2).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 100.0);
        push_compute(&mut s, &mut state, &mut out, 2, 100.0);
        push_compute(&mut s, &mut state, &mut out, 3, 0.0);
        push_compute(&mut s, &mut state, &mut out, 4, 0.0);

        // After two 0.0 values with window=2, the 100.0s should be evicted
        let last = out.last().unwrap()[0];
        assert!(
            last.abs() < 1e-10,
            "expected ~0.0 after window eviction, got {last}"
        );
    }

    #[test]
    fn ema_with_span() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Ema::<f64>::with_span(3, 100).init((&s,), i64::MIN);

        // alpha = 2/(3+1) = 0.5
        push_compute(&mut s, &mut state, &mut out, 1, 10.0);
        push_compute(&mut s, &mut state, &mut out, 2, 20.0);
        let last = out.last().unwrap()[0];
        let expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
        assert!((last - expected).abs() < 1e-10);
    }

    #[test]
    fn ema_vector() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = Ema::<f64>::new(0.5, 10).init((&s,), i64::MIN);

        s.push(1, &[10.0, 100.0]);
        Ema::compute(&mut state, (&s,), &mut out, 1);
        assert_eq!(out.last().unwrap(), &[10.0, 100.0]);

        s.push(2, &[20.0, 200.0]);
        Ema::compute(&mut state, (&s,), &mut out, 2);
        let row = out.last().unwrap();
        let expected_0 = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25);
        let expected_1 = (0.5 * 200.0 + 0.25 * 100.0) / (0.5 + 0.25);
        assert!((row[0] - expected_0).abs() < 1e-10);
        assert!((row[1] - expected_1).abs() < 1e-10);
    }
}
