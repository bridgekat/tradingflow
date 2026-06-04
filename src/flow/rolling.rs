//! Rolling (windowed) operators — port of [`crate::operators::rolling`]. The
//! [`Accumulator`] / [`Window`] / [`Rolling`] framework and the four
//! accumulators + [`Ema`] are transcribed verbatim except for the trait surface
//! (`Input`→`Port`, `produced`→notify) and the `Accumulator: Send + Sync` bound
//! (the accumulator lives in the operator State, which is a `Send + Sync` cell).
//!
//! Note: rolling reads event time from `series.timestamps()`, NOT the threaded
//! `Instant`, so the time-delta window needs no clock wiring.

use std::marker::PhantomData;

use num_traits::Float;

use flowgraph::typed::Port;

use super::op::Operator;
use crate::{Array, Duration, Instant, Scalar, Series};

// ===========================================================================
// Accumulator trait + Window
// ===========================================================================

/// Incremental computation over a rolling window of array elements.
pub trait Accumulator: Send + Sync + 'static {
    type Scalar: Scalar + Float;

    fn new(input_shape: &[usize]) -> Self;

    fn output_shape(input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    fn add(&mut self, element: &[Self::Scalar]);
    fn remove(&mut self, element: &[Self::Scalar]);
    fn write(&self, count: usize, output: &mut [Self::Scalar]);
}

/// Rolling window selection strategy.
#[derive(Debug, Clone, Copy)]
pub enum Window {
    Count(usize),
    TimeDelta(Duration),
}

// ===========================================================================
// Generic rolling operator
// ===========================================================================

/// Pairs an [`Accumulator`] with a [`Window`] strategy.
pub struct Rolling<A: Accumulator> {
    window: Window,
    _phantom: PhantomData<A>,
}

impl<A: Accumulator> Clone for Rolling<A> {
    fn clone(&self) -> Self {
        Self {
            window: self.window,
            _phantom: PhantomData,
        }
    }
}

impl<A: Accumulator> Rolling<A> {
    /// Count-based window of the last `window` elements; output only once full.
    pub fn count(window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        Self {
            window: Window::Count(window),
            _phantom: PhantomData,
        }
    }

    /// Time-delta window: all elements within `window` of the latest timestamp.
    pub fn time_delta(window: Duration) -> Self {
        assert!(window.as_nanos() >= 0, "window must be non-negative");
        Self {
            window: Window::TimeDelta(window),
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Rolling`].
pub struct RollingState<A: Accumulator> {
    window: Window,
    start: usize,
    count: usize,
    accumulator: A,
}

impl<A: Accumulator> Operator for Rolling<A> {
    type State = RollingState<A>;
    type Inputs = Port<Series<A::Scalar>>;
    type Output = Array<A::Scalar>;

    fn init(
        &self,
        inputs: &Series<A::Scalar>,
        _ts: Instant,
    ) -> (RollingState<A>, Array<A::Scalar>) {
        let input_shape = inputs.shape();
        let output_shape = A::output_shape(input_shape);
        let output_stride: usize = output_shape.iter().product();
        let state = RollingState {
            window: self.window,
            start: 0,
            count: 0,
            accumulator: A::new(input_shape),
        };
        (
            state,
            Array::from_vec(&output_shape, vec![A::Scalar::nan(); output_stride]),
        )
    }

    fn compute(
        state: &mut RollingState<A>,
        inputs: &Series<A::Scalar>,
        output: &mut Array<A::Scalar>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let series = inputs;
        let len = series.len();

        state.accumulator.add(series.at(len - 1));
        state.count += 1;

        match state.window {
            Window::Count(w) => {
                while state.count > w {
                    state.accumulator.remove(series.at(state.start));
                    state.start += 1;
                    state.count -= 1;
                }
                if state.count < w {
                    return false;
                }
            }
            Window::TimeDelta(w) => {
                let current_ts = series.timestamps()[len - 1];
                let cutoff = current_ts - w;
                while state.start < len && series.timestamps()[state.start] < cutoff {
                    state.accumulator.remove(series.at(state.start));
                    state.start += 1;
                    state.count -= 1;
                }
                if state.count == 0 {
                    return false;
                }
            }
        }

        state.accumulator.write(state.count, output.as_mut_slice());
        true
    }
}

// ===========================================================================
// Accumulators
// ===========================================================================

/// Incremental sum (non-finite values skipped + counted; NaN if any present).
pub struct SumAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for SumAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        let stride: usize = input_shape.iter().product();
        Self {
            sum: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
        }
    }

    fn add(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] += 1;
            } else {
                self.sum[j] = self.sum[j] + v;
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] -= 1;
            } else {
                self.sum[j] = self.sum[j] - v;
            }
        }
    }

    fn write(&self, _count: usize, output: &mut [T]) {
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] > 0 {
                T::nan()
            } else {
                self.sum[j]
            };
        }
    }
}

/// Incremental mean.
pub struct MeanAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for MeanAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        let stride: usize = input_shape.iter().product();
        Self {
            sum: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
        }
    }

    fn add(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] += 1;
            } else {
                self.sum[j] = self.sum[j] + v;
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] -= 1;
            } else {
                self.sum[j] = self.sum[j] - v;
            }
        }
    }

    fn write(&self, count: usize, output: &mut [T]) {
        let n = T::from(count).unwrap();
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] > 0 {
                T::nan()
            } else {
                self.sum[j] / n
            };
        }
    }
}

/// Incremental population variance via `E[x²] − E[x]²`.
pub struct VarianceAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    sum_sq: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for VarianceAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        let stride: usize = input_shape.iter().product();
        Self {
            sum: vec![T::zero(); stride],
            sum_sq: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
        }
    }

    fn add(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] += 1;
            } else {
                self.sum[j] = self.sum[j] + v;
                self.sum_sq[j] = self.sum_sq[j] + v * v;
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] -= 1;
            } else {
                self.sum[j] = self.sum[j] - v;
                self.sum_sq[j] = self.sum_sq[j] - v * v;
            }
        }
    }

    fn write(&self, count: usize, output: &mut [T]) {
        let n = T::from(count).unwrap();
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] == 0 {
                let mean = self.sum[j] / n;
                self.sum_sq[j] / n - mean * mean
            } else {
                T::nan()
            };
        }
    }
}

/// Incremental pairwise covariance matrix (`[K] → [K, K]`).
pub struct CovarianceAccumulator<T: Scalar + Float> {
    k: usize,
    sum: Vec<T>,
    sum_cross: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for CovarianceAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        assert_eq!(
            input_shape.len(),
            1,
            "CovarianceAccumulator requires 1D input, got shape {input_shape:?}",
        );
        let k = input_shape[0];
        Self {
            k,
            sum: vec![T::zero(); k],
            sum_cross: vec![T::zero(); k * k],
            nonfinite_count: vec![0; k],
        }
    }

    fn output_shape(input_shape: &[usize]) -> Vec<usize> {
        assert_eq!(
            input_shape.len(),
            1,
            "CovarianceAccumulator requires 1D input, got shape {input_shape:?}",
        );
        vec![input_shape[0], input_shape[0]]
    }

    fn add(&mut self, element: &[T]) {
        let k = self.k;
        for i in 0..k {
            let xi = element[i];
            if !xi.is_finite() {
                self.nonfinite_count[i] += 1;
            } else {
                self.sum[i] = self.sum[i] + xi;
            }
        }
        for i in 0..k {
            let xi = element[i];
            if !xi.is_finite() {
                continue;
            }
            for j in i..k {
                let xj = element[j];
                if !xj.is_finite() {
                    continue;
                }
                let prod = xi * xj;
                self.sum_cross[i * k + j] = self.sum_cross[i * k + j] + prod;
                if i != j {
                    self.sum_cross[j * k + i] = self.sum_cross[j * k + i] + prod;
                }
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        let k = self.k;
        for i in 0..k {
            let xi = element[i];
            if !xi.is_finite() {
                self.nonfinite_count[i] -= 1;
            } else {
                self.sum[i] = self.sum[i] - xi;
            }
        }
        for i in 0..k {
            let xi = element[i];
            if !xi.is_finite() {
                continue;
            }
            for j in i..k {
                let xj = element[j];
                if !xj.is_finite() {
                    continue;
                }
                let prod = xi * xj;
                self.sum_cross[i * k + j] = self.sum_cross[i * k + j] - prod;
                if i != j {
                    self.sum_cross[j * k + i] = self.sum_cross[j * k + i] - prod;
                }
            }
        }
    }

    fn write(&self, count: usize, output: &mut [T]) {
        let k = self.k;
        let n = T::from(count).unwrap();
        for i in 0..k {
            for j in 0..k {
                output[i * k + j] = if self.nonfinite_count[i] == 0 && self.nonfinite_count[j] == 0 {
                    self.sum_cross[i * k + j] / n - (self.sum[i] / n) * (self.sum[j] / n)
                } else {
                    T::nan()
                };
            }
        }
    }
}

/// Element-wise rolling sum.
pub type RollingSum<T> = Rolling<SumAccumulator<T>>;
/// Element-wise rolling mean.
pub type RollingMean<T> = Rolling<MeanAccumulator<T>>;
/// Element-wise rolling population variance.
pub type RollingVariance<T> = Rolling<VarianceAccumulator<T>>;
/// Pairwise rolling covariance matrix (`[K] → [K, K]`).
pub type RollingCovariance<T> = Rolling<CovarianceAccumulator<T>>;

// ===========================================================================
// EMA (standalone — does not use the Accumulator abstraction)
// ===========================================================================

/// Exponential moving average with window-normalized weights.
#[derive(Clone)]
pub struct Ema<T: Scalar + Float> {
    alpha: T,
    window: usize,
}

impl<T: Scalar + Float> Ema<T> {
    pub fn new(alpha: T, window: usize) -> Self {
        assert!(
            alpha > T::zero() && alpha <= T::one(),
            "alpha must be in (0, 1]"
        );
        assert!(window >= 1, "window must be >= 1");
        Self { alpha, window }
    }

    pub fn with_span(span: usize, window: usize) -> Self {
        assert!(span >= 1, "span must be >= 1");
        let alpha = T::from(2.0).unwrap() / T::from(span + 1).unwrap();
        Self::new(alpha, window)
    }

    pub fn with_half_life(half_life: T, window: usize) -> Self {
        assert!(half_life > T::zero(), "half_life must be > 0");
        let alpha = T::one() - (-T::from(2.0).unwrap().ln() / half_life).exp();
        Self::new(alpha, window)
    }
}

/// Runtime state for [`Ema`].
pub struct EmaState<T: Scalar + Float> {
    alpha: T,
    one_minus_alpha: T,
    decay_factor: T,
    window: usize,
    weighted_sum: Vec<T>,
    nonfinite_count: Vec<u32>,
    fill_decay: T,
}

impl<T: Scalar + Float> Operator for Ema<T> {
    type State = EmaState<T>;
    type Inputs = Port<Series<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Series<T>, _ts: Instant) -> (EmaState<T>, Array<T>) {
        let stride = inputs.stride();
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
            nonfinite_count: vec![0; stride],
            fill_decay: T::one(),
        };
        let shape = inputs.shape();
        let stride = shape.iter().product::<usize>();
        (state, Array::from_vec(shape, vec![T::nan(); stride]))
    }

    fn compute(
        state: &mut EmaState<T>,
        inputs: &Series<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let series = inputs;
        let len = series.len();
        let row = series.at(len - 1);
        let stride = row.len();
        let alpha = state.alpha;
        let one_minus_alpha = state.one_minus_alpha;

        state.fill_decay = state.fill_decay * one_minus_alpha;
        let weight_sum = T::one()
            - if len >= state.window {
                state.decay_factor
            } else {
                state.fill_decay
            };

        for i in 0..stride {
            let x = row[i];
            state.weighted_sum[i] = state.weighted_sum[i] * one_minus_alpha;
            if !x.is_finite() {
                state.nonfinite_count[i] += 1;
            } else {
                state.weighted_sum[i] = state.weighted_sum[i] + alpha * x;
            }
            if len > state.window {
                let x_old = series.at(len - 1 - state.window)[i];
                if !x_old.is_finite() {
                    state.nonfinite_count[i] -= 1;
                } else {
                    let evict_weight = alpha * state.decay_factor;
                    state.weighted_sum[i] = state.weighted_sum[i] - evict_weight * x_old;
                }
            }
        }

        if len < state.window {
            false
        } else {
            let out = output.as_mut_slice();
            for i in 0..stride {
                out[i] = if state.nonfinite_count[i] == 0 && weight_sum > T::zero() {
                    state.weighted_sum[i] / weight_sum
                } else {
                    T::nan()
                };
            }
            true
        }
    }
}
