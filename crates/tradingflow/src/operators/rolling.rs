//! Rolling (windowed) operators, implemented directly on
//! [`Operator`]. The [`Accumulator`] / [`Window`] /
//! [`Rolling`] framework and the four accumulators + [`Ema`] keep the output
//! buffer in the operator state (sized on the `init` build call); the
//! `Accumulator: Send + Sync` bound holds because the accumulator lives in the
//! operator State, which is a `Send + Sync` cell.
//!
//! Note: rolling reads event time from the series window (`timestamp`), NOT
//! the threaded `Instant`, so the time-delta window needs no clock wiring. A
//! [`SeriesView`] window is view-local, so all addressing is relative to the
//! window's newest row (the accumulated span is always the window's tail);
//! a retention-bounded record works as long as the bound covers the rolling
//! window — the rows the accumulator still holds are never trimmed away.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Duration, Instant, Layout, Scalar, SeriesView};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, SeriesPort};

/// Convert an accumulator's dynamic output shape into a static `[usize; NO]`.
#[inline]
fn out_extents<const NO: usize>(shape: &[usize]) -> [usize; NO] {
    <[usize; NO]>::try_from(shape)
        .unwrap_or_else(|_| panic!("rolling: output rank {} != NO {NO}", shape.len()))
}

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

/// Pairs an [`Accumulator`] with a [`Window`] strategy. `NI` is the input
/// series' element rank, `NO` the output rank (they differ only for
/// rank-changing accumulators, e.g. covariance `[K] → [K, K]`).
pub struct Rolling<A: Accumulator, const NI: usize, const NO: usize> {
    window: Window,
    _phantom: PhantomData<A>,
}

impl<A: Accumulator, const NI: usize, const NO: usize> Clone for Rolling<A, NI, NO> {
    fn clone(&self) -> Self {
        Self {
            window: self.window,
            _phantom: PhantomData,
        }
    }
}

impl<A: Accumulator, const NI: usize, const NO: usize> Rolling<A, NI, NO> {
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

/// Runtime state for [`Rolling`]: the window config, accumulator bookkeeping
/// (`count` rows accumulated — always the newest `count` rows of the series
/// window), plus the output buffer.
pub struct RollingState<A: Accumulator, const NO: usize> {
    window: Window,
    count: usize,
    accumulator: A,
    out: Array<A::Scalar, NO>,
}

impl<A: Accumulator, const NI: usize, const NO: usize> Operator for Rolling<A, NI, NO> {
    type Inputs = SeriesPort<A::Scalar, NI>;
    type Outputs = ArrayPort<A::Scalar, NO>;
    type Context = Instant;
    type State = RollingState<A, NO>;

    fn init(self, (_, series): (bool, SeriesView<'_, A::Scalar, NI>)) -> Self::State {
        let input_shape = series.layout().extents();
        let output_shape = A::output_shape(&input_shape);
        let output_stride: usize = output_shape.iter().product();
        RollingState {
            window: self.window,
            count: 0,
            accumulator: A::new(&input_shape),
            out: Array::from_parts(
                out_extents::<NO>(&output_shape),
                vec![A::Scalar::nan(); output_stride].into(),
            ),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, A::Scalar, NI>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, A::Scalar, NO>) {
        // The accumulated rows are always the newest `count` rows of the
        // window, so the oldest accumulated row sits at view-local index
        // `len - count` — stable under retention trims of older rows.
        let len = series.len();

        state.accumulator.add(series.at(len - 1).unwrap().1.data());
        state.count += 1;

        match state.window {
            Window::Count(w) => {
                while state.count > w {
                    state
                        .accumulator
                        .remove(series.at(len - state.count).unwrap().1.data());
                    state.count -= 1;
                }
                if state.count < w {
                    return (false, state.out.view());
                }
            }
            Window::TimeDelta(w) => {
                let current_ts = series.at(len - 1).unwrap().0;
                let cutoff = current_ts - w;
                while state.count > 0 && series.at(len - state.count).unwrap().0 < cutoff {
                    state
                        .accumulator
                        .remove(series.at(len - state.count).unwrap().1.data());
                    state.count -= 1;
                }
                if state.count == 0 {
                    return (false, state.out.view());
                }
            }
        }

        state.accumulator.write(state.count, state.out.data_mut());
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, A::Scalar, NI>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, A::Scalar, NO>) {
        (false, state.out.view())
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

    #[expect(
        clippy::needless_range_loop,
        reason = "i/j also address the flat k*k `sum_cross` cross-moment matrix"
    )]
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

    #[expect(
        clippy::needless_range_loop,
        reason = "i/j also address the flat k*k `sum_cross` cross-moment matrix"
    )]
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
                output[i * k + j] = if self.nonfinite_count[i] == 0 && self.nonfinite_count[j] == 0
                {
                    self.sum_cross[i * k + j] / n - (self.sum[i] / n) * (self.sum[j] / n)
                } else {
                    T::nan()
                };
            }
        }
    }
}

/// Element-wise rolling sum (output rank `NO` = input element rank).
pub type RollingSum<T, const NO: usize> = Rolling<SumAccumulator<T>, NO, NO>;
/// Element-wise rolling mean (output rank `NO` = input element rank).
pub type RollingMean<T, const NO: usize> = Rolling<MeanAccumulator<T>, NO, NO>;
/// Element-wise rolling population variance (output rank `NO` = input rank).
pub type RollingVariance<T, const NO: usize> = Rolling<VarianceAccumulator<T>, NO, NO>;
/// Pairwise rolling covariance matrix (`[K] → [K, K]`, output rank 2).
pub type RollingCovariance<T> = Rolling<CovarianceAccumulator<T>, 1, 2>;

// ===========================================================================
// Constructors
// ===========================================================================

/// [`Rolling`] over an explicit [`Window`] — the accumulator-generic form.
pub fn rolling<A: Accumulator, const NI: usize, const NO: usize>(
    window: Window,
) -> Rolling<A, NI, NO> {
    match window {
        Window::Count(w) => Rolling::count(w),
        Window::TimeDelta(w) => Rolling::time_delta(w),
    }
}

/// Rolling sum over a recorded [`Series`](tradingflow_data::Series): `rolling_sum(Window::Count(20)) @ xs`.
pub fn rolling_sum<T: Scalar + Float, const NO: usize>(window: Window) -> RollingSum<T, NO> {
    rolling(window)
}

/// Rolling mean over a recorded [`Series`](tradingflow_data::Series). The self-recording counterpart
/// over a live array handle is [`ma`](super::formula::ma) / [`ma_time`](super::formula::ma_time).
pub fn rolling_mean<T: Scalar + Float, const NO: usize>(window: Window) -> RollingMean<T, NO> {
    rolling(window)
}

/// Rolling population variance over a recorded [`Series`](tradingflow_data::Series). Self-recording
/// counterpart: [`mvar`](super::formula::mvar).
pub fn rolling_variance<T: Scalar + Float, const NO: usize>(
    window: Window,
) -> RollingVariance<T, NO> {
    rolling(window)
}

/// Pairwise rolling covariance matrix (`[K] → [K, K]`) over a recorded
/// [`Series`](tradingflow_data::Series).
pub fn rolling_covariance<T: Scalar + Float>(window: Window) -> RollingCovariance<T> {
    rolling(window)
}

/// [`Ema`] over a recorded [`Series`](tradingflow_data::Series) — the primitive behind the
/// self-recording [`ema`](super::formula::ema). (Named `_series` because `ema` is taken
/// by its live-array counterpart.)
pub fn ema_series<T: Scalar + Float, const NO: usize>(alpha: T, window: usize) -> Ema<T, NO> {
    Ema::new(alpha, window)
}

// ===========================================================================
// EMA (standalone — does not use the Accumulator abstraction)
// ===========================================================================

/// Exponential moving average with window-normalized weights (output rank `NO`
/// = input element rank).
#[derive(Clone)]
pub struct Ema<T: Scalar + Float, const NO: usize> {
    alpha: T,
    window: usize,
}

impl<T: Scalar + Float, const NO: usize> Ema<T, NO> {
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

/// Runtime state for [`Ema`]: the decay config, weighted-sum bookkeeping,
/// plus the output buffer.
pub struct EmaState<T: Scalar + Float, const NO: usize> {
    alpha: T,
    one_minus_alpha: T,
    decay_factor: T,
    window: usize,
    weighted_sum: Vec<T>,
    nonfinite_count: Vec<u32>,
    fill_decay: T,
    out: Array<T, NO>,
}

impl<T: Scalar + Float, const NO: usize> Operator for Ema<T, NO> {
    type Inputs = SeriesPort<T, NO>;
    type Outputs = ArrayPort<T, NO>;
    type Context = Instant;
    type State = EmaState<T, NO>;

    fn init(self, (_, series): (bool, SeriesView<'_, T, NO>)) -> Self::State {
        let one_minus_alpha = T::one() - self.alpha;
        let mut decay_factor = T::one();
        for _ in 0..self.window {
            decay_factor = decay_factor * one_minus_alpha;
        }
        let stride = series.layout().len();
        EmaState {
            alpha: self.alpha,
            one_minus_alpha,
            decay_factor,
            window: self.window,
            weighted_sum: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
            fill_decay: T::one(),
            out: Array::from_parts(series.layout().extents(), vec![T::nan(); stride].into()),
        }
    }

    #[expect(
        clippy::needless_range_loop,
        reason = "i walks several parallel per-column arrays plus `series.elem(..)[[i]]`"
    )]
    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, T, NO>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, NO>) {
        let len = series.len();
        let row = series.at(len - 1).unwrap().1.data();
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
                let x_old = series.at(len - 1 - state.window).unwrap().1.data()[i];
                if !x_old.is_finite() {
                    state.nonfinite_count[i] -= 1;
                } else {
                    let evict_weight = alpha * state.decay_factor;
                    state.weighted_sum[i] = state.weighted_sum[i] - evict_weight * x_old;
                }
            }
        }

        if len < state.window {
            (false, state.out.view())
        } else {
            let out = state.out.data_mut();
            for i in 0..stride {
                out[i] = if state.nonfinite_count[i] == 0 && weight_sum > T::zero() {
                    state.weighted_sum[i] / weight_sum
                } else {
                    T::nan()
                };
            }
            (true, state.out.view())
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, T, NO>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, NO>) {
        (false, state.out.view())
    }
}
