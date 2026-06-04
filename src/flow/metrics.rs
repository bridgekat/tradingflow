//! Clock-driven since-inception financial metrics — port of
//! [`crate::operators::metrics`]. The first four take `(data, clock)` inputs
//! and gate on the clock's notify bit (returning `false` off-tick); `Drawdown`
//! is single-input.

use std::marker::PhantomData;

use num_traits::Float;

use flowgraph::typed::Port;

use super::op::Operator;
use crate::{Array, Instant, Scalar};

// ---------------------------------------------------------------------------
// CompoundReturn
// ---------------------------------------------------------------------------

/// `(current / first)^(1/n) - 1` over clock ticks since inception.
#[derive(Clone)]
pub struct CompoundReturn<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> CompoundReturn<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for CompoundReturn<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CompoundReturnState<T: Scalar + Float> {
    first_value: T,
    count: usize,
}

impl<T: Scalar + Float> Operator for CompoundReturn<T> {
    type State = CompoundReturnState<T>;
    type Inputs = (Port<Array<T>>, Port<()>);
    type Output = Array<T>;

    fn init(&self, _inputs: (&Array<T>, &()), _ts: Instant) -> (Self::State, Array<T>) {
        (
            CompoundReturnState {
                first_value: T::nan(),
                count: 0,
            },
            Array::scalar(T::nan()),
        )
    }

    fn compute(
        state: &mut CompoundReturnState<T>,
        inputs: (&Array<T>, &()),
        output: &mut Array<T>,
        _ts: Instant,
        produced: (bool, bool),
    ) -> bool {
        let (_produced_data, produced_clock) = produced;
        if !produced_clock {
            return false;
        }
        let current = inputs.0[0];
        if current.is_nan() {
            return false;
        }

        state.count += 1;

        if state.first_value.is_nan() {
            state.first_value = current;
            output[0] = T::zero();
            return true;
        }

        if state.first_value <= T::zero() || current <= T::zero() {
            output[0] = T::nan();
            return true;
        }

        let ratio = current / state.first_value;
        let n = T::from(state.count - 1).unwrap();
        if n <= T::zero() {
            output[0] = T::zero();
        } else {
            output[0] = ratio.powf(T::one() / n) - T::one();
        }
        true
    }
}

// ---------------------------------------------------------------------------
// AverageReturn
// ---------------------------------------------------------------------------

/// Arithmetic mean of period returns since inception.
#[derive(Clone)]
pub struct AverageReturn<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> AverageReturn<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for AverageReturn<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AverageReturnState<T: Scalar + Float> {
    prev: T,
    sum: T,
    count: usize,
}

impl<T: Scalar + Float> Operator for AverageReturn<T> {
    type State = AverageReturnState<T>;
    type Inputs = (Port<Array<T>>, Port<()>);
    type Output = Array<T>;

    fn init(&self, _inputs: (&Array<T>, &()), _ts: Instant) -> (Self::State, Array<T>) {
        (
            AverageReturnState {
                prev: T::nan(),
                sum: T::zero(),
                count: 0,
            },
            Array::scalar(T::nan()),
        )
    }

    fn compute(
        state: &mut AverageReturnState<T>,
        inputs: (&Array<T>, &()),
        output: &mut Array<T>,
        _ts: Instant,
        produced: (bool, bool),
    ) -> bool {
        let (_produced_data, produced_clock) = produced;
        if !produced_clock {
            return false;
        }
        let current = inputs.0[0];
        if current.is_nan() {
            return false;
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.count += 1;
            output[0] = state.sum / T::from(state.count).unwrap();
        }

        state.prev = current;
        true
    }
}

// ---------------------------------------------------------------------------
// Volatility
// ---------------------------------------------------------------------------

/// Population standard deviation of period returns since inception.
#[derive(Clone)]
pub struct Volatility<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Volatility<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Volatility<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct VolatilityState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
}

impl<T: Scalar + Float> Operator for Volatility<T> {
    type State = VolatilityState<T>;
    type Inputs = (Port<Array<T>>, Port<()>);
    type Output = Array<T>;

    fn init(&self, _inputs: (&Array<T>, &()), _ts: Instant) -> (Self::State, Array<T>) {
        (
            VolatilityState {
                prev: T::nan(),
                sum: T::zero(),
                sum_sq: T::zero(),
                count: 0,
            },
            Array::scalar(T::nan()),
        )
    }

    fn compute(
        state: &mut VolatilityState<T>,
        inputs: (&Array<T>, &()),
        output: &mut Array<T>,
        _ts: Instant,
        produced: (bool, bool),
    ) -> bool {
        let (_produced_data, produced_clock) = produced;
        if !produced_clock {
            return false;
        }
        let current = inputs.0[0];
        if current.is_nan() {
            return false;
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.sum_sq = state.sum_sq + r * r;
            state.count += 1;
            let n = T::from(state.count).unwrap();
            let mean = state.sum / n;
            let var = state.sum_sq / n - mean * mean;
            output[0] = if var > T::zero() { var.sqrt() } else { T::zero() };
        }

        state.prev = current;
        true
    }
}

// ---------------------------------------------------------------------------
// SharpeRatio
// ---------------------------------------------------------------------------

/// `mean(r) / std(r)` of period returns since inception.
#[derive(Clone)]
pub struct SharpeRatio<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> SharpeRatio<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for SharpeRatio<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SharpeRatioState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
}

impl<T: Scalar + Float> Operator for SharpeRatio<T> {
    type State = SharpeRatioState<T>;
    type Inputs = (Port<Array<T>>, Port<()>);
    type Output = Array<T>;

    fn init(&self, _inputs: (&Array<T>, &()), _ts: Instant) -> (Self::State, Array<T>) {
        (
            SharpeRatioState {
                prev: T::nan(),
                sum: T::zero(),
                sum_sq: T::zero(),
                count: 0,
            },
            Array::scalar(T::nan()),
        )
    }

    fn compute(
        state: &mut SharpeRatioState<T>,
        inputs: (&Array<T>, &()),
        output: &mut Array<T>,
        _ts: Instant,
        produced: (bool, bool),
    ) -> bool {
        let (_produced_data, produced_clock) = produced;
        if !produced_clock {
            return false;
        }
        let current = inputs.0[0];
        if current.is_nan() {
            return false;
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.sum_sq = state.sum_sq + r * r;
            state.count += 1;

            let n = T::from(state.count).unwrap();
            let mean = state.sum / n;
            let var = state.sum_sq / n - mean * mean;

            output[0] = if var > T::zero() {
                mean / var.sqrt()
            } else {
                T::nan()
            };
        }

        state.prev = current;
        true
    }
}

// ---------------------------------------------------------------------------
// Drawdown (single input, no clock)
// ---------------------------------------------------------------------------

/// Drawdown from the running maximum: `(current - max) / max` (non-positive).
#[derive(Clone)]
pub struct Drawdown<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Drawdown<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Drawdown<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DrawdownState<T: Scalar + Float> {
    running_max: T,
}

impl<T: Scalar + Float> Operator for Drawdown<T> {
    type State = DrawdownState<T>;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, _inputs: &Array<T>, _ts: Instant) -> (Self::State, Array<T>) {
        (
            DrawdownState {
                running_max: T::nan(),
            },
            Array::scalar(T::zero()),
        )
    }

    fn compute(
        state: &mut DrawdownState<T>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let current = inputs[0];
        if current.is_nan() {
            return false;
        }

        if state.running_max.is_nan() || current > state.running_max {
            state.running_max = current;
        }

        output[0] = if state.running_max > T::zero() {
            (current - state.running_max) / state.running_max
        } else {
            T::zero()
        };
        true
    }
}
