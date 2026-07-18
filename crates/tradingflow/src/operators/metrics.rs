//! Clock-driven since-inception financial metrics over the [`ArrayView`]
//! currency. The first four take `(clock, data)` and gate on the clock's notify
//! bit (emitting `notify = false` off-tick); `Drawdown`/`Turnover` are
//! single-input. The data input is a rank-`N` view (the leading element is read
//! for the scalar metrics; `Turnover` reads the whole weight vector); every
//! output is a rank-0 scalar view.
//!
//! The clock is the **leading** port, matching every other clock-gated operator
//! in the library ([`Clocked`](super::structural::Clocked),
//! [`ResampleClocked`](super::structural::ResampleClocked)), so the gated shapes stay
//! interchangeable.

use std::marker::PhantomData;

use num_traits::Float;

use crate::graph::Operator;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::ports::{ArrayPort, UnitPort};

// ---------------------------------------------------------------------------
// CompoundReturn
// ---------------------------------------------------------------------------

/// `(current / first)^(1/n) - 1` over clock ticks since inception.
#[derive(Clone)]
pub struct CompoundReturn<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> CompoundReturn<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for CompoundReturn<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`CompoundReturn`]: the accumulators plus the output
/// buffer.
pub struct CompoundReturnState<T: Scalar + Float> {
    first_value: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for CompoundReturn<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = CompoundReturnState<T>;

    fn init(self) -> Self::State {
        CompoundReturnState {
            first_value: T::nan(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((produced_clock, _), (_, data)): ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, 0>) {
        if init || !produced_clock {
            return (false, state.out.view());
        }
        let current = data.to_contiguous()[0];
        if current.is_nan() {
            return (false, state.out.view());
        }

        state.count += 1;

        if state.first_value.is_nan() {
            state.first_value = current;
            state.out[[]] = T::zero();
            return (true, state.out.view());
        }

        if state.first_value <= T::zero() || current <= T::zero() {
            state.out[[]] = T::nan();
            return (true, state.out.view());
        }

        let ratio = current / state.first_value;
        let n = T::from(state.count - 1).unwrap();
        if n <= T::zero() {
            state.out[[]] = T::zero();
        } else {
            state.out[[]] = ratio.powf(T::one() / n) - T::one();
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// AverageReturn
// ---------------------------------------------------------------------------

/// Arithmetic mean of period returns since inception.
#[derive(Clone)]
pub struct AverageReturn<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> AverageReturn<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for AverageReturn<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`AverageReturn`]: the accumulators plus the output buffer.
pub struct AverageReturnState<T: Scalar + Float> {
    prev: T,
    sum: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for AverageReturn<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = AverageReturnState<T>;

    fn init(self) -> Self::State {
        AverageReturnState {
            prev: T::nan(),
            sum: T::zero(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((produced_clock, _), (_, data)): ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, 0>) {
        if init || !produced_clock {
            return (false, state.out.view());
        }
        let current = data.to_contiguous()[0];
        if current.is_nan() {
            return (false, state.out.view());
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.count += 1;
            state.out[[]] = state.sum / T::from(state.count).unwrap();
        }

        state.prev = current;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Volatility
// ---------------------------------------------------------------------------

/// Population standard deviation of period returns since inception.
#[derive(Clone)]
pub struct Volatility<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Volatility<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Volatility<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Volatility`]: the accumulators plus the output buffer.
pub struct VolatilityState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for Volatility<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = VolatilityState<T>;

    fn init(self) -> Self::State {
        VolatilityState {
            prev: T::nan(),
            sum: T::zero(),
            sum_sq: T::zero(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((produced_clock, _), (_, data)): ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, 0>) {
        if init || !produced_clock {
            return (false, state.out.view());
        }
        let current = data.to_contiguous()[0];
        if current.is_nan() {
            return (false, state.out.view());
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.sum_sq = state.sum_sq + r * r;
            state.count += 1;
            let n = T::from(state.count).unwrap();
            let mean = state.sum / n;
            let var = state.sum_sq / n - mean * mean;
            state.out[[]] = if var > T::zero() {
                var.sqrt()
            } else {
                T::zero()
            };
        }

        state.prev = current;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// SharpeRatio
// ---------------------------------------------------------------------------

/// `mean(r) / std(r)` of period returns since inception.
#[derive(Clone)]
pub struct SharpeRatio<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> SharpeRatio<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for SharpeRatio<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`SharpeRatio`]: the accumulators plus the output buffer.
pub struct SharpeRatioState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for SharpeRatio<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = SharpeRatioState<T>;

    fn init(self) -> Self::State {
        SharpeRatioState {
            prev: T::nan(),
            sum: T::zero(),
            sum_sq: T::zero(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((produced_clock, _), (_, data)): ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, 0>) {
        if init || !produced_clock {
            return (false, state.out.view());
        }
        let current = data.to_contiguous()[0];
        if current.is_nan() {
            return (false, state.out.view());
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.sum_sq = state.sum_sq + r * r;
            state.count += 1;

            let n = T::from(state.count).unwrap();
            let mean = state.sum / n;
            let var = state.sum_sq / n - mean * mean;

            state.out[[]] = if var > T::zero() {
                mean / var.sqrt()
            } else {
                T::nan()
            };
        }

        state.prev = current;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Drawdown (single input, no clock)
// ---------------------------------------------------------------------------

/// Drawdown from the running maximum: `(current - max) / max` (non-positive).
#[derive(Clone)]
pub struct Drawdown<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Drawdown<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Drawdown<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Drawdown`]: the running maximum plus the output buffer.
pub struct DrawdownState<T: Scalar + Float> {
    running_max: T,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for Drawdown<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = DrawdownState<T>;

    fn init(self) -> Self::State {
        DrawdownState {
            running_max: T::nan(),
            out: Array::scalar(T::zero()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, data): (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, 0>) {
        if init {
            return (false, state.out.view());
        }
        let current = data.to_contiguous()[0];
        if current.is_nan() {
            return (false, state.out.view());
        }

        if state.running_max.is_nan() || current > state.running_max {
            state.running_max = current;
        }

        state.out[[]] = if state.running_max > T::zero() {
            (current - state.running_max) / state.running_max
        } else {
            T::zero()
        };
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Turnover (single input, no clock)
// ---------------------------------------------------------------------------

/// Per-update portfolio turnover: on each new weight vector, emits the L1 norm
/// of the change since the previous one, `Σᵢ |wₜ,ᵢ − wₜ₋₁,ᵢ|` (non-finite
/// weights treated as `0`). The first update is a warmup (caches the weights,
/// does not notify); every later update emits a finite scalar.
#[derive(Clone)]
pub struct Turnover<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Turnover<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Turnover<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Turnover`]: the previous (NaN-cleaned) weight vector, a
/// warmup flag, and the scalar output buffer.
pub struct TurnoverState<T: Scalar + Float> {
    prev: Vec<T>,
    initialized: bool,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for Turnover<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = TurnoverState<T>;

    fn init(self) -> Self::State {
        TurnoverState {
            prev: Vec::new(),
            initialized: false,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, data): (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, 0>) {
        if init {
            return (false, state.out.view());
        }
        let xs = data.to_contiguous();
        let cur: &[T] = &xs;
        let clean = |v: T| if v.is_finite() { v } else { T::zero() };

        if !state.initialized {
            state.prev = cur.iter().map(|&v| clean(v)).collect();
            state.initialized = true;
            return (false, state.out.view());
        }

        let mut turnover = T::zero();
        for (&raw, prev) in cur.iter().zip(state.prev.iter_mut()) {
            let c = clean(raw);
            turnover = turnover + (c - *prev).abs();
            *prev = c;
        }
        state.out[[]] = turnover;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Cumulative compounded return of a per-tick return stream.
pub fn compound_return<T: Scalar + Float, const N: usize>() -> CompoundReturn<T, N> {
    CompoundReturn::new()
}

/// Running mean of a per-tick return stream.
pub fn average_return<T: Scalar + Float, const N: usize>() -> AverageReturn<T, N> {
    AverageReturn::new()
}

/// Running standard deviation of a per-tick return stream.
pub fn volatility<T: Scalar + Float, const N: usize>() -> Volatility<T, N> {
    Volatility::new()
}

/// Running Sharpe ratio (mean / standard deviation) of a return stream.
pub fn sharpe_ratio<T: Scalar + Float, const N: usize>() -> SharpeRatio<T, N> {
    SharpeRatio::new()
}

/// Running drawdown from the running peak.
pub fn drawdown<T: Scalar + Float, const N: usize>() -> Drawdown<T, N> {
    Drawdown::new()
}

/// Per-tick turnover: the L1 change in a weight vector.
pub fn turnover<T: Scalar + Float, const N: usize>() -> Turnover<T, N> {
    Turnover::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pool::Pool;
    use crate::graph::typed::Builder;
    use crate::operators::constant::const_array;

    /// Turnover: warmup emits nothing; later updates emit the L1 change, with
    /// non-finite weights treated as zero.
    #[test]
    fn turnover_l1_change_with_nan_as_zero() {
        let mut b = Builder::new(Instant::MIN);
        let (w_cell, w) = b.source(const_array(Array::from_vec([5], vec![0.0_f64; 5])));
        let out = b.segment(Turnover::<f64, 1>::new(), w);
        let mut g = b.build();
        let mut pool = Pool::new(0);

        // Warmup: caches the weights, does not notify → output stays NaN.
        *g.state_mut(w_cell) = Array::from_vec([5], vec![0.2, 0.2, 0.2, 0.2, 0.2]);
        g.stabilize(&mut pool);
        assert!(
            g.view(out).as_slice().unwrap()[0].is_nan(),
            "warmup should not emit"
        );

        // L1 change: |0.4-0.2|+|0.1-0.2|+0+0+|0.1-0.2| = 0.2+0.1+0.1 = 0.4.
        *g.state_mut(w_cell) = Array::from_vec([5], vec![0.4, 0.1, 0.2, 0.2, 0.1]);
        g.stabilize(&mut pool);
        assert!((g.view(out).as_slice().unwrap()[0] - 0.4).abs() < 1e-12);

        // NaN treated as 0: stock 1 leaves (0.1 → 0), contributing its full 0.1;
        // |0.4-0.4|+|0-0.1|+0+0+|0.2-0.1| = 0.1 + 0.1 = 0.2.
        *g.state_mut(w_cell) = Array::from_vec([5], vec![0.4, f64::NAN, 0.2, 0.2, 0.2]);
        g.stabilize(&mut pool);
        assert!((g.view(out).as_slice().unwrap()[0] - 0.2).abs() < 1e-12);
    }
}
