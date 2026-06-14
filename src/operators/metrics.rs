//! Clock-driven since-inception financial metrics, implemented directly on
//! [`flowgraph::typed::Operator`]. The first four take `(data, clock)` inputs
//! and gate on the clock's notify bit (emitting `notify = false` off-tick);
//! `Drawdown` is single-input.

use std::marker::PhantomData;

use num_traits::Float;

use flowgraph::typed::{Operator, RefPort};

use crate::{Array, Scalar};

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

/// Runtime state for [`CompoundReturn`]: the accumulators plus the output
/// buffer.
pub struct CompoundReturnState<T: Scalar + Float> {
    first_value: T,
    count: usize,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for CompoundReturn<T> {
    type Inputs = (RefPort<Array<T>>, RefPort<()>);
    type Outputs = RefPort<Array<T>>;
    type State = CompoundReturnState<T>;

    fn init(self) -> CompoundReturnState<T> {
        CompoundReturnState {
            first_value: T::nan(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((_, data), (produced_clock, _)): ((bool, &'a Array<T>), (bool, &'a ())),
        state: &'b mut CompoundReturnState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            return (false, &state.out);
        }
        if !produced_clock {
            return (false, &state.out);
        }
        let current = data[0];
        if current.is_nan() {
            return (false, &state.out);
        }

        state.count += 1;

        if state.first_value.is_nan() {
            state.first_value = current;
            state.out[0] = T::zero();
            return (true, &state.out);
        }

        if state.first_value <= T::zero() || current <= T::zero() {
            state.out[0] = T::nan();
            return (true, &state.out);
        }

        let ratio = current / state.first_value;
        let n = T::from(state.count - 1).unwrap();
        if n <= T::zero() {
            state.out[0] = T::zero();
        } else {
            state.out[0] = ratio.powf(T::one() / n) - T::one();
        }
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, &'a Array<T>), (bool, &'a ())),
        state: &'b CompoundReturnState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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

/// Runtime state for [`AverageReturn`]: the accumulators plus the output
/// buffer.
pub struct AverageReturnState<T: Scalar + Float> {
    prev: T,
    sum: T,
    count: usize,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for AverageReturn<T> {
    type Inputs = (RefPort<Array<T>>, RefPort<()>);
    type Outputs = RefPort<Array<T>>;
    type State = AverageReturnState<T>;

    fn init(self) -> AverageReturnState<T> {
        AverageReturnState {
            prev: T::nan(),
            sum: T::zero(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((_, data), (produced_clock, _)): ((bool, &'a Array<T>), (bool, &'a ())),
        state: &'b mut AverageReturnState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            return (false, &state.out);
        }
        if !produced_clock {
            return (false, &state.out);
        }
        let current = data[0];
        if current.is_nan() {
            return (false, &state.out);
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.count += 1;
            state.out[0] = state.sum / T::from(state.count).unwrap();
        }

        state.prev = current;
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, &'a Array<T>), (bool, &'a ())),
        state: &'b AverageReturnState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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

/// Runtime state for [`Volatility`]: the accumulators plus the output buffer.
pub struct VolatilityState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Volatility<T> {
    type Inputs = (RefPort<Array<T>>, RefPort<()>);
    type Outputs = RefPort<Array<T>>;
    type State = VolatilityState<T>;

    fn init(self) -> VolatilityState<T> {
        VolatilityState {
            prev: T::nan(),
            sum: T::zero(),
            sum_sq: T::zero(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((_, data), (produced_clock, _)): ((bool, &'a Array<T>), (bool, &'a ())),
        state: &'b mut VolatilityState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            return (false, &state.out);
        }
        if !produced_clock {
            return (false, &state.out);
        }
        let current = data[0];
        if current.is_nan() {
            return (false, &state.out);
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.sum_sq = state.sum_sq + r * r;
            state.count += 1;
            let n = T::from(state.count).unwrap();
            let mean = state.sum / n;
            let var = state.sum_sq / n - mean * mean;
            state.out[0] = if var > T::zero() { var.sqrt() } else { T::zero() };
        }

        state.prev = current;
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, &'a Array<T>), (bool, &'a ())),
        state: &'b VolatilityState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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

/// Runtime state for [`SharpeRatio`]: the accumulators plus the output buffer.
pub struct SharpeRatioState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for SharpeRatio<T> {
    type Inputs = (RefPort<Array<T>>, RefPort<()>);
    type Outputs = RefPort<Array<T>>;
    type State = SharpeRatioState<T>;

    fn init(self) -> SharpeRatioState<T> {
        SharpeRatioState {
            prev: T::nan(),
            sum: T::zero(),
            sum_sq: T::zero(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((_, data), (produced_clock, _)): ((bool, &'a Array<T>), (bool, &'a ())),
        state: &'b mut SharpeRatioState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            return (false, &state.out);
        }
        if !produced_clock {
            return (false, &state.out);
        }
        let current = data[0];
        if current.is_nan() {
            return (false, &state.out);
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.sum_sq = state.sum_sq + r * r;
            state.count += 1;

            let n = T::from(state.count).unwrap();
            let mean = state.sum / n;
            let var = state.sum_sq / n - mean * mean;

            state.out[0] = if var > T::zero() {
                mean / var.sqrt()
            } else {
                T::nan()
            };
        }

        state.prev = current;
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, &'a Array<T>), (bool, &'a ())),
        state: &'b SharpeRatioState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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

/// Runtime state for [`Drawdown`]: the running maximum plus the output buffer.
pub struct DrawdownState<T: Scalar + Float> {
    running_max: T,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Drawdown<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = DrawdownState<T>;

    fn init(self) -> DrawdownState<T> {
        DrawdownState {
            running_max: T::nan(),
            out: Array::scalar(T::zero()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, data): (bool, &'a Array<T>),
        state: &'b mut DrawdownState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            return (false, &state.out);
        }
        let current = data[0];
        if current.is_nan() {
            return (false, &state.out);
        }

        if state.running_max.is_nan() || current > state.running_max {
            state.running_max = current;
        }

        state.out[0] = if state.running_max > T::zero() {
            (current - state.running_max) / state.running_max
        } else {
            T::zero()
        };
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b DrawdownState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// Turnover (single input, no clock)
// ---------------------------------------------------------------------------

/// Per-update portfolio turnover: on each new weight vector, emits the L1 norm
/// of the change since the previous one, `Σᵢ |wₜ,ᵢ − wₜ₋₁,ᵢ|` (non-finite
/// weights treated as `0`, so a stock entering/leaving contributes its full
/// weight). The first update is a warmup (caches the weights, does not notify);
/// every later update emits a finite scalar. `Record` it for a turnover series,
/// or feed `RollingMean` for an average. For long-only books summing to 1 the
/// value lies in `[0, 2]`.
#[derive(Clone)]
pub struct Turnover<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Turnover<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Turnover<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Turnover`]: the previous (NaN-cleaned) weight vector, a
/// warmup flag, and the scalar output buffer.
pub struct TurnoverState<T: Scalar + Float> {
    prev: Vec<T>,
    initialized: bool,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Turnover<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = TurnoverState<T>;

    fn init(self) -> TurnoverState<T> {
        TurnoverState {
            prev: Vec::new(),
            initialized: false,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, data): (bool, &'a Array<T>),
        state: &'b mut TurnoverState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            return (false, &state.out);
        }
        let cur = data.as_slice();
        let clean = |v: T| if v.is_finite() { v } else { T::zero() };

        if !state.initialized {
            state.prev = cur.iter().map(|&v| clean(v)).collect();
            state.initialized = true;
            return (false, &state.out);
        }

        let mut turnover = T::zero();
        for i in 0..cur.len() {
            let c = clean(cur[i]);
            turnover = turnover + (c - state.prev[i]).abs();
            state.prev[i] = c;
        }
        state.out[0] = turnover;
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b TurnoverState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowgraph::core::Pool;
    use flowgraph::typed::{Graph, GraphBuilder, RefSource};

    /// Turnover: warmup emits nothing; later updates emit the L1 change, with
    /// non-finite weights treated as zero.
    #[test]
    fn turnover_l1_change_with_nan_as_zero() {
        let mut b = GraphBuilder::new();
        let w = b.push_source(RefSource::new(Array::from_vec(&[5], vec![0.0_f64; 5])));
        let out = b.push(Turnover::<f64>::new(), *w);
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        // Warmup: caches the weights, does not notify → output stays NaN.
        *g.state_mut(w) = Array::from_vec(&[5], vec![0.2, 0.2, 0.2, 0.2, 0.2]);
        g.stabilize(&mut pool);
        assert!(g.ref_view(out).as_slice()[0].is_nan(), "warmup should not emit");

        // L1 change: |0.4-0.2|+|0.1-0.2|+0+0+|0.1-0.2| = 0.2+0.1+0.1 = 0.4.
        *g.state_mut(w) = Array::from_vec(&[5], vec![0.4, 0.1, 0.2, 0.2, 0.1]);
        g.stabilize(&mut pool);
        assert!((g.ref_view(out).as_slice()[0] - 0.4).abs() < 1e-12);

        // NaN treated as 0: stock 1 leaves (0.1 → 0), contributing its full 0.1;
        // |0.4-0.4|+|0-0.1|+0+0+|0.2-0.1| = 0.1 + 0.1 = 0.2.
        *g.state_mut(w) = Array::from_vec(&[5], vec![0.4, f64::NAN, 0.2, 0.2, 0.2]);
        g.stabilize(&mut pool);
        assert!((g.ref_view(out).as_slice()[0] - 0.2).abs() < 1e-12);
    }
}
