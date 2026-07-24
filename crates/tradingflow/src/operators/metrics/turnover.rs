//! Per-update portfolio turnover (single input, no clock).

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

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

/// Non-finite weights count as zero exposure.
fn clean<T: Scalar + Float>(v: T) -> T {
    if v.is_finite() { v } else { T::zero() }
}

impl<T: Scalar + Float, const N: usize> Operator for Turnover<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = TurnoverState<T>;

    fn init(self, _: (bool, ArrayView<'_, T, N>)) -> Self::State {
        TurnoverState {
            prev: Vec::new(),
            initialized: false,
            out: Array::scalar(T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, data): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, 0>) {
        let cur = data.to_contiguous();

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
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }
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
    use crate::operators::array;

    /// Turnover: warmup emits nothing; later updates emit the L1 change, with
    /// non-finite weights treated as zero.
    #[test]
    fn turnover_l1_change_with_nan_as_zero() {
        let mut b = Builder::new();
        let (w_cell, w) = b.source(array::from_parts([5], vec![0.0_f64; 5].into()));
        let out = b.segment(Turnover::<f64, 1>::new(), w);
        let mut g = b.build();
        let mut pool = Pool::new(0);

        // Warmup: caches the weights, does not notify → output stays NaN.
        *g.state_mut(w_cell) = Array::from_parts([5], vec![0.2, 0.2, 0.2, 0.2, 0.2].into());
        g.stabilize(&mut pool, &Instant::MIN);
        assert!(
            g.view(out).as_slice().unwrap()[0].is_nan(),
            "warmup should not emit"
        );

        // L1 change: |0.4-0.2|+|0.1-0.2|+0+0+|0.1-0.2| = 0.2+0.1+0.1 = 0.4.
        *g.state_mut(w_cell) = Array::from_parts([5], vec![0.4, 0.1, 0.2, 0.2, 0.1].into());
        g.stabilize(&mut pool, &Instant::MIN);
        assert!((g.view(out).as_slice().unwrap()[0] - 0.4).abs() < 1e-12);

        // NaN treated as 0: stock 1 leaves (0.1 → 0), contributing its full 0.1;
        // |0.4-0.4|+|0-0.1|+0+0+|0.2-0.1| = 0.1 + 0.1 = 0.2.
        *g.state_mut(w_cell) = Array::from_parts([5], vec![0.4, f64::NAN, 0.2, 0.2, 0.2].into());
        g.stabilize(&mut pool, &Instant::MIN);
        assert!((g.view(out).as_slice().unwrap()[0] - 0.2).abs() < 1e-12);
    }
}
