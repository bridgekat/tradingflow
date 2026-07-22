//! `Count` — stateful per-tick counter (anti-corruption demonstrator).

use crate::data::{Array, ArrayView, Instant};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Increments a counter every time it runs and emits the running count (a
/// rank-0 scalar). Used to prove gating advances state only when an input
/// actually notifies.
pub struct Count<const N: usize>;

/// Runtime state for [`Count`]: the counter plus the scalar output buffer.
pub struct CountState {
    count: i64,
    out: Array<f64, 0>,
}

impl<const N: usize> Operator for Count<N> {
    type Inputs = ArrayPort<f64, N>;
    type Outputs = ArrayPort<f64, 0>;
    type Context = Instant;
    type State = CountState;

    fn init(self, _: (bool, ArrayView<'_, f64, N>)) -> Self::State {
        CountState {
            count: 0,
            out: Array::scalar(0.0),
        }
    }

    fn compute<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, f64, 0>) {
        state.count += 1;
        state.out.data_mut()[0] = state.count as f64;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, f64, 0>) {
        (false, state.out.view())
    }
}

/// Count the notified ticks seen so far.
pub fn count<const N: usize>() -> Count<N> {
    Count
}
