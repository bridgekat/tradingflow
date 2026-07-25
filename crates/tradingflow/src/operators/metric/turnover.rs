use num_traits::Float;
use std::marker::PhantomData;
use tradingflow_data::Layout;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::{Operator, Segment};
use crate::ports::ArrayPort;

/// Operator signature for [`turnover`].
pub struct Turnover<T: Scalar + Float> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float> Turnover<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Turnover<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Turnover`].
pub struct TurnoverState<T: Scalar + Float> {
    prev: Vec<T>,
    out: Array<T, 0>,
}

impl<T: Scalar + Float> Operator for Turnover<T> {
    type Inputs = ArrayPort<T, 1>;
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = TurnoverState<T>;

    fn init(self, (_, weights): (bool, ArrayView<'_, T, 1>)) -> Self::State {
        let len = weights.layout().len();
        TurnoverState {
            prev: vec![T::zero(); len],
            out: Array::scalar(T::zero()),
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, 1>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, 0>) {
        (false, state.out.view())
    }

    fn compute<'a, 'b: 'a>(
        (_, weights): (bool, ArrayView<'a, T, 1>),
        state: &'b mut Self::State,
        _: &Self::Context,
    ) -> (bool, ArrayView<'a, T, 0>) {
        assert!(
            weights.layout().len() == state.prev.len(),
            "turnover: input length mismatch"
        );
        for w in weights.iter() {
            assert!(w.is_finite(), "turnover: input values must be finite");
        }
        crate::data::array::for_each(weights, |j, &w| {
            *state.out = *state.out + (w - state.prev[j]).abs();
            state.prev[j] = w;
        });
        (true, state.out.view())
    }
}

/// Cumulative turnover: the sum of L1 norm of changes in a weight vector.
pub fn turnover<T: Scalar + Float>()
-> impl Segment<Inputs = ArrayPort<T, 1>, Outputs = ArrayPort<T, 0>, Context = Instant> {
    Turnover::new()
}
