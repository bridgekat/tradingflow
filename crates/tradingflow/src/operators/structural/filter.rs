//! `Filter` — whole-array gate by predicate (the cutoff operator).

use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Passes the input through when the predicate holds, else drops it (emits
/// `notify = false` → downstream gated off, previous value retained).
pub struct Filter<T: Scalar, F, const N: usize>(pub F, pub PhantomData<T>);

/// Runtime state for [`Filter`]: the predicate plus the retained output.
pub struct FilterState<T: Scalar, F, const N: usize> {
    predicate: F,
    out: Array<T, N>,
}

impl<T: Scalar, F, const N: usize> Operator for Filter<T, F, N>
where
    F: for<'x> Fn(ArrayView<'x, T, N>) -> bool + Send + Sync + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = FilterState<T, F, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        FilterState {
            predicate: self.0,
            out: x.to_array(),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        if (state.predicate)(x) {
            state.out.assign(x);
            (true, state.out.view())
        } else {
            (false, state.out.view())
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// Pass the input through iff `predicate` holds, else drop the tick (emitting
/// `notify = false`). The cutoff operator: a dropped tick suppresses every
/// downstream side effect, including a [`Record`](crate::operators::series::Record) append.
pub fn filter<T: Scalar, F, const N: usize>(predicate: F) -> Filter<T, F, N> {
    Filter(predicate, PhantomData)
}
