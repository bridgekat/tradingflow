//! `ResampleClocked` — the unit-clock counterpart of `ResampleView`.

use std::marker::PhantomData;

use super::resample_view::ResampleViewState;
use crate::data::{ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, UnitPort};

/// Clock-gated **view** passthrough whose clock is a unit (`RefPort<()>`) clock
/// source (e.g. a rebalance [`pulse`](crate::sources::basic::pulse): re-emits
/// the rank-`N` data view on every clock tick. The unit-clock counterpart of
/// [`ResampleView`](super::ResampleView).
#[derive(Clone)]
pub struct ResampleClocked<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> ResampleClocked<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for ResampleClocked<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Operator for ResampleClocked<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ResampleViewState<T, N>;

    fn init(self, (_, (_, x)): ((bool, ()), (bool, ArrayView<'_, T, N>))) -> Self::State {
        ResampleViewState { out: x.to_array() }
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ()), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), (_, x)): ((bool, ()), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        if clock_fired {
            state.out = x.to_array();
            return (true, state.out.view());
        }
        (false, state.out.view())
    }
}

/// Re-emit an array view on every tick of a leading *clock* pulse.
pub fn resample_clocked<T: Scalar, const N: usize>() -> ResampleClocked<T, N> {
    ResampleClocked::new()
}
