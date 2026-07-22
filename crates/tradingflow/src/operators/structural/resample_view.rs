//! `ResampleView` — clock-gated view passthrough driven by a data-view pulse.
//!
//! Also home to [`ResampleViewState`], which
//! [`ResampleClocked`](super::ResampleClocked) shares.

use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// State shared by the view-currency resamplers: the last data view
/// materialized into an owned buffer, so it survives between clock ticks while
/// the upstream view's storage may change.
pub struct ResampleViewState<T: Scalar, const N: usize> {
    pub(super) out: Array<T, N>,
}

/// Clock-gated **view** passthrough whose clock is another data view (only the
/// clock's notify bit is consulted): re-emits the rank-`N` data view on every
/// tick of the rank-1 clock view, holding the last value in between — e.g.
/// resample a feature panel onto the daily-close pulse. Stays in the
/// [`ArrayView`] currency end-to-end.
#[derive(Clone)]
pub struct ResampleView<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> ResampleView<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for ResampleView<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// The auto-gate routes to `compute` when the clock OR the data notifies, but
// `compute` re-emits only on a clock tick and otherwise returns the cached view
// unchanged — identical to `passthrough`. So a data-notify-without-clock is a
// no-op whether it lands in `compute` or `passthrough`.
impl<T: Scalar, const N: usize> Operator for ResampleView<T, N> {
    type Inputs = (ArrayPort<T, 1>, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ResampleViewState<T, N>;

    fn init(
        self,
        (_, (_, x)): ((bool, ArrayView<'_, T, 1>), (bool, ArrayView<'_, T, N>)),
    ) -> Self::State {
        ResampleViewState { out: x.to_array() }
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ArrayView<'a, T, 1>), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), (_, x)): ((bool, ArrayView<'a, T, 1>), (bool, ArrayView<'a, T, N>)),
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

/// Re-emit an array view on every tick of a leading *view* pulse.
pub fn resample_view<T: Scalar, const N: usize>() -> ResampleView<T, N> {
    ResampleView::new()
}
