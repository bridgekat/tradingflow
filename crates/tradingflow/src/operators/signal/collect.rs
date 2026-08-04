use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar, array};
use crate::graph::Operator;
use crate::ports::{ArrayPort, SignalPort};

/// Operator signature for [`collect`].
pub struct Collect<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Collect<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Collect<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Collect`].
pub struct CollectState<T: Scalar + Float, const N: usize> {
    pending: Array<bool, N>,
    latest: Array<T, N>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Collect<T, N> {
    type Inputs = (SignalPort<N>, ArrayPort<T, N>, SignalPort<0>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = CollectState<T, N>;

    fn init(
        self,
        (signals, values, _): (
            ArrayView<'_, bool, N>,
            ArrayView<'_, T, N>,
            ArrayView<'_, bool, 0>,
        ),
    ) -> Self::State {
        let _ = array::broadcast_to(signals, values.extents());
        CollectState {
            pending: Array::zeros(values.extents()),
            latest: Array::full(values.extents(), T::nan()),
            out: Array::full(values.extents(), T::nan()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (
            ArrayView<'a, bool, N>,
            ArrayView<'a, T, N>,
            ArrayView<'a, bool, 0>,
        ),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (signals, values, clock): (
            ArrayView<'a, bool, N>,
            ArrayView<'a, T, N>,
            ArrayView<'a, bool, 0>,
        ),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        let signals = array::broadcast_to(signals, values.extents());
        for (((&set, &value), pending), latest) in signals
            .iter()
            .zip(values.iter())
            .zip(state.pending.data_mut())
            .zip(state.latest.data_mut())
        {
            if set {
                *pending = true;
                *latest = value;
            }
        }
        if *clock {
            for ((out, pending), &latest) in state
                .out
                .data_mut()
                .iter_mut()
                .zip(state.pending.data_mut())
                .zip(state.latest.data())
            {
                *out = if *pending { latest } else { T::nan() };
                *pending = false;
            }
        }
        state.out.view()
    }
}

/// Collects element-wise events into a single event on a clock signal. Each
/// element's latest event value is latched as it arrives; on every `clock`
/// pulse, the pending batch is emitted: `values` where an event was latched
/// since the previous `clock` pulse, `NaN` elsewhere.
///
/// # Inputs
///
/// - `signals`: element-wise event signals. Extents must be broadcastable to
///   `values`.
/// - `values`: element-wise event values.
/// - `clock`: a single clock pulse to emit the pending batch.
///
/// # Outputs
///
/// - `values`: the pending batch, updated on `clock == true`.
pub fn collect<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (SignalPort<N>, ArrayPort<T, N>, SignalPort<0>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    Collect::new()
}
