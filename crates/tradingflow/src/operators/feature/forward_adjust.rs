use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{self, Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, SignalPort};

/// Operator signature for [`forward_adjust`].
pub struct ForwardAdjust<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> ForwardAdjust<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for ForwardAdjust<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`ForwardAdjust`].
pub struct ForwardAdjustState<T: Scalar + Float, const N: usize> {
    prev_closes: Array<T, N>,
    multipliers: Array<T, N>,
    adj_closes: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Segment for ForwardAdjust<T, N> {
    type Inputs = (
        (SignalPort<N>, ArrayPort<T, N>),
        (SignalPort<N>, ArrayPort<T, N>, ArrayPort<T, N>),
    );
    type Outputs = (ArrayPort<T, N>, ArrayPort<T, N>);
    type Context = Instant;
    type State = ForwardAdjustState<T, N>;

    fn init(
        self,
        ((close_signals, closes), (div_signals, share_divs, cash_divs)): (
            (ArrayView<'_, bool, N>, ArrayView<'_, T, N>),
            (
                ArrayView<'_, bool, N>,
                ArrayView<'_, T, N>,
                ArrayView<'_, T, N>,
            ),
        ),
    ) -> Self::State {
        let _ = data::array::broadcast_to(close_signals, closes.extents());
        let _ = data::array::broadcast_to(div_signals, closes.extents());
        let _ = data::array::broadcast_to(share_divs, closes.extents());
        let _ = data::array::broadcast_to(cash_divs, closes.extents());
        ForwardAdjustState {
            prev_closes: Array::full(closes.extents(), T::nan()),
            multipliers: Array::full(closes.extents(), T::one()),
            adj_closes: Array::full(closes.extents(), T::nan()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (
            (ArrayView<'_, bool, N>, ArrayView<'_, T, N>),
            (
                ArrayView<'_, bool, N>,
                ArrayView<'_, T, N>,
                ArrayView<'_, T, N>,
            ),
        ),
        state: &'b mut Self::State,
    ) -> (ArrayView<'a, T, N>, ArrayView<'a, T, N>) {
        (state.multipliers.view(), state.adj_closes.view())
    }

    fn compute<'a, 'b: 'a>(
        ((close_signals, closes), (div_signals, share_divs, cash_divs)): (
            (ArrayView<'_, bool, N>, ArrayView<'_, T, N>),
            (
                ArrayView<'_, bool, N>,
                ArrayView<'_, T, N>,
                ArrayView<'_, T, N>,
            ),
        ),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (ArrayView<'a, T, N>, ArrayView<'a, T, N>) {
        let close_signals = data::array::broadcast_to(close_signals, closes.extents());
        let div_signals = data::array::broadcast_to(div_signals, closes.extents());
        let share_divs = data::array::broadcast_to(share_divs, closes.extents());
        let cash_divs = data::array::broadcast_to(cash_divs, closes.extents());
        for (
            ((((((&close, &close_c), &div_c), &share_div), &cash_div), prev_close), multiplier),
            adj_close,
        ) in closes
            .iter()
            .zip(close_signals.iter())
            .zip(div_signals.iter())
            .zip(share_divs.iter())
            .zip(cash_divs.iter())
            .zip(state.prev_closes.data_mut())
            .zip(state.multipliers.data_mut())
            .zip(state.adj_closes.data_mut())
        {
            if div_c {
                assert!(
                    share_div.is_finite() && cash_div.is_finite(),
                    "forward_adjust: dividends must be finite"
                );
                *multiplier = *multiplier * (T::one() + share_div);
                if !prev_close.is_nan() {
                    assert!(
                        *prev_close > cash_div,
                        "forward_adjust: cash dividends must be less than previous close"
                    );
                    *multiplier = *multiplier * (T::one() + cash_div / (*prev_close - cash_div));
                }
            }
            if close_c {
                assert!(close.is_finite(), "forward_adjust: prices must be finite");
                *prev_close = close;
                *adj_close = close * *multiplier;
            }
        }
        (state.multipliers.view(), state.adj_closes.view())
    }
}

/// Forward-adjusts stock prices for dividends.
///
/// Inputs:
///
/// - `(close_signals, closes)`: closing prices, one signal per element
///   per trading day.
/// - `(div_signals, share_divs, cash_divs)`: share and cash dividends per
///   share, one signal per element per dividend event.
///
/// All extents must be broadcastable to `closes`.
///
/// Outputs:
///
/// - `multipliers`: elementwise cumulative forward-adjustment multipliers.
/// - `adj_closes`: elementwise forward-adjusted closing prices.
#[allow(clippy::type_complexity)]
pub fn forward_adjust<T: Scalar + Float, const N: usize>() -> impl Segment<
    Inputs = (
        (SignalPort<N>, ArrayPort<T, N>),
        (SignalPort<N>, ArrayPort<T, N>, ArrayPort<T, N>),
    ),
    Outputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Context = Instant,
> {
    ForwardAdjust::new()
}
