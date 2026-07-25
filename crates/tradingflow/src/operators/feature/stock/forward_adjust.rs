use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{self, Array, ArrayView, Instant, Scalar};
use crate::graph::{Operator, Segment};
use crate::ports::{ArrayPort, event_or};

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

impl<T: Scalar + Float, const N: usize> Operator for ForwardAdjust<T, N> {
    type Inputs = (ArrayPort<T, N>, ArrayPort<T, N>, ArrayPort<T, N>);
    type Outputs = (ArrayPort<T, N>, ArrayPort<T, N>);
    type Context = Instant;
    type State = ForwardAdjustState<T, N>;

    fn init(
        self,
        (closes, share_divs, cash_divs): (
            (bool, ArrayView<'_, T, N>),
            (bool, ArrayView<'_, T, N>),
            (bool, ArrayView<'_, T, N>),
        ),
    ) -> Self::State {
        let nan = T::nan();
        let closes = event_or(closes, &nan);
        let _ = data::array::broadcast_to(event_or(share_divs, &nan), closes.extents());
        let _ = data::array::broadcast_to(event_or(cash_divs, &nan), closes.extents());
        ForwardAdjustState {
            prev_closes: Array::full(closes.extents(), T::nan()),
            multipliers: Array::full(closes.extents(), T::one()),
            adj_closes: Array::full(closes.extents(), T::nan()),
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (
            (bool, ArrayView<'_, T, N>),
            (bool, ArrayView<'_, T, N>),
            (bool, ArrayView<'_, T, N>),
        ),
        state: &'b mut Self::State,
    ) -> ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)) {
        (
            (false, state.multipliers.view()),
            (false, state.adj_closes.view()),
        )
    }

    fn compute<'a, 'b: 'a>(
        (closes, share_divs, cash_divs): (
            (bool, ArrayView<'_, T, N>),
            (bool, ArrayView<'_, T, N>),
            (bool, ArrayView<'_, T, N>),
        ),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)) {
        let nan = T::nan();
        let closes = event_or(closes, &nan);
        let share_divs = data::array::broadcast_to(event_or(share_divs, &nan), closes.extents());
        let cash_divs = data::array::broadcast_to(event_or(cash_divs, &nan), closes.extents());
        for ((((&close, &share_div), &cash_div), prev_close), multiplier) in closes
            .iter()
            .zip(share_divs.iter())
            .zip(cash_divs.iter())
            .zip(state.prev_closes.data_mut())
            .zip(state.multipliers.data_mut())
        {
            if !share_div.is_nan() {
                *multiplier = *multiplier * (T::one() + share_div);
            }
            if !cash_div.is_nan() && !prev_close.is_nan() {
                assert!(
                    *prev_close > cash_div,
                    "forward_adjust: cash dividends must be less than previous close"
                );
                *multiplier = *multiplier * (T::one() + cash_div / (*prev_close - cash_div));
            }
            if !close.is_nan() {
                *prev_close = close;
            }
        }
        for ((&close, &multiplier), adj_close) in closes
            .iter()
            .zip(state.multipliers.iter())
            .zip(state.adj_closes.data_mut())
        {
            if !close.is_nan() {
                *adj_close = close * multiplier;
            } else {
                *adj_close = nan;
            }
        }
        (
            (true, state.multipliers.view()),
            (true, state.adj_closes.view()),
        )
    }
}

/// Forward-adjusts stock prices for dividends.
///
/// Inputs:
///
/// - `closes`: the closing prices of each stock. Notifies once per trading day.
/// - `share_divs`: share dividends per stock. Extents must be broadcastable to
///   `closes`. Notifies once per dividend event (extra zeros are allowed).
/// - `cash_divs`: cash dividends per stock. Extents must be broadcastable to
///   `closes`. Notifies once per dividend event (extra zeros are allowed).
///
/// Outputs:
///
/// - `multipliers`: cumulative forward-adjustment multipliers for each stock.
///   Extents are the same as `closes`. Notifies on change.
/// - `adj_closes`: forward-adjusted closing prices for each stock. Extents are
///   the same as `closes`. Notifies once per trading day.
#[allow(clippy::type_complexity)]
pub fn forward_adjust<T: Scalar + Float, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Context = Instant,
> {
    ForwardAdjust::new()
}
