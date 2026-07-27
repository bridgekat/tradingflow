use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{self, Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::ArrayPort;

/// Operator signature for [`annualize`].
pub struct Annualize<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Annualize<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Annualize<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Annualize`].
pub struct AnnualizeState<T: Scalar + Float, const N: usize> {
    prev: Array<(T, i32, i32), N>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Segment for Annualize<T, N> {
    type Inputs = (ArrayPort<T, N>, ArrayPort<i32, N>, ArrayPort<i32, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = AnnualizeState<T, N>;

    fn init(
        self,
        (values, years, days): (
            ArrayView<'_, T, N>,
            ArrayView<'_, i32, N>,
            ArrayView<'_, i32, N>,
        ),
    ) -> Self::State {
        let _ = data::array::broadcast_to(years, values.extents());
        let _ = data::array::broadcast_to(days, values.extents());
        AnnualizeState {
            prev: Array::full(values.extents(), (T::nan(), 0, 0)),
            out: Array::full(values.extents(), T::nan()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (
            ArrayView<'_, T, N>,
            ArrayView<'_, i32, N>,
            ArrayView<'_, i32, N>,
        ),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (values, years, days): (
            ArrayView<'_, T, N>,
            ArrayView<'_, i32, N>,
            ArrayView<'_, i32, N>,
        ),
        state: &'b mut Self::State,
        _: &Self::Context,
    ) -> ArrayView<'a, T, N> {
        let years = data::array::broadcast_to(years, values.extents());
        let days = data::array::broadcast_to(days, values.extents());
        for ((((&value, &year), &day), (prev_value, prev_year, prev_day)), out) in values
            .iter()
            .zip(years.iter())
            .zip(days.iter())
            .zip(state.prev.data_mut())
            .zip(state.out.data_mut())
        {
            if !value.is_nan() {
                if !prev_value.is_nan() {
                    assert!(
                        year >= *prev_year,
                        "annualize: year must be monotonic, got {year} after {prev_year}"
                    );
                    assert!(
                        year != *prev_year || day >= *prev_day,
                        "annualize: day must be monotonic within a year, got {day} after {prev_day} in year {year}"
                    );
                }
                let is_new_year = prev_value.is_nan() || year != *prev_year;
                let (delta_value, delta_day) = if is_new_year {
                    (value, day)
                } else {
                    (value - *prev_value, day - *prev_day)
                };
                if delta_day > 0 {
                    *out = delta_value * T::from(365).unwrap() / T::from(delta_day).unwrap();
                }
                *prev_value = value;
                *prev_year = year;
                *prev_day = day;
            } else {
                *out = T::nan();
            }
        }
        state.out.view()
    }
}

/// Converts YTD-cumulative arrays into annualized values.
///
/// Inputs:
///
/// - `ytd_values` (event, one per data point): the YTD-cumulative values of
///   the data point.
/// - `year` (state): the year of the current data point. Extents must be
///   broadcastable to `ytd_values`.
/// - `day` (state): the day-of-year of the current data point. Extents must be
///   broadcastable to `ytd_values`.
///
/// Outputs:
///
/// - `annual_values` (event, one per data point): elementwise annualized
///   `ytd_values`, with the same extents.
pub fn annualize<T: Scalar + Float, const N: usize>() -> impl Segment<
    Inputs = (ArrayPort<T, N>, ArrayPort<i32, N>, ArrayPort<i32, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    Annualize::new()
}
