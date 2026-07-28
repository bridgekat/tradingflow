use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{self, Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ClockArrayPort};

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
    prev: Array<(T, u16, u16), N>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Segment for Annualize<T, N> {
    type Inputs = (
        ClockArrayPort<N>,
        ArrayPort<T, N>,
        ArrayPort<u16, N>,
        ArrayPort<u16, N>,
    );
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = AnnualizeState<T, N>;

    fn init(
        self,
        (clocks, values, years, days): (
            ArrayView<'_, bool, N>,
            ArrayView<'_, T, N>,
            ArrayView<'_, u16, N>,
            ArrayView<'_, u16, N>,
        ),
    ) -> Self::State {
        let _ = data::array::broadcast_to(clocks, values.extents());
        let _ = data::array::broadcast_to(years, values.extents());
        let _ = data::array::broadcast_to(days, values.extents());
        AnnualizeState {
            prev: Array::full(values.extents(), (T::nan(), 0, 0)),
            out: Array::full(values.extents(), T::nan()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (
            ArrayView<'_, bool, N>,
            ArrayView<'_, T, N>,
            ArrayView<'_, u16, N>,
            ArrayView<'_, u16, N>,
        ),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (clocks, values, years, days): (
            ArrayView<'_, bool, N>,
            ArrayView<'_, T, N>,
            ArrayView<'_, u16, N>,
            ArrayView<'_, u16, N>,
        ),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        let clocks = data::array::broadcast_to(clocks, values.extents());
        let years = data::array::broadcast_to(years, values.extents());
        let days = data::array::broadcast_to(days, values.extents());
        for (((((&clock, &value), &year), &day), (prev_value, prev_year, prev_day)), out) in clocks
            .iter()
            .zip(values.iter())
            .zip(years.iter())
            .zip(days.iter())
            .zip(state.prev.data_mut())
            .zip(state.out.data_mut())
        {
            // A NaN value under a set clock is a missing field within the
            // fired row: skip it and bridge to the next report (the day
            // delta then spans the gap).
            if clock && !value.is_nan() {
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
            }
        }
        state.out.view()
    }
}

/// Converts YTD-cumulative arrays into annualized values.
///
/// Inputs:
///
/// - `clock`: emits one clock signal per data point.
/// - `ytd_values`: the YTD-cumulative values of the current data point.
/// - `year`: the year of the current data point.
/// - `day`: the day-of-year of the current data point.
///
/// All extents must be broadcastable to `ytd_values`.
///
/// Outputs:
///
/// - `annual_values`: elementwise annualized `ytd_values`.
pub fn annualize<T: Scalar + Float, const N: usize>() -> impl Segment<
    Inputs = (
        ClockArrayPort<N>,
        ArrayPort<T, N>,
        ArrayPort<u16, N>,
        ArrayPort<u16, N>,
    ),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    Annualize::new()
}
