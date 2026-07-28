use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Retention, Scalar, Series, SeriesView};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ClockPort, SeriesPort};

/// Operator signature for [`record_on`] etc.
pub struct RecordOn<T: Scalar, const N: usize> {
    retention: Retention,
    delayed: bool,
    _marker: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> RecordOn<T, N> {
    pub fn new(retention: Retention, delayed: bool) -> Self {
        Self {
            retention,
            delayed,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for [`RecordOn`].
pub struct RecordState<T: Scalar, const N: usize> {
    retention: Retention,
    delayed: bool,
    out: Series<T, N>,
}

impl<T: Scalar + Float, const N: usize> RecordState<T, N> {
    fn push(&mut self, now: &Instant, a: ArrayView<'_, T, N>) {
        if self.delayed {
            self.maybe_trim();
        }
        self.out.push(*now, a);
        if !self.delayed {
            self.maybe_trim();
        }
    }

    fn maybe_trim(&mut self) {
        if let Some(now) = self.out.instants().last() {
            let count = self.retention.trim_count(self.out.view(), now);
            if count > 0 && count * 2 >= self.out.len() {
                self.out.trim(count);
            }
        };
    }
}

impl<T: Scalar + Float, const N: usize> Segment for RecordOn<T, N> {
    type Inputs = (ClockPort, ArrayPort<T, N>);
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = RecordState<T, N>;

    fn init(self, (_, a): (ArrayView<'_, bool, 0>, ArrayView<'_, T, N>)) -> RecordState<T, N> {
        RecordState {
            retention: self.retention,
            delayed: self.delayed,
            out: Series::new(a.extents()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, T, N>),
        state: &'b mut RecordState<T, N>,
    ) -> SeriesView<'a, T, N> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (clock, a): (ArrayView<'a, bool, 0>, ArrayView<'a, T, N>),
        state: &'b mut RecordState<T, N>,
        now: &Instant,
    ) -> SeriesView<'a, T, N> {
        if *clock {
            state.push(now, a);
        }
        state.out.view()
    }
}

/// Records an array into a time series on clock signals, stamping each element
/// with the current timestamp (graph context).
///
/// A `retention` bound is used to control trimming of the series.
///
/// When `delayed` is set, trimming will be performed on the next clock signal.
/// This allows downstream operators to read elements that are about to be
/// trimmed.
pub fn record_on<T: Scalar + Float, const N: usize>(
    retention: impl Into<Retention>,
    delayed: bool,
) -> impl Segment<Inputs = (ClockPort, ArrayPort<T, N>), Outputs = SeriesPort<T, N>, Context = Instant>
{
    RecordOn::new(retention.into(), delayed)
}

/// Shorthand for [`record_on`] with `delayed = true`.
pub fn buffer<T: Scalar + Float, const N: usize>(
    retention: impl Into<Retention>,
) -> impl Segment<Inputs = (ClockPort, ArrayPort<T, N>), Outputs = SeriesPort<T, N>, Context = Instant>
{
    RecordOn::new(retention.into(), true)
}

/// Shorthand for [`record_on`] with [`Retention::unbounded`].
pub fn record_all<T: Scalar + Float, const N: usize>()
-> impl Segment<Inputs = (ClockPort, ArrayPort<T, N>), Outputs = SeriesPort<T, N>, Context = Instant>
{
    RecordOn::new(Retention::unbounded(), false)
}
