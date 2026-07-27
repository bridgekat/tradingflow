use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Retention, Scalar, Series, SeriesView};
use crate::graph::Segment;
use crate::ports::{ArrayPort, ClockPort, SeriesPort, is_eventful};

/// Operator signature for [`record`] etc.
pub struct Record<T: Scalar, const N: usize> {
    retention: Retention,
    delayed: bool,
    _marker: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Record<T, N> {
    pub fn new(retention: Retention, delayed: bool) -> Self {
        Self {
            retention,
            delayed,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for [`Record`].
pub struct RecordState<T: Scalar, const N: usize> {
    retention: Retention,
    delayed: bool,
    out: Series<T, N>,
}

impl<T: Scalar + Float, const N: usize> RecordState<T, N> {
    /// Append one row, trimming before or after per `delayed`.
    fn push(&mut self, now: Instant, a: ArrayView<'_, T, N>) {
        if self.delayed {
            self.maybe_trim();
        }
        self.out.push(now, a);
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

impl<T: Scalar + Float, const N: usize> Segment for Record<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = RecordState<T, N>;

    fn init(self, a: ArrayView<'_, T, N>) -> RecordState<T, N> {
        RecordState {
            retention: self.retention,
            delayed: self.delayed,
            out: Series::new(a.extents()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: ArrayView<'a, T, N>,
        state: &'b mut RecordState<T, N>,
    ) -> SeriesView<'a, T, N> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        a: ArrayView<'a, T, N>,
        state: &'b mut RecordState<T, N>,
        now: &Instant,
    ) -> SeriesView<'a, T, N> {
        // Quiescence gate: a batch with no events appends nothing — the
        // record is the event log, and a no-event generation is not a row.
        if !is_eventful(a) {
            return state.out.view();
        }
        state.push(*now, a);
        state.out.view()
    }
}

/// Records an event array into a time series, stamping each element with the
/// current event time (graph context).
///
/// A `retention` bound is used to control trimming of the series.
///
/// When `delayed` is set, trimming will be performed one tick later.
/// This allows downstream operators to read elements that are about to be
/// trimmed.
pub fn record<T: Scalar + Float, const N: usize>(
    retention: impl Into<Retention>,
    delayed: bool,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(retention.into(), delayed)
}

/// Shorthand for [`record`] with `delayed = true`.
pub fn buffer<T: Scalar + Float, const N: usize>(
    retention: impl Into<Retention>,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(retention.into(), true)
}

/// Shorthand for [`record`] with [`Retention::unbounded`].
pub fn record_all<T: Scalar + Float, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(Retention::unbounded(), false)
}

/// Operator signature for [`record_clocked`].
pub struct RecordClocked<T: Scalar, const N: usize>(Record<T, N>);

impl<T: Scalar + Float, const N: usize> Segment for RecordClocked<T, N> {
    type Inputs = (ClockPort, ArrayPort<T, N>);
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = RecordState<T, N>;

    fn init(self, (_, a): (bool, ArrayView<'_, T, N>)) -> RecordState<T, N> {
        Record::init(self.0, a)
    }

    fn reset<'a, 'b: 'a>(
        (_, a): (bool, ArrayView<'a, T, N>),
        state: &'b mut RecordState<T, N>,
    ) -> SeriesView<'a, T, N> {
        Record::reset(a, state)
    }

    fn compute<'a, 'b: 'a>(
        (clock, a): (bool, ArrayView<'a, T, N>),
        state: &'b mut RecordState<T, N>,
        now: &Instant,
    ) -> SeriesView<'a, T, N> {
        if !clock {
            return state.out.view();
        }
        state.push(*now, a);
        state.out.view()
    }
}

/// Records a **state** array into a time series on every tick of a clock,
/// stamping each row with the current event time.
///
/// Unlike [`record`], which logs the events standing on its input and so skips
/// a generation carrying none, this samples on the clock alone: a tick whose
/// values happen to be all-`NaN` still appends a row. That is what keeps two
/// series sampled on one clock index-aligned — a consumer pairing
/// `a[i]` with `b[i]` needs the row index to be a time index, which only holds
/// if neither series can silently drop a row.
pub fn record_clocked<T: Scalar + Float, const N: usize>(
    retention: impl Into<Retention>,
    delayed: bool,
) -> impl Segment<Inputs = (ClockPort, ArrayPort<T, N>), Outputs = SeriesPort<T, N>, Context = Instant>
{
    RecordClocked(Record::new(retention.into(), delayed))
}
