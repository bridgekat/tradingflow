use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Retention, Scalar, Series, SeriesView};
use crate::graph::{Operator, Segment};
use crate::ports::{ArrayPort, SeriesPort};

/// Operator signature for [`record`] etc.
pub struct Record<T: Scalar, const N: usize> {
    retention: Retention,
    delayed: bool,
    _marker: PhantomData<T>,
}

impl<T: Scalar, const N: usize> Record<T, N> {
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

impl<T: Scalar, const N: usize> RecordState<T, N> {
    fn maybe_trim(&mut self) {
        if let Some(now) = self.out.instants().last() {
            let count = self.retention.trim_count(self.out.view(), now);
            if count > 0 && count * 2 >= self.out.len() {
                self.out.trim(count);
            }
        };
    }
}

impl<T: Scalar, const N: usize> Operator for Record<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = RecordState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> RecordState<T, N> {
        RecordState {
            retention: self.retention,
            delayed: self.delayed,
            out: Series::new(x.extents()),
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut RecordState<T, N>,
    ) -> (bool, SeriesView<'a, T, N>) {
        (false, state.out.view())
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut RecordState<T, N>,
        now: &Instant,
    ) -> (bool, SeriesView<'a, T, N>) {
        if state.delayed {
            state.maybe_trim();
        }
        state.out.push(*now, x);
        if !state.delayed {
            state.maybe_trim();
        }
        (true, state.out.view())
    }
}

/// Records an array input into a time series, stamping each element with the
/// current event time (graph context).
///
/// A `retention` bound is used to control trimming of the series.
///
/// When `delayed` is set, trimming will be performed one tick later.
/// This allows downstream operators to read elements that are about to be
/// trimmed.
pub fn record<T: Scalar, const N: usize>(
    retention: impl Into<Retention>,
    delayed: bool,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(retention.into(), delayed)
}

/// Shorthand for [`record`] with `delayed = true`.
pub fn buffer<T: Scalar, const N: usize>(
    retention: impl Into<Retention>,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(retention.into(), true)
}

/// Shorthand for [`record`] with [`Retention::unbounded`].
pub fn record_all<T: Scalar, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(Retention::unbounded(), false)
}
