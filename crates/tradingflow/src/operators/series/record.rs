use std::marker::PhantomData;

use crate::data::{ArrayView, Duration, Instant, Scalar, Series, SeriesView};
use crate::graph::{Operator, Segment};
use crate::ports::{ArrayPort, SeriesPort};

/// A retention bound for [`Record`]: how much history it keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Retention {
    pub delayed: bool,
    pub unbounded: bool,
    pub count: Option<usize>,
    pub duration: Option<Duration>,
}

impl Retention {
    /// Keeps all history.
    pub const fn unbounded() -> Self {
        Self {
            delayed: false,
            unbounded: true,
            count: None,
            duration: None,
        }
    }

    /// Keeps the most-recent `count` elements.
    pub fn count(count: usize) -> Self {
        Self {
            delayed: false,
            unbounded: false,
            count: Some(count),
            duration: None,
        }
    }

    /// Keeps all elements within `duration` of the latest timestamp.
    pub fn duration(duration: Duration) -> Self {
        Self {
            delayed: false,
            unbounded: false,
            count: None,
            duration: Some(duration),
        }
    }

    /// Keeps the union of a `count` window and a `duration` window.
    pub fn count_and_duration(count: usize, duration: Duration) -> Self {
        Self {
            delayed: false,
            unbounded: false,
            count: Some(count),
            duration: Some(duration),
        }
    }

    /// When [`delayed`](Self::delayed) is set, trimming will be performed
    /// one tick later. This allows downstream operators to read elements that
    /// are about to be trimmed. Default is `false`.
    pub fn delayed(self, delayed: bool) -> Self {
        Self { delayed, ..self }
    }
}

/// Operator signature for [`record`] etc.
pub struct Record<T: Scalar, const N: usize> {
    retention: Retention,
    _p: PhantomData<T>,
}

impl<T: Scalar, const N: usize> Record<T, N> {
    pub fn new(retention: Retention) -> Self {
        Self {
            retention,
            _p: PhantomData,
        }
    }
}

/// Runtime state for [`Record`].
pub struct RecordState<T: Scalar, const N: usize> {
    retention: Retention,
    out: Series<T, N>,
}

impl<T: Scalar, const N: usize> RecordState<T, N> {
    fn trim(&mut self) {
        if self.out.is_empty() {
            return;
        }
        let retention = self.retention;
        let range = self.out.range();
        let mut start = range.end;
        if retention.unbounded {
            start = start.min(range.start);
        }
        if let Some(c) = retention.count {
            start = start.min(range.end.saturating_sub(c));
        }
        if let Some(d) = retention.duration {
            let instants = self.out.instants();
            let cutoff = instants.last().unwrap().saturating_sub(d);
            let i = instants.partition_point(|&t| t < cutoff);
            start = start.min(range.start + i);
        }
        let count = start.saturating_sub(range.start);
        if count > 0 && count * 2 >= self.out.len() {
            self.out.trim(count);
        }
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
        if state.retention.delayed {
            state.trim();
        }
        state.out.push(*now, x);
        if !state.retention.delayed {
            state.trim();
        }
        (true, state.out.view())
    }
}

/// Records an array input into a time series, stamping each element with the
/// current event time (graph context).
///
/// A [`Retention`] bound is used to control trimming of the series.
pub fn record<T: Scalar, const N: usize>(
    retention: Retention,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(retention)
}

/// Shorthand for [`record`] with [`Retention::unbounded`].
pub fn record_all<T: Scalar, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(Retention::unbounded())
}

/// Shorthand for [`record`] with [`Retention::count`], with
/// [`Retention::delayed`] set to `true`.
pub fn buffer_n<T: Scalar, const N: usize>(
    n: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(Retention::count(n).delayed(true))
}

/// Shorthand for [`record`] with [`Retention::duration`], with
/// [`Retention::delayed`] set to `true`.
pub fn buffer_duration<T: Scalar, const N: usize>(
    duration: Duration,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    Record::new(Retention::duration(duration).delayed(true))
}
