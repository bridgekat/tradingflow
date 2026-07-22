//! `Record` — append an Array stream into a Series, stamping with event time.

use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Retention, Scalar, Series, SeriesView};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, SeriesPort};

/// Records an `Array<T, N>` stream into a `Series<T, N>`, stamping each row with
/// the event time — the graph context the driver sets before each `stabilize`.
/// The only native operator that reads time (the Python host is the other, behind
/// the `python` feature), and it needs nothing at construction:
/// [`record`] / [`record_bounded`] take
/// no clock.
///
/// An optional [`Retention`] bound (via [`with_retention`](Self::with_retention)
/// / [`record_bounded`]) caps the recorded history.
pub struct Record<T: Scalar, const N: usize> {
    retention: Retention,
    _p: PhantomData<T>,
}

impl<T: Scalar, const N: usize> Record<T, N> {
    /// An unbounded record (retains full history).
    pub fn new() -> Self {
        Self::with_retention(Retention::unbounded())
    }

    /// A record whose `Series` keeps only the history within `retention`.
    pub fn with_retention(retention: Retention) -> Self {
        Self {
            retention,
            _p: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for Record<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Record`]: the recorded series (which carries its own
/// retention bound).
pub struct RecordState<T: Scalar, const N: usize> {
    out: Series<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Record<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = RecordState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        // Init only sizes the series — no row is appended, so the
        // pre-first-batch context value is never stamped into it.
        RecordState {
            out: Series::new(x.extents(), self.retention),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        time: &Instant,
    ) -> (bool, SeriesView<'a, T, N>) {
        state.out.push(*time, x);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, SeriesView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// An unbounded [`Record`] of the input stream: `record() @ x` appends
/// every notified value of `x`, stamped with event time. Prefer
/// [`record_bounded`] whenever the consumers' look-back is known.
pub fn record<T: Scalar, const N: usize>() -> Record<T, N> {
    Record::new()
}

/// A [`Record`] keeping only the history within `retention` — the hoisted
/// shared-record form: record once, feed many windowed consumers. Size
/// `retention` to the deepest consumer look-back plus a compaction margin (see
/// the module docs).
pub fn record_bounded<T: Scalar, const N: usize>(retention: Retention) -> Record<T, N> {
    Record::with_retention(retention)
}

/// A private record sized for a count look-back of `n`.
pub fn buffer<T: Scalar, const N: usize>(n: usize) -> Record<T, N> {
    // Extra rows a private record retains beyond its consumer's exact count
    // look-back — absorbs the amortized-compaction slack plus the one-row
    // overshoot a sliding window reads while evicting.
    const COUNT_MARGIN: usize = 8;
    Record::with_retention(Retention::count(n + COUNT_MARGIN))
}
