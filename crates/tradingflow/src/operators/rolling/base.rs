use num_traits::Float;
use std::marker::PhantomData;
use std::ops::Range;

use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Base trait for rolling window operator accumulators.
///
/// Elements are always added and removed from the window in time order.
pub trait Accumulator<T: Scalar, const N: usize, U: Scalar + Float, const M: usize>:
    Send + 'static
{
    /// Whether [`write`](Accumulator::write) reads its `window` argument.
    /// If `false`, [`write`](Accumulator::write) receives an empty `window`.
    const NEEDS_WINDOW: bool = true;

    /// Produces the initial output buffer. Called once at build time.
    fn init(&mut self, extents: [usize; N]) -> Array<U, M>;

    /// Adds a new element to the window.
    fn add(&mut self, a: ArrayView<T, N>);

    /// Removes an element from the window.
    fn remove(&mut self, a: ArrayView<T, N>);

    /// Writes the current output to the provided buffer. All elements in
    /// the current window is accessible from `window` if needed.
    fn write(&mut self, out: &mut Array<U, M>, window: SeriesView<'_, T, N>);
}

/// Operator signature for rolling window operators.
#[allow(clippy::type_complexity)]
pub struct Rolling<
    T: Scalar,
    const N: usize,
    U: Scalar + Float,
    const M: usize,
    A: Accumulator<T, N, U, M>,
> {
    window: Retention,
    acc: A,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar + Float, const M: usize, A: Accumulator<T, N, U, M>>
    Rolling<T, N, U, M, A>
{
    pub fn new(window: Retention, acc: A) -> Self {
        Self {
            window,
            acc,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for rolling window operators.
#[allow(clippy::type_complexity)]
pub struct RollingState<
    T: Scalar,
    const N: usize,
    U: Scalar + Float,
    const M: usize,
    A: Accumulator<T, N, U, M>,
> {
    window: Retention,
    range: Range<usize>,
    acc: A,
    out: Array<U, M>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, const N: usize, U: Scalar + Float, const M: usize, A: Accumulator<T, N, U, M>>
    Operator for Rolling<T, N, U, M, A>
{
    type Inputs = SeriesPort<T, N>;
    type Outputs = ArrayPort<U, M>;
    type Context = Instant;
    type State = RollingState<T, N, U, M, A>;

    fn init(self, series: SeriesView<'_, T, N>) -> Self::State {
        assert!(
            series.range().is_empty(),
            "rolling: operator must be initialized with an empty input series (got range {:?})",
            series.range()
        );
        let mut acc = self.acc;
        let out = acc.init(series.extents());
        RollingState {
            window: self.window,
            range: series.range(),
            acc,
            out,
            _marker: PhantomData,
        }
    }

    fn reset<'a, 'b: 'a>(
        _: SeriesView<'a, T, N>,
        state: &'b mut Self::State,
    ) -> ArrayView<'a, U, M> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        series: SeriesView<'a, T, N>,
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, U, M> {
        let is_streaming = !A::NEEDS_WINDOW && state.window.is_unbounded();
        let end = series.range().end;
        assert!(
            state.range.end <= end,
            "rolling: input series range end must be monotonic (last window {:?}, got new end {})",
            state.range,
            end
        );
        if state.range.end == end {
            return state.out.view();
        }
        let start = if is_streaming {
            assert!(
                series.range().start <= state.range.end,
                "rolling: input series dropped elements before operator can read them (last window {:?}, got range {:?})",
                state.range,
                series.range()
            );
            state.range.start
        } else {
            assert!(
                series.range().start <= state.range.start,
                "rolling: input series dropped elements before operator can drop them (last window {:?}, got range {:?})",
                state.range,
                series.range()
            );
            let instant = series.at(end - 1).0;
            series.range().start + state.window.trim_count(series, &instant)
        };
        assert!(
            state.range.start <= start,
            "rolling: window start must be monotonic (last window {:?}, got new start {})",
            state.range,
            start
        );
        for i in state.range.end..end {
            state.acc.add(series.at(i).1);
        }
        for i in state.range.start..start {
            state.acc.remove(series.at(i).1);
        }
        let window = if A::NEEDS_WINDOW {
            series.window(start..end)
        } else {
            series.window(end..end)
        };
        state.range = start..end;
        state.acc.write(&mut state.out, window);
        state.out.view()
    }
}

/// Accumulator adaptor for window-scanning statistics: `add`/`remove` are
/// no-ops and the statistic is recomputed per lane from the window view on
/// every [`write`](Accumulator::write).
///
/// The closure receives the lane's **finite** samples as `(position, value)`
/// pairs, oldest first — positions are 0-based from the window start and count
/// every row, so non-finite samples leave gaps — plus the total number of rows
/// in the window. Lanes with fewer than `min_count` finite samples emit `NaN`
/// without calling the closure.
pub struct Scanning<T: Scalar + Float, F> {
    min_count: usize,
    lanes: Vec<Vec<(usize, T)>>,
    f: F,
}

impl<T: Scalar + Float, F> Scanning<T, F> {
    pub fn new(min_count: usize, f: F) -> Self {
        Self {
            min_count,
            lanes: Vec::new(),
            f,
        }
    }
}

impl<T: Scalar + Float, const N: usize, F> Accumulator<T, N, T, N> for Scanning<T, F>
where
    F: FnMut(&[(usize, T)], usize) -> T + Send + 'static,
{
    const NEEDS_WINDOW: bool = true;

    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.lanes = vec![Vec::new(); stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, _: ArrayView<T, N>) {}

    fn remove(&mut self, _: ArrayView<T, N>) {}

    fn write(&mut self, out: &mut Array<T, N>, window: SeriesView<'_, T, N>) {
        for lane in self.lanes.iter_mut() {
            lane.clear();
        }
        let start = window.range().start;
        for i in window.range() {
            let lanes = &mut self.lanes;
            array::for_each(window.at(i).1, |j, &x| {
                if x.is_finite() {
                    lanes[j].push((i - start, x));
                }
            });
        }
        let len = window.len();
        for (o, pairs) in out.data_mut().iter_mut().zip(self.lanes.iter()) {
            *o = if pairs.len() < self.min_count {
                T::nan()
            } else {
                (self.f)(pairs, len)
            };
        }
    }
}

/// [`rolling`] over an explicitly recorded series.
pub fn series_rolling<
    T: Scalar + Float,
    const N: usize,
    U: Scalar + Float,
    const M: usize,
    A: Accumulator<T, N, U, M>,
>(
    window: impl Into<Retention>,
    acc: A,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<U, M>, Context = Instant> {
    Rolling::new(window.into(), acc)
}

/// Rolling window operator with the given [`Accumulator`].
pub fn rolling<
    T: Scalar + Float,
    const N: usize,
    U: Scalar + Float,
    const M: usize,
    A: Accumulator<T, N, U, M>,
>(
    window: impl Into<Retention>,
    acc: A,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<U, M>, Context = Instant>
{
    let window = window.into();
    let is_streaming = !A::NEEDS_WINDOW && window.is_unbounded();
    let retention = if is_streaming {
        Retention::count(0)
    } else {
        window
    };
    series::buffer(retention).then(Rolling::new(window, acc))
}
