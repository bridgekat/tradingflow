use std::marker::PhantomData;
use std::ops::Range;

use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView};
use crate::graph::Segment;
use crate::ports::{ArrayPort, SeriesPort};

/// Base trait for rolling window operator accumulators.
///
/// Elements are added and removed from the window in time order:
///
/// - The [`init`](Self::init) method is called once at build time to
///   produce the initial output buffer.
/// - The [`add`](Self::add) method is called when a new element enters the
///   window.
/// - The [`remove`](Self::remove) method is called when an element leaves the
///   window.
/// - The [`write`](Self::write) method is called to write the current output.
pub trait Accumulator<T: Scalar, const N: usize, U: Scalar, const M: usize>:
    Send + 'static
{
    fn init(&mut self, extents: [usize; N]) -> Array<U, M>;
    fn add(&mut self, a: ArrayView<T, N>);
    fn remove(&mut self, a: ArrayView<T, N>);
    fn write(&mut self, out: &mut Array<U, M>, count: usize);
}

/// Operator signature for rolling window operators.
#[allow(clippy::type_complexity)]
pub struct Rolling<T: Scalar, const N: usize, U: Scalar, const M: usize, A: Accumulator<T, N, U, M>>
{
    window: Retention,
    acc: A,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, A: Accumulator<T, N, U, M>>
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
    U: Scalar,
    const M: usize,
    A: Accumulator<T, N, U, M>,
> {
    window: Retention,
    range: Range<usize>,
    acc: A,
    out: Array<U, M>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, A: Accumulator<T, N, U, M>> Segment
    for Rolling<T, N, U, M, A>
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
        instant: &Self::Context,
    ) -> ArrayView<'a, U, M> {
        assert!(
            series.range().start <= state.range.start,
            "rolling: input series dropped elements before operator can drop them (last window {:?}, got range {:?})",
            state.range,
            series.range()
        );
        let start = series.range().start + state.window.trim_count(series, instant);
        assert!(
            state.range.start <= start,
            "rolling: window start must be monotonic (last window {:?}, got new start {})",
            state.range,
            start
        );
        let end = series.range().end;
        assert!(
            state.range.end <= end,
            "rolling: input series range end must be monotonic (last window {:?}, got new end {})",
            state.range,
            end
        );
        for i in state.range.end..end {
            state.acc.add(series.at(i).1);
        }
        for i in state.range.start..start {
            state.acc.remove(series.at(i).1);
        }
        state.range = start..end;
        state.acc.write(&mut state.out, state.range.len());
        state.out.view()
    }
}
