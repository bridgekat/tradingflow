use num_traits::Float;
use std::marker::PhantomData;
use std::ops::Range;

use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView};
use crate::graph::Operator;
use crate::ports::{ArrayPort, SeriesPort};

/// Base trait for rolling window operator accumulators.
///
/// Elements are always added and removed from the window in time order.
pub trait Accumulator<T: Scalar, const N: usize, U: Scalar + Float, const M: usize>:
    Send + 'static
{
    /// Produces the initial output buffer. Called once at build time.
    fn init(&mut self, extents: [usize; N]) -> Array<U, M>;

    /// Adds a new element to the window.
    fn add(&mut self, a: ArrayView<T, N>);

    /// Removes an element from the window.
    fn remove(&mut self, a: ArrayView<T, N>);

    /// Writes the current output to the provided buffer.
    fn write(&mut self, out: &mut Array<U, M>, count: usize);
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
        assert!(
            series.range().start <= state.range.start,
            "rolling: input series dropped elements before operator can drop them (last window {:?}, got range {:?})",
            state.range,
            series.range()
        );
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
        let instant = series.at(end - 1).0;
        let start = series.range().start + state.window.trim_count(series, &instant);
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
        state.range = start..end;
        state.acc.write(&mut state.out, state.range.len());
        state.out.view()
    }
}
