use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Duration, Instant, SeriesView};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, SeriesPort};

use super::{Accumulator, Window};

/// Convert an accumulator's dynamic output shape into a static `[usize; NO]`.
fn out_extents<const NO: usize>(shape: &[usize]) -> [usize; NO] {
    <[usize; NO]>::try_from(shape)
        .unwrap_or_else(|_| panic!("rolling: output rank {} != NO {NO}", shape.len()))
}

/// Pairs an [`Accumulator`] with a [`Window`] strategy. `NI` is the input
/// series' element rank, `NO` the output rank (they differ only for
/// rank-changing accumulators, e.g. covariance `[K] → [K, K]`).
pub struct Rolling<A: Accumulator, const NI: usize, const NO: usize> {
    window: Window,
    _phantom: PhantomData<A>,
}

impl<A: Accumulator, const NI: usize, const NO: usize> Clone for Rolling<A, NI, NO> {
    fn clone(&self) -> Self {
        Self {
            window: self.window,
            _phantom: PhantomData,
        }
    }
}

impl<A: Accumulator, const NI: usize, const NO: usize> Rolling<A, NI, NO> {
    /// Count-based window of the last `window` elements; output only once full.
    pub fn count(window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        Self {
            window: Window::Count(window),
            _phantom: PhantomData,
        }
    }

    /// Time-delta window: all elements within `window` of the latest timestamp.
    pub fn time_delta(window: Duration) -> Self {
        assert!(window.as_nanos() >= 0, "window must be non-negative");
        Self {
            window: Window::TimeDelta(window),
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Rolling`]: the window config, accumulator bookkeeping
/// (`count` rows accumulated — always the newest `count` rows of the series
/// window), plus the output buffer.
pub struct RollingState<A: Accumulator, const NO: usize> {
    window: Window,
    count: usize,
    accumulator: A,
    out: Array<A::Scalar, NO>,
}

impl<A: Accumulator, const NI: usize, const NO: usize> Operator for Rolling<A, NI, NO> {
    type Inputs = SeriesPort<A::Scalar, NI>;
    type Outputs = ArrayPort<A::Scalar, NO>;
    type Context = Instant;
    type State = RollingState<A, NO>;

    fn init(self, (_, series): (bool, SeriesView<'_, A::Scalar, NI>)) -> Self::State {
        let input_shape = series.extents();
        let output_shape = A::output_shape(&input_shape);
        let output_stride: usize = output_shape.iter().product();
        RollingState {
            window: self.window,
            count: 0,
            accumulator: A::new(&input_shape),
            out: Array::from_parts(
                out_extents::<NO>(&output_shape),
                vec![A::Scalar::nan(); output_stride].into(),
            ),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, A::Scalar, NI>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, A::Scalar, NO>) {
        // The accumulated rows are always the newest `count` rows, so the
        // oldest accumulated row sits at logical index `end - count` — and
        // logical indices are unaffected by the record's trims.
        let end = series.range().end;

        state.accumulator.add(&series.at(end - 1).1.to_contiguous());
        state.count += 1;

        match state.window {
            Window::Count(w) => {
                while state.count > w {
                    state
                        .accumulator
                        .remove(&series.at(end - state.count).1.to_contiguous());
                    state.count -= 1;
                }
                if state.count < w {
                    return (false, state.out.view());
                }
            }
            Window::TimeDelta(w) => {
                let current_ts = series.at(end - 1).0;
                let cutoff = current_ts - w;
                while state.count > 0 && series.at(end - state.count).0 < cutoff {
                    state
                        .accumulator
                        .remove(&series.at(end - state.count).1.to_contiguous());
                    state.count -= 1;
                }
                if state.count == 0 {
                    return (false, state.out.view());
                }
            }
        }

        state.accumulator.write(state.count, state.out.data_mut());
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, A::Scalar, NI>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, A::Scalar, NO>) {
        (false, state.out.view())
    }
}

/// [`Rolling`] over an explicit [`Window`] — the accumulator-generic form.
pub fn rolling<A: Accumulator, const NI: usize, const NO: usize>(
    window: Window,
) -> Rolling<A, NI, NO> {
    match window {
        Window::Count(w) => Rolling::count(w),
        Window::TimeDelta(w) => Rolling::time_delta(w),
    }
}
