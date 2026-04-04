//! Rolling window accumulator trait and window strategy.
//!
//! [`Accumulator`] defines the incremental add/remove/write interface for
//! rolling computations.  [`Window`] selects between count-based and
//! time-delta-based windowing.  [`Rolling`] is the generic operator that
//! combines an accumulator with a window strategy and implements
//! [`Operator`](crate::Operator).

use std::marker::PhantomData;

use num_traits::Float;

use crate::{Array, Notify, Operator, Scalar, Series};

// ===========================================================================
// Accumulator trait
// ===========================================================================

/// Incremental computation over a rolling window of array elements.
///
/// Implementors maintain running aggregates that are updated incrementally
/// as elements enter and leave the window.
pub trait Accumulator: Send + 'static {
    /// Scalar element type.
    type Scalar: Scalar + Float;

    /// Create a new accumulator from the input element shape.
    ///
    /// Most accumulators only need `input_shape.iter().product()` (the
    /// stride).  Covariance additionally validates 1D input and extracts
    /// the vector dimension.
    fn new(input_shape: &[usize]) -> Self;

    /// Output array shape given the input element shape.
    ///
    /// Default: same as input (element-wise operators).
    /// Override for operators that change shape (e.g. covariance: `[K] → [K, K]`).
    fn output_shape(input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    /// Incorporate a new element into the running aggregate.
    fn add(&mut self, element: &[Self::Scalar]);

    /// Remove a previously added element from the running aggregate.
    fn remove(&mut self, element: &[Self::Scalar]);

    /// Write the current aggregate value to `output`.
    ///
    /// `count` is the number of elements currently in the window.
    fn write(&self, count: usize, output: &mut [Self::Scalar]);
}

// ===========================================================================
// Window strategy
// ===========================================================================

/// Rolling window selection strategy.
#[derive(Debug, Clone, Copy)]
pub enum Window {
    /// Fixed number of most recent elements.
    Count(usize),
    /// All elements whose timestamp is within `window_ns` nanoseconds of
    /// the most recent element's timestamp.
    TimeDelta(i64),
}

// ===========================================================================
// Generic rolling operator
// ===========================================================================

/// Generic rolling operator: pairs an [`Accumulator`] with a [`Window`]
/// strategy.
///
/// Manages the sliding window (adding the newest element, evicting stale
/// elements) and delegates the actual computation to the accumulator.
pub struct Rolling<A: Accumulator> {
    window: Window,
    _phantom: PhantomData<A>,
}

impl<A: Accumulator> Rolling<A> {
    /// Create a count-based rolling operator.
    ///
    /// The window contains the last `window` elements.  Output is produced
    /// only once the window is full.
    pub fn count(window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        Self {
            window: Window::Count(window),
            _phantom: PhantomData,
        }
    }

    /// Create a time-delta-based rolling operator.
    ///
    /// The window contains all elements whose timestamp is within
    /// `window_ns` nanoseconds of the most recent element's timestamp.
    /// Output is produced as soon as at least one element is in the window.
    pub fn time_delta(window_ns: i64) -> Self {
        assert!(window_ns >= 0, "window_ns must be >= 0");
        Self {
            window: Window::TimeDelta(window_ns),
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Rolling`].
pub struct RollingState<A: Accumulator> {
    window: Window,
    /// Index into the series of the first element in the current window.
    start: usize,
    /// Number of elements currently in the window.
    count: usize,
    accumulator: A,
}

impl<A: Accumulator> Operator for Rolling<A> {
    type State = RollingState<A>;
    type Inputs = (Series<A::Scalar>,);
    type Output = Array<A::Scalar>;

    fn init(
        self,
        inputs: (&Series<A::Scalar>,),
        _timestamp: i64,
    ) -> (RollingState<A>, Array<A::Scalar>) {
        let input_shape = inputs.0.shape();
        let output_shape = A::output_shape(input_shape);
        let output_stride: usize = output_shape.iter().product();
        let state = RollingState {
            window: self.window,
            start: 0,
            count: 0,
            accumulator: A::new(input_shape),
        };
        (
            state,
            Array::from_vec(&output_shape, vec![A::Scalar::nan(); output_stride]),
        )
    }

    fn compute(
        state: &mut RollingState<A>,
        inputs: (&Series<A::Scalar>,),
        output: &mut Array<A::Scalar>,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        let series = inputs.0;
        let len = series.len();

        // Add the newest element.
        state.accumulator.add(series.at(len - 1));
        state.count += 1;

        // Evict elements that have left the window.
        match state.window {
            Window::Count(w) => {
                while state.count > w {
                    state.accumulator.remove(series.at(state.start));
                    state.start += 1;
                    state.count -= 1;
                }
                // Produce output only when the window is full.
                if state.count < w {
                    return false;
                }
            }
            Window::TimeDelta(w_ns) => {
                let current_ts = series.timestamps()[len - 1];
                let cutoff = current_ts - w_ns;
                while state.start < len && series.timestamps()[state.start] < cutoff {
                    state.accumulator.remove(series.at(state.start));
                    state.start += 1;
                    state.count -= 1;
                }
                // Always produce output if at least one element.
                if state.count == 0 {
                    return false;
                }
            }
        }

        state.accumulator.write(state.count, output.as_mut_slice());
        true
    }
}
