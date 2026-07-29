use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, SignalPort};

/// Base trait for return operator accumulators.
///
/// Percentage returns are added to the window in time order:
///
/// - The [`add`](Self::add) method is called when a new element enters the
///   window.
/// - The [`output`](Self::output) method is called for the current output.
pub trait Accumulator<T: Scalar + Float>: Send + 'static {
    fn add(&mut self, value: T);
    fn output(&mut self, count: usize) -> T;
}

/// Operator signature for return operators.
pub struct Return<T: Scalar + Float, A: Accumulator<T>> {
    acc: A,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, A: Accumulator<T>> Return<T, A> {
    pub fn new(acc: A) -> Self {
        Self {
            acc,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for return operators.
pub struct ReturnState<T: Scalar + Float, A: Accumulator<T>> {
    acc: A,
    prev: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, A: Accumulator<T>> Segment for Return<T, A> {
    type Inputs = (SignalPort<0>, ArrayPort<T, 0>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = ReturnState<T, A>;

    fn init(self, _: (ArrayView<'_, bool, 0>, ArrayView<'_, T, 0>)) -> Self::State {
        ReturnState {
            acc: self.acc,
            prev: T::nan(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, T, 0>),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, 0> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (signal, value): (ArrayView<'a, bool, 0>, ArrayView<'a, T, 0>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, 0> {
        if !*signal {
            return state.out.view();
        }
        assert!(value.is_finite(), "return: input value must be finite");
        assert!(*value > T::zero(), "return: input value must be positive");
        if state.prev.is_nan() {
            state.prev = *value;
            state.count = 0;
            *state.out = T::zero();
        } else {
            state.acc.add(*value / state.prev - T::one());
            state.prev = *value;
            state.count += 1;
            *state.out = state.acc.output(state.count);
        }
        state.out.view()
    }
}

/// Operator signature for log-return operators.
pub struct LogReturn<T: Scalar + Float, A: Accumulator<T>> {
    acc: A,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, A: Accumulator<T>> LogReturn<T, A> {
    pub fn new(acc: A) -> Self {
        Self {
            acc,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for log-return operators.
pub struct LogReturnState<T: Scalar + Float, A: Accumulator<T>> {
    acc: A,
    prev: T,
    count: usize,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, A: Accumulator<T>> Segment for LogReturn<T, A> {
    type Inputs = (SignalPort<0>, ArrayPort<T, 0>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = LogReturnState<T, A>;

    fn init(self, _: (ArrayView<'_, bool, 0>, ArrayView<'_, T, 0>)) -> Self::State {
        LogReturnState {
            acc: self.acc,
            prev: T::nan(),
            count: 0,
            out: Array::scalar(T::nan()),
        }
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, T, 0>),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, 0> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (signal, value): (ArrayView<'a, bool, 0>, ArrayView<'a, T, 0>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, 0> {
        if !*signal {
            return state.out.view();
        }
        assert!(value.is_finite(), "log_return: input value must be finite");
        assert!(
            *value > T::zero(),
            "log_return: input value must be positive"
        );
        if state.prev.is_nan() {
            state.prev = *value;
            state.count = 0;
            *state.out = T::zero();
        } else {
            state.acc.add(value.ln() - state.prev.ln());
            state.prev = *value;
            state.count += 1;
            *state.out = state.acc.output(state.count);
        }
        state.out.view()
    }
}
