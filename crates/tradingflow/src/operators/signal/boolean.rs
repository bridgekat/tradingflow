use std::marker::PhantomData;

use crate::data::{ArrayView, Instant, Scalar};
use crate::graph::Segment;
use crate::ports::{ArrayPort, SignalPort};

/// Operator signature for [`always`].
pub struct Always<T: Scalar, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, const N: usize> Always<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for Always<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Segment for Always<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = SignalPort<0>;
    type Context = Instant;
    type State = ();

    fn init(self, _: ArrayView<'_, T, N>) {}

    fn reset<'a, 'b: 'a>(_: ArrayView<'a, T, N>, _: &'b mut ()) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(&false)
    }

    fn compute<'a, 'b: 'a>(
        _: ArrayView<'a, T, N>,
        _: &'b mut (),
        _: &Instant,
    ) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(&true)
    }
}

/// Operator signature for [`filter`].
pub struct Filter<T: Scalar, const N: usize, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
{
    predicate: F,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar, const N: usize, F> Filter<T, N, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
{
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, F> Segment for Filter<T, N, F>
where
    F: FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
{
    type Inputs = (SignalPort<0>, ArrayPort<T, N>);
    type Outputs = SignalPort<0>;
    type Context = Instant;
    type State = F;

    fn init(self, _: (ArrayView<'_, bool, 0>, ArrayView<'_, T, N>)) -> Self::State {
        self.predicate
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, T, N>),
        _: &'b mut Self::State,
    ) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(&false)
    }

    fn compute<'a, 'b: 'a>(
        (signal, a): (ArrayView<'a, bool, 0>, ArrayView<'a, T, N>),
        f: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(if *signal && f(a) { &true } else { &false })
    }
}

/// Operator signature for [`or`].
pub struct Or;

impl Segment for Or {
    type Inputs = (SignalPort<0>, SignalPort<0>);
    type Outputs = SignalPort<0>;
    type Context = Instant;
    type State = ();

    fn init(self, _: (ArrayView<'_, bool, 0>, ArrayView<'_, bool, 0>)) {}

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, bool, 0>),
        _: &'b mut (),
    ) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(&false)
    }

    fn compute<'a, 'b: 'a>(
        (a, b): (ArrayView<'a, bool, 0>, ArrayView<'a, bool, 0>),
        _: &'b mut (),
        _: &Instant,
    ) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(if *a || *b { &true } else { &false })
    }
}

/// Operator signature for [`and`].
pub struct And;

impl Segment for And {
    type Inputs = (SignalPort<0>, SignalPort<0>);
    type Outputs = SignalPort<0>;
    type Context = Instant;
    type State = ();

    fn init(self, _: (ArrayView<'_, bool, 0>, ArrayView<'_, bool, 0>)) {}

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, bool, 0>),
        _: &'b mut (),
    ) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(&false)
    }

    fn compute<'a, 'b: 'a>(
        (a, b): (ArrayView<'a, bool, 0>, ArrayView<'a, bool, 0>),
        _: &'b mut (),
        _: &Instant,
    ) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(if *a && *b { &true } else { &false })
    }
}

/// Operator signature for [`any`].
pub struct Any<const N: usize>;

impl<const N: usize> Segment for Any<N> {
    type Inputs = SignalPort<N>;
    type Outputs = SignalPort<0>;
    type Context = Instant;
    type State = ();

    fn init(self, _: ArrayView<'_, bool, N>) {}

    fn reset<'a, 'b: 'a>(_: ArrayView<'a, bool, N>, _: &'b mut ()) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(&false)
    }

    fn compute<'a, 'b: 'a>(
        a: ArrayView<'a, bool, N>,
        _: &'b mut (),
        _: &Instant,
    ) -> ArrayView<'a, bool, 0> {
        let any = if a.data().len() == 1 {
            a.data()[0]
        } else {
            a.iter().any(|&b| b)
        };
        ArrayView::scalar(if any { &true } else { &false })
    }
}

/// A signal that always signals `true` when computed (for testing purposes).
pub fn always<T: Scalar, const N: usize>()
-> impl Segment<Inputs = ArrayPort<T, N>, Outputs = SignalPort<0>, Context = Instant> {
    Always::new()
}

/// Derives a signal which signals `true` when the input signals `true`
/// and the input array satisfies `predicate`.
pub fn filter<T: Scalar, const N: usize>(
    predicate: impl FnMut(ArrayView<'_, T, N>) -> bool + Send + 'static,
) -> impl Segment<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = SignalPort<0>, Context = Instant>
{
    Filter::new(predicate)
}

/// Derives a signal which signals `true` when both of the input signals
/// signal `true`.
pub fn and()
-> impl Segment<Inputs = (SignalPort<0>, SignalPort<0>), Outputs = SignalPort<0>, Context = Instant>
{
    And
}

/// Derives a signal which signals `true` when either of the input signals
/// signals `true`.
pub fn or()
-> impl Segment<Inputs = (SignalPort<0>, SignalPort<0>), Outputs = SignalPort<0>, Context = Instant>
{
    Or
}

/// Reduces a signal array to a single signal which signals `true` when
/// any element signals `true`.
pub fn any<const N: usize>()
-> impl Segment<Inputs = SignalPort<N>, Outputs = SignalPort<0>, Context = Instant> {
    Any
}
