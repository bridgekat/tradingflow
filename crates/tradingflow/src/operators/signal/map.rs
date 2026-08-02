use crate::data::ArrayView;
use crate::graph::{Interface, Operator};
use crate::ports::{ArrayPort, ArrayPorts, SignalPort, SignalPorts};

/// Interface mapping for [`as_signal_map`].
pub trait Signify: Interface {
    type Interface: Interface;

    fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_>;
}

impl Signify for () {
    type Interface = ();

    fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_> {
        values
    }
}

macro_rules! impl_signify_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: Signify),+> Signify for ($($T,)+) {
            type Interface = ($($T::Interface,)+);

            fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_> {
                ( $( $T::inputs(values.$idx), )+ )
            }
        }
    };
}

impl_signify_for_tuple!(0: A);
impl_signify_for_tuple!(0: A, 1: B);
impl_signify_for_tuple!(0: A, 1: B, 2: C);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_signify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

impl<const N: usize> Signify for ArrayPort<bool, N> {
    type Interface = SignalPort<N>;

    fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_> {
        values
    }
}

impl<const N: usize> Signify for ArrayPorts<bool, N> {
    type Interface = SignalPorts<N>;

    fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_> {
        values
    }
}

/// Operator signature for [`as_signal_map`].
pub struct SignalMap<const M: usize, T>
where
    T: Operator<Inputs: Signify, Outputs = ArrayPort<bool, M>>,
{
    op: T,
}

impl<const M: usize, T> SignalMap<M, T>
where
    T: Operator<Inputs: Signify, Outputs = ArrayPort<bool, M>>,
{
    pub fn new(op: T) -> Self {
        Self { op }
    }
}

impl<const M: usize, T> Operator for SignalMap<M, T>
where
    T: Operator<Inputs: Signify, Outputs = ArrayPort<bool, M>>,
{
    type Inputs = <T::Inputs as Signify>::Interface;
    type Outputs = SignalPort<M>;
    type Context = T::Context;
    type State = T::State;

    fn init(self, inputs: <Self::Inputs as Interface>::Values<'_>) -> T::State {
        T::init(self.op, <T::Inputs as Signify>::inputs(inputs))
    }

    fn reset<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut T::State,
    ) -> ArrayView<'a, bool, M> {
        let inputs = <T::Inputs as Signify>::inputs(inputs);
        ArrayView::full(T::reset(inputs, state).extents(), &false)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut T::State,
        context: &T::Context,
    ) -> ArrayView<'a, bool, M> {
        T::compute(<T::Inputs as Signify>::inputs(inputs), state, context)
    }
}

/// Applies a bool-array operator on signal arrays, resetting output each time.
pub fn as_signal_map<const M: usize, T>(
    op: T,
) -> impl Operator<
    Inputs = <T::Inputs as Signify>::Interface,
    Outputs = SignalPort<M>,
    Context = T::Context,
>
where
    T: Operator<Inputs: Signify, Outputs = ArrayPort<bool, M>>,
{
    SignalMap::new(op)
}
