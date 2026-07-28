use crate::data::ArrayView;
use crate::graph::{Interface, Segment};
use crate::ports::{ArrayPort, ArrayPorts, ClockArrayPort, ClockArrayPorts};

/// Interface mapping for [`as_clock_map`].
pub trait Clockify: Interface {
    type Interface: Interface;

    fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_>;
}

impl Clockify for () {
    type Interface = ();

    fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_> {
        values
    }
}

macro_rules! impl_clockify_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: Clockify),+> Clockify for ($($T,)+) {
            type Interface = ($($T::Interface,)+);

            fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_> {
                ( $( $T::inputs(values.$idx), )+ )
            }
        }
    };
}

impl_clockify_for_tuple!(0: A);
impl_clockify_for_tuple!(0: A, 1: B);
impl_clockify_for_tuple!(0: A, 1: B, 2: C);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_clockify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

impl<const N: usize> Clockify for ArrayPort<bool, N> {
    type Interface = ClockArrayPort<N>;

    fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_> {
        values
    }
}

impl<const N: usize> Clockify for ArrayPorts<bool, N> {
    type Interface = ClockArrayPorts<N>;

    fn inputs(values: <Self::Interface as Interface>::Values<'_>) -> Self::Values<'_> {
        values
    }
}

/// Operator signature for [`as_clock_map`].
pub struct ClockMap<const M: usize, T>
where
    T: Segment<Inputs: Clockify, Outputs = ArrayPort<bool, M>>,
{
    segment: T,
}

impl<const M: usize, T> ClockMap<M, T>
where
    T: Segment<Inputs: Clockify, Outputs = ArrayPort<bool, M>>,
{
    pub fn new(segment: T) -> Self {
        Self { segment }
    }
}

impl<const M: usize, T> Segment for ClockMap<M, T>
where
    T: Segment<Inputs: Clockify, Outputs = ArrayPort<bool, M>>,
{
    type Inputs = <T::Inputs as Clockify>::Interface;
    type Outputs = ClockArrayPort<M>;
    type Context = T::Context;
    type State = T::State;

    fn init(self, inputs: <Self::Inputs as Interface>::Values<'_>) -> T::State {
        T::init(self.segment, <T::Inputs as Clockify>::inputs(inputs))
    }

    fn reset<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut T::State,
    ) -> ArrayView<'a, bool, M> {
        let inputs = <T::Inputs as Clockify>::inputs(inputs);
        ArrayView::full(T::reset(inputs, state).extents(), &false)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut T::State,
        context: &T::Context,
    ) -> ArrayView<'a, bool, M> {
        T::compute(<T::Inputs as Clockify>::inputs(inputs), state, context)
    }
}

/// Applies a bool-array segment on clock arrays, resetting output each time.
pub fn as_clock_map<const M: usize, T>(
    segment: T,
) -> impl Segment<
    Inputs = <T::Inputs as Clockify>::Interface,
    Outputs = ClockArrayPort<M>,
    Context = T::Context,
>
where
    T: Segment<Inputs: Clockify, Outputs = ArrayPort<bool, M>>,
{
    ClockMap::new(segment)
}
