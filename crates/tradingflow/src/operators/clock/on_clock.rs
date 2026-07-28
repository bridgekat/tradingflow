use crate::data::ArrayView;
use crate::graph::{Interface, Segment};
use crate::ports::ClockPort;

/// Operator signature for [`on_clock`].
pub struct OnClock<T: Segment> {
    segment: T,
}

impl<T: Segment> OnClock<T> {
    pub fn new(segment: T) -> Self {
        Self { segment }
    }
}

impl<T: Segment> Segment for OnClock<T> {
    type Inputs = (ClockPort, T::Inputs);
    type Outputs = T::Outputs;
    type Context = T::Context;
    type State = T::State;

    fn init(
        self,
        (_, inputs): (ArrayView<'_, bool, 0>, <T::Inputs as Interface>::Values<'_>),
    ) -> T::State {
        self.segment.init(inputs)
    }

    fn reset<'a, 'b: 'a>(
        (_, inputs): (ArrayView<'a, bool, 0>, <T::Inputs as Interface>::Values<'a>),
        state: &'b mut T::State,
    ) -> <T::Outputs as Interface>::Values<'a> {
        T::reset(inputs, state)
    }

    fn compute<'a, 'b: 'a>(
        (clock, inputs): (ArrayView<'a, bool, 0>, <T::Inputs as Interface>::Values<'a>),
        state: &'b mut T::State,
        context: &T::Context,
    ) -> <T::Outputs as Interface>::Values<'a> {
        if *clock {
            T::compute(inputs, state, context)
        } else {
            T::reset(inputs, state)
        }
    }
}

/// Wraps a segment to only compute when the input clock signals `true`.
pub fn on_clock<T: Segment>(
    segment: T,
) -> impl Segment<Inputs = (ClockPort, T::Inputs), Outputs = T::Outputs, Context = T::Context> {
    OnClock::new(segment)
}
