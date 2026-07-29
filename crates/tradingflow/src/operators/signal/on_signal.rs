use crate::data::ArrayView;
use crate::graph::{Interface, Segment};
use crate::ports::SignalPort;

/// Operator signature for [`on_signal`].
pub struct OnSignal<T: Segment> {
    segment: T,
}

impl<T: Segment> OnSignal<T> {
    pub fn new(segment: T) -> Self {
        Self { segment }
    }
}

impl<T: Segment> Segment for OnSignal<T> {
    type Inputs = (SignalPort<0>, T::Inputs);
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
        (signal, inputs): (ArrayView<'a, bool, 0>, <T::Inputs as Interface>::Values<'a>),
        state: &'b mut T::State,
        context: &T::Context,
    ) -> <T::Outputs as Interface>::Values<'a> {
        if *signal {
            T::compute(inputs, state, context)
        } else {
            T::reset(inputs, state)
        }
    }
}

/// Wraps a segment to only compute when the input signals `true`.
pub fn on_signal<T: Segment>(
    segment: T,
) -> impl Segment<Inputs = (SignalPort<0>, T::Inputs), Outputs = T::Outputs, Context = T::Context> {
    OnSignal::new(segment)
}
