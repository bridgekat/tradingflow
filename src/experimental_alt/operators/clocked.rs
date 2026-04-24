//! Clock-gated operator wrapper — ported from `src/operators/clocked.rs`.

use std::marker::PhantomData;

use super::super::data::{Input, InputTypes, Instant};
use super::super::operator::Operator;

pub struct Clocked<O, C = ()> {
    inner: O,
    _phantom: PhantomData<fn() -> C>,
}

impl<O, C> Clocked<O, C> {
    pub fn new(inner: O) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<O: Operator, C: Send + Sync + 'static> Operator for Clocked<O, C>
where
    O::Output: Clone + Send + Sync + 'static,
{
    type State = O::State;
    type Inputs = (Input<C>, O::Inputs);
    type Output = O::Output;

    fn init(
        self,
        inputs: (&C, <O::Inputs as InputTypes>::Refs<'_>),
        timestamp: Instant,
    ) -> (O::State, O::Output) {
        self.inner.init(inputs.1, timestamp)
    }

    fn compute(
        state: &mut O::State,
        inputs: (&C, <O::Inputs as InputTypes>::Refs<'_>),
        output: &mut O::Output,
        timestamp: Instant,
        produced: (bool, <O::Inputs as InputTypes>::Produced<'_>),
    ) -> bool {
        let (clock_fired, inner_produced) = produced;
        if !clock_fired {
            return false;
        }
        O::compute(state, inputs.1, output, timestamp, inner_produced)
    }
}
