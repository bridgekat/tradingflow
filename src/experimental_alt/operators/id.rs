//! Identity operator — ported from `src/operators/id.rs`.

use std::marker::PhantomData;

use super::super::data::{Input, InputTypes, Instant};
use super::super::operator::Operator;

pub struct Id<T: Clone + Send + Sync + 'static> {
    _phantom: PhantomData<T>,
}

impl<T: Clone + Send + Sync + 'static> Id<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Clone + Send + Sync + 'static> Default for Id<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Send + Sync + 'static> Operator for Id<T> {
    type State = ();
    type Inputs = Input<T>;
    type Output = T;

    fn init(self, inputs: &T, _timestamp: Instant) -> ((), T) {
        ((), inputs.clone())
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        inputs: &T,
        output: &mut T,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        output.clone_from(inputs);
        true
    }
}
