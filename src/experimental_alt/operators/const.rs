//! Constant operator — a 0-input permanent-slot node.
//!
//! Ported from `src/operators/const.rs` with the added `T: Clone`
//! bound required by the wavefront runtime's `Output: Clone` trait
//! invariant.  The operator's `compute` is never invoked in the new
//! runtime because a 0-input node has no trigger edges — its initial
//! value sits forever as the single committed slot of its output
//! queue.

use super::super::data::{InputTypes, Instant};
use super::super::operator::Operator;

pub struct Const<T: Clone + Send + Sync + 'static> {
    value: T,
}

impl<T: Clone + Send + Sync + 'static> Const<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Clone + Send + Sync + 'static> Operator for Const<T> {
    type State = ();
    type Inputs = ();
    type Output = T;

    fn init(self, _inputs: (), _timestamp: Instant) -> ((), T) {
        ((), self.value)
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        _inputs: (),
        _output: &mut T,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        true
    }
}
