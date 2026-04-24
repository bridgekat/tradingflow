//! Record operator — ported from `src/operators/record.rs`.
//!
//! Each tick the scheduler clones the last committed `Series<T>` slot
//! (via [`OutputStore::alloc_fresh`](super::super::scenario::node::OutputStore::alloc_fresh))
//! and passes it to `compute` as the output buffer; `compute` appends one
//! element and returns `true`, so the committed slot at tick `k` is the
//! full history `[e0, e1, ..., e_k]`.  Auto-GC reclaims older snapshots
//! once no reader holds them.

use super::super::data::{Array, Input, InputTypes, Instant, Scalar, Series};
use super::super::operator::Operator;

pub struct Record<T: Scalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Record<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> Default for Record<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> Operator for Record<T> {
    type State = ();
    type Inputs = Input<Array<T>>;
    type Output = Series<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> ((), Series<T>) {
        ((), Series::new(inputs.shape()))
    }

    fn compute(
        _state: &mut (),
        inputs: &Array<T>,
        output: &mut Series<T>,
        timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        output.push(timestamp, inputs.as_slice());
        true
    }
}
