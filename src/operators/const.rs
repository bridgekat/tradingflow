//! Constant operator — a 0-input node holding a fixed initial value.

use crate::data::Instant;
use crate::{Notify, Operator};

/// A 0-input operator that holds a constant value.
///
/// The output is set once at init and never changes.  Compute returns
/// `true` so that, when the operator is bound to a clock trigger, each
/// clock tick propagates a production signal downstream.
///
/// Without a trigger the operator has no triggers and compute is never
/// invoked, so the return value is inert.  The value can still be mutated
/// externally via [`Scenario::value_mut`](crate::Scenario::value_mut).
pub struct Const<T: Send + 'static> {
    value: T,
}

impl<T: Send + 'static> Const<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Send + 'static> Operator for Const<T> {
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
        _notify: &Notify<'_>,
    ) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Array;
    use crate::operator::Operator;

    #[test]
    fn const_scalar() {
        let (mut s, o) = Const::new(Array::scalar(42.0_f64)).init((), Instant::MIN);
        assert_eq!(o.as_slice(), &[42.0]);
        let mut o = o;
        assert!(Const::<Array<f64>>::compute(
            &mut s,
            (),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0)
        ));
        assert_eq!(o.as_slice(), &[42.0]);
    }

    #[test]
    fn const_arbitrary_type() {
        use std::collections::BTreeMap;
        let (_, o) = Const::new(BTreeMap::<String, f64>::new()).init((), Instant::MIN);
        assert!(o.is_empty());
    }
}
