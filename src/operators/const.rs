//! Constant operator — a 0-input node holding a fixed initial value.

use crate::operator::Notify;
use crate::Operator;

/// A 0-input operator that holds a constant value.
///
/// The output is set once at init and never changes (compute always returns
/// `false`).  This is the canonical way to inject an initial value into the
/// DAG; the value can still be mutated externally via
/// [`Scenario::value_mut`](crate::Scenario::value_mut).
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

    fn init(self, _inputs: (), _timestamp: i64) -> ((), T) {
        ((), self.value)
    }

    #[inline(always)]
    fn compute(_state: &mut (), _inputs: (), _output: &mut T, _timestamp: i64, _notify: &Notify<'_>) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::operator::Operator;

    #[test]
    fn const_scalar() {
        let (mut s, o) = Const::new(Array::scalar(42.0_f64)).init((), i64::MIN);
        assert_eq!(o.as_slice(), &[42.0]);
        let mut o = o;
        assert!(!Const::<Array<f64>>::compute(&mut s, (), &mut o, 1, &Notify::new(&[], &[])));
        assert_eq!(o.as_slice(), &[42.0]);
    }

    #[test]
    fn const_arbitrary_type() {
        use std::collections::BTreeMap;
        let (_, o) = Const::new(BTreeMap::<String, f64>::new()).init((), i64::MIN);
        assert!(o.is_empty());
    }
}
