//! Identity operator — passes input through unchanged.

use std::marker::PhantomData;

use crate::time::Instant;
use crate::{Notify, Operator};

/// Identity operator: clones input to output unchanged.
///
/// Generic over any `T: Clone + Send + 'static`. Useful as a trigger-gated
/// passthrough when combined with a clock trigger.
pub struct Id<T: Clone + Send + 'static> {
    _phantom: PhantomData<T>,
}

impl<T: Clone + Send + 'static> Id<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Clone + Send + 'static> Default for Id<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Send + 'static> Operator for Id<T> {
    type State = ();
    type Inputs = (T,);
    type Output = T;

    fn init(self, inputs: (&T,), _timestamp: Instant) -> ((), T) {
        ((), inputs.0.clone())
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        inputs: (&T,),
        output: &mut T,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        output.clone_from(inputs.0);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::operator::Operator;

    #[test]
    fn id_array() {
        let a = Array::scalar(42.0_f64);
        let (mut s, mut o) = Id::<Array<f64>>::new().init((&a,), Instant::MIN);
        assert_eq!(o.as_slice(), &[42.0]);
        let b = Array::scalar(99.0_f64);
        assert!(Id::<Array<f64>>::compute(
            &mut s,
            (&b,),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0)
        ));
        assert_eq!(o.as_slice(), &[99.0]);
    }

    #[test]
    fn id_string() {
        let a = String::from("hello");
        let (mut s, mut o) = Id::<String>::new().init((&a,), Instant::MIN);
        assert_eq!(o, "hello");
        let b = String::from("world");
        assert!(Id::<String>::compute(
            &mut s,
            (&b,),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0)
        ));
        assert_eq!(o, "world");
    }
}
