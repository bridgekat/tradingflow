//! Filter operator — whole-element filter by predicate.

use std::marker::PhantomData;

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar};

/// Filter operator: passes or drops the entire input array based on a predicate.
///
/// When the predicate returns `true`, the input is copied to the output and
/// `compute` returns `true`. Otherwise the output is left unchanged and
/// `compute` returns `false`.
pub struct Filter<T: Scalar, F: Fn(&Array<T>) -> bool> {
    predicate: F,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, F: Fn(&Array<T>) -> bool> Filter<T, F> {
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, F: Fn(&Array<T>) -> bool + Send + 'static> Operator for Filter<T, F> {
    type State = Self;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (Self, Array<T>) {
        (self, inputs.clone())
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        if (state.predicate)(inputs) {
            output.as_mut_slice().clone_from_slice(inputs.as_slice());
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instant;
    use crate::operator::Operator;

    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    #[test]
    fn passes() {
        let a = Array::scalar(5.0_f64);
        let (mut s, mut o) = Filter::new(|v: &Array<f64>| v[0] > 3.0).init(&a, Instant::MIN);
        assert!(Filter::compute(
            &mut s,
            &a,
            &mut o,
            ts(1),
            false
        ));
        assert_eq!(o.as_slice(), &[5.0]);
    }

    #[test]
    fn drops() {
        let a = Array::scalar(1.0_f64);
        let (mut s, mut o) = Filter::new(|v: &Array<f64>| v[0] > 3.0).init(&a, Instant::MIN);
        assert!(!Filter::compute(
            &mut s,
            &a,
            &mut o,
            ts(1),
            false
        ));
    }

    #[test]
    fn vector_sum() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let (mut s, mut o) = Filter::new(|v: &Array<f64>| v.as_slice().iter().sum::<f64>() > 5.0)
            .init(&a, Instant::MIN);
        assert!(Filter::compute(
            &mut s,
            &a,
            &mut o,
            ts(1),
            false
        ));
    }

    #[test]
    fn multi_step() {
        let a = Array::scalar(5.0_f64);
        let (mut s, mut o) = Filter::new(|v: &Array<f64>| v[0] > 3.0).init(&a, Instant::MIN);
        assert!(Filter::compute(
            &mut s,
            &Array::scalar(5.0),
            &mut o,
            ts(1),
            false
        ));
        assert!(!Filter::compute(
            &mut s,
            &Array::scalar(1.0),
            &mut o,
            ts(2),
            false
        ));
        assert_eq!(o[0], 5.0);
        assert!(Filter::compute(
            &mut s,
            &Array::scalar(10.0),
            &mut o,
            ts(3),
            false
        ));
        assert_eq!(o[0], 10.0);
    }
}
