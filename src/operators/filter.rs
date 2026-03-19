//! Filter operator — whole-element filter by predicate.
//!
//! Drops the entire element (returns `false` from `compute`) when the
//! predicate returns `false`.  When the predicate passes, the input value
//! is copied to the output unchanged.
//!
//! Register via [`Scenario::add_operator`] with `(Obs<T>,)` input.

use std::marker::PhantomData;

use crate::observable::Observable;
use crate::operator::Operator;

/// Filter operator: passes or drops the entire element based on a predicate.
///
/// `F` receives the flat value slice `&[T]` and returns `true` to keep the
/// element or `false` to drop it.  When dropped, the operator returns
/// `false` from `compute`, so the series is not appended to and downstream
/// operators are not triggered.
pub struct Filter<T: Copy, F: Fn(&[T]) -> bool> {
    predicate: F,
    _phantom: PhantomData<T>,
}

impl<T: Copy, F: Fn(&[T]) -> bool> Filter<T, F> {
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy + 'static, F: Fn(&[T]) -> bool + 'static> Operator for Filter<T, F> {
    type Inputs<'a>
        = (&'a Observable<T>,)
    where
        Self: 'a;
    type Scalar = T;

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Box<[usize]> {
        input_shapes[0].into()
    }

    #[inline(always)]
    fn compute(&mut self, _ts: i64, inputs: (&Observable<T>,), out: &mut [T]) -> bool {
        let (obs,) = inputs;
        let input = obs.current();
        if (self.predicate)(input) {
            out.copy_from_slice(input);
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observable;

    #[test]
    fn filter_passes() {
        let obs = Observable::new(&[], &[5.0]);
        let mut op = Filter::new(|v: &[f64]| v[0] > 3.0);
        let mut out = [0.0];
        assert!(op.compute(1, (&obs,), &mut out));
        assert_eq!(out, [5.0]);
    }

    #[test]
    fn filter_drops() {
        let obs = Observable::new(&[], &[1.0]);
        let mut op = Filter::new(|v: &[f64]| v[0] > 3.0);
        let mut out = [0.0];
        assert!(!op.compute(1, (&obs,), &mut out));
    }

    #[test]
    fn filter_strided() {
        let obs = Observable::new(&[3], &[1.0, 2.0, 3.0]);
        // Keep only if the sum > 5
        let mut op = Filter::new(|v: &[f64]| v.iter().sum::<f64>() > 5.0);
        let mut out = [0.0; 3];
        // Sum = 6 > 5 → keep
        assert!(op.compute(1, (&obs,), &mut out));
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }
}
