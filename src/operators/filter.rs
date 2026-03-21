//! Filter operator — whole-element filter by predicate.
//!
//! Drops the entire element (returns `false` from `compute`) when the
//! predicate returns `false`.  When the predicate passes, the input value
//! is copied to the output unchanged.

use std::marker::PhantomData;

use crate::operator::Operator;
use crate::store::{ElementViewMut, Store};
use crate::types::Scalar;

/// Filter operator: passes or drops the entire element based on a predicate.
///
/// `F` receives the flat value slice `&[T]` and returns `true` to keep the
/// element or `false` to drop it.  When dropped, the operator returns
/// `false` from `compute`, so the store is not appended to and downstream
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

impl<T: Scalar, F: Fn(&[T]) -> bool + Send + 'static> Operator for Filter<T, F> {
    type State = Self;
    type Inputs = (Store<T>,);
    type Scalar = T;

    fn window_sizes(&self, _: &[&[usize]]) -> (usize,) {
        (1,)
    }

    fn default(&self, input_shapes: &[&[usize]]) -> (Box<[usize]>, Box<[T]>) {
        let shape: Box<[usize]> = input_shapes[0].into();
        let stride = shape.iter().product::<usize>();
        (shape, vec![T::default(); stride].into())
    }

    fn init(self) -> Self {
        self
    }

    #[inline(always)]
    fn compute(state: &mut Self, inputs: (&Store<T>,), output: ElementViewMut<'_, T>) -> bool {
        let input = inputs.0.current();
        if (state.predicate)(input) {
            output.values.copy_from_slice(input);
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
    use crate::store::Store;

    #[test]
    fn filter_passes() {
        let store = Store::element(&[], &[5.0]);
        let mut state = Filter::new(|v: &[f64]| v[0] > 3.0);
        let mut out = Store::element(&[], &[0.0]);
        out.push_default(1);
        let produced = Filter::compute(&mut state, (&store,), out.current_view_mut());
        if produced {
            out.commit();
        } else {
            out.rollback();
        }
        assert!(produced);
        assert_eq!(out.current(), &[5.0]);
    }

    #[test]
    fn filter_drops() {
        let store = Store::element(&[], &[1.0]);
        let mut state = Filter::new(|v: &[f64]| v[0] > 3.0);
        let mut out = Store::element(&[], &[0.0]);
        out.push_default(1);
        let produced = Filter::compute(&mut state, (&store,), out.current_view_mut());
        if produced {
            out.commit();
        } else {
            out.rollback();
        }
        assert!(!produced);
    }

    #[test]
    fn filter_strided() {
        let store = Store::element(&[3], &[1.0, 2.0, 3.0]);
        // Keep only if the sum > 5
        let mut state = Filter::new(|v: &[f64]| v.iter().sum::<f64>() > 5.0);
        let mut out = Store::element(&[3], &[0.0; 3]);
        // Sum = 6 > 5 -> keep
        out.push_default(1);
        let produced = Filter::compute(&mut state, (&store,), out.current_view_mut());
        if produced {
            out.commit();
        } else {
            out.rollback();
        }
        assert!(produced);
        assert_eq!(out.current(), &[1.0, 2.0, 3.0]);
    }
}
