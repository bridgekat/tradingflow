//! Where operator — element-wise conditional replacement.
//!
//! For each element in the input, keeps the value if the per-element
//! condition returns `true`, otherwise replaces it with a fill value.
//! Unlike [`Filter`](super::Filter), `Where` always produces output.

use std::marker::PhantomData;

use crate::operator::Operator;
use crate::store::{ElementViewMut, Store};
use crate::types::Scalar;

/// Element-wise conditional operator.
///
/// `F` is a per-element predicate: `fn(T) -> bool`.  Elements where `F`
/// returns `false` are replaced with `fill`.
pub struct Where<T: Scalar, F: Fn(T) -> bool> {
    condition: F,
    fill: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, F: Fn(T) -> bool> Where<T, F> {
    pub fn new(condition: F, fill: T) -> Self {
        Self {
            condition,
            fill,
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, F: Fn(T) -> bool + Send + 'static> Operator for Where<T, F> {
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
        let out = output.values;
        for i in 0..out.len() {
            out[i] = if (state.condition)(input[i].clone()) {
                input[i].clone()
            } else {
                state.fill.clone()
            };
        }
        true
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
    fn where_keeps_passing() {
        let store = Store::element(&[3], &[1.0, 5.0, 2.0]);
        let mut state = Where::new(|v: f64| v > 3.0, 0.0);
        let mut out = Store::element(&[3], &[0.0; 3]);
        out.push_default(1);
        Where::compute(&mut state, (&store,), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[0.0, 5.0, 0.0]);
    }

    #[test]
    fn where_all_pass() {
        let store = Store::element(&[2], &[10.0, 20.0]);
        let mut state = Where::new(|v: f64| v > 0.0, -1.0);
        let mut out = Store::element(&[2], &[0.0; 2]);
        out.push_default(1);
        Where::compute(&mut state, (&store,), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[10.0, 20.0]);
    }

    #[test]
    fn where_none_pass() {
        let store = Store::element(&[2], &[-1.0, -2.0]);
        let mut state = Where::new(|v: f64| v > 0.0, 0.0);
        let mut out = Store::element(&[2], &[0.0; 2]);
        out.push_default(1);
        Where::compute(&mut state, (&store,), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[0.0, 0.0]);
    }

    #[test]
    fn where_with_nan_fill() {
        let store = Store::element(&[3], &[1.0, f64::NAN, 3.0]);
        let mut state = Where::new(|v: f64| !v.is_nan(), f64::NAN);
        let mut out = Store::element(&[3], &[0.0; 3]);
        out.push_default(1);
        Where::compute(&mut state, (&store,), out.current_view_mut());
        out.commit();
        assert_eq!(out.current()[0], 1.0);
        assert!(out.current()[1].is_nan());
        assert_eq!(out.current()[2], 3.0);
    }
}
