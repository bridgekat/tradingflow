//! Where operator — element-wise conditional replacement.
//!
//! For each element in the input, keeps the value if the per-element
//! condition returns `true`, otherwise replaces it with a fill value.
//! Unlike [`Filter`](super::Filter), `Where` always produces output.
//!
//! Register via [`Scenario::add_operator`] with `(Obs<T>,)` input.

use std::marker::PhantomData;

use crate::observable::Observable;
use crate::operator::Operator;

/// Element-wise conditional operator.
///
/// `F` is a per-element predicate: `fn(T) -> bool`.  Elements where `F`
/// returns `false` are replaced with `fill`.
pub struct Where<T: Copy, F: Fn(T) -> bool> {
    condition: F,
    fill: T,
    _phantom: PhantomData<T>,
}

impl<T: Copy, F: Fn(T) -> bool> Where<T, F> {
    pub fn new(condition: F, fill: T) -> Self {
        Self {
            condition,
            fill,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy + 'static, F: Fn(T) -> bool + 'static> Operator for Where<T, F> {
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
        for i in 0..out.len() {
            out[i] = if (self.condition)(input[i]) {
                input[i]
            } else {
                self.fill
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
    use crate::observable::Observable;

    #[test]
    fn where_keeps_passing() {
        let obs = Observable::new(&[3], &[1.0, 5.0, 2.0]);
        let mut op = Where::new(|v: f64| v > 3.0, 0.0);
        let mut out = [0.0; 3];
        assert!(op.compute(1, (&obs,), &mut out));
        assert_eq!(out, [0.0, 5.0, 0.0]);
    }

    #[test]
    fn where_all_pass() {
        let obs = Observable::new(&[2], &[10.0, 20.0]);
        let mut op = Where::new(|v: f64| v > 0.0, -1.0);
        let mut out = [0.0; 2];
        assert!(op.compute(1, (&obs,), &mut out));
        assert_eq!(out, [10.0, 20.0]);
    }

    #[test]
    fn where_none_pass() {
        let obs = Observable::new(&[2], &[-1.0, -2.0]);
        let mut op = Where::new(|v: f64| v > 0.0, 0.0);
        let mut out = [0.0; 2];
        assert!(op.compute(1, (&obs,), &mut out));
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn where_with_nan_fill() {
        let obs = Observable::new(&[3], &[1.0, f64::NAN, 3.0]);
        let mut op = Where::new(|v: f64| !v.is_nan(), f64::NAN);
        let mut out = [0.0; 3];
        assert!(op.compute(1, (&obs,), &mut out));
        assert_eq!(out[0], 1.0);
        assert!(out[1].is_nan());
        assert_eq!(out[2], 3.0);
    }
}
