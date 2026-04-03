//! Where operator — element-wise conditional replacement.

use std::marker::PhantomData;

use crate::{Array, Notify, Operator, Scalar};

/// Element-wise conditional operator: keeps the value if the condition
/// returns `true`, otherwise replaces it with `fill`.
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
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: i64) -> (Self, Array<T>) {
        (self, inputs.0.clone())
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        let a = inputs.0.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = if (state.condition)(a[i].clone()) {
                a[i].clone()
            } else {
                state.fill.clone()
            };
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;

    #[test]
    fn mixed() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 5.0, 2.0]);
        let (mut s, mut o) = Where::new(|v: f64| v > 3.0, 0.0).init((&a,), i64::MIN);
        Where::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[0.0, 5.0, 0.0]);
    }

    #[test]
    fn all_pass() {
        let a = Array::from_vec(&[2], vec![10.0_f64, 20.0]);
        let (mut s, mut o) = Where::new(|v: f64| v > 0.0, -1.0).init((&a,), i64::MIN);
        Where::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[10.0, 20.0]);
    }

    #[test]
    fn none_pass() {
        let a = Array::from_vec(&[2], vec![-1.0_f64, -2.0]);
        let (mut s, mut o) = Where::new(|v: f64| v > 0.0, 0.0).init((&a,), i64::MIN);
        Where::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[0.0, 0.0]);
    }
}
