//! Map operator — applies a function to transform input into output.

use crate::{Notify, Operator};

/// Map operator: applies a function `S → T` to the input on each tick.
///
/// Generic over any input `S` and output `T` that satisfy
/// `Clone + Send + 'static`.  The function receives an immutable
/// reference to the input and must return the output by value.
pub struct Map<S, T, F>
where
    S: Clone + Send + 'static,
    T: Clone + Send + 'static,
    F: Fn(&S) -> T + Send + 'static,
{
    f: F,
    _phantom: std::marker::PhantomData<(S, T)>,
}

impl<S, T, F> Map<S, T, F>
where
    S: Clone + Send + 'static,
    T: Clone + Send + 'static,
    F: Fn(&S) -> T + Send + 'static,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S, T, F> Operator for Map<S, T, F>
where
    S: Clone + Send + 'static,
    T: Clone + Send + 'static,
    F: Fn(&S) -> T + Send + 'static,
{
    type State = Self;
    type Inputs = (S,);
    type Output = T;

    fn init(self, inputs: (&S,), _timestamp: i64) -> (Self, T) {
        let output = (self.f)(inputs.0);
        (self, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&S,),
        output: &mut T,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        *output = (state.f)(inputs.0);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::operator::Operator;

    #[test]
    fn map_scalar_double() {
        let a = Array::scalar(5.0_f64);
        let (mut s, mut o) = Map::new(|a: &Array<f64>| {
            let mut out = a.clone();
            out[0] *= 2.0;
            out
        })
        .init((&a,), i64::MIN);
        assert_eq!(o.as_slice(), &[10.0]);

        let b = Array::scalar(3.0_f64);
        Map::compute(&mut s, (&b,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[6.0]);
    }

    #[test]
    fn map_type_change() {
        // Array<f64> → String
        let a = Array::scalar(42.0_f64);
        let (mut s, mut o) =
            Map::new(|a: &Array<f64>| format!("{:.0}", a[0])).init((&a,), i64::MIN);
        assert_eq!(o, "42");

        let b = Array::scalar(99.0_f64);
        Map::compute(&mut s, (&b,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o, "99");
    }

    #[test]
    fn map_vector_sum() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let (mut s, mut o) = Map::new(|a: &Array<f64>| {
            Array::scalar(a.as_slice().iter().sum::<f64>())
        })
        .init((&a,), i64::MIN);
        assert_eq!(o.as_slice(), &[6.0]);

        let b = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        Map::compute(&mut s, (&b,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[60.0]);
    }
}
