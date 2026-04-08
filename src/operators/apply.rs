//! Apply operators — apply functions to transform tuple inputs into outputs.
//!
//! These are the multi-input counterparts of [`Map`](super::Map) and
//! [`MapInplace`](super::MapInplace):
//!
//! - [`Apply`] — allocating: `Fn(Inputs::Refs<'_>) -> T`.
//! - [`ApplyInplace`] — in-place: `Fn(Inputs::Refs<'_>, &mut T) -> bool`.

use std::marker::PhantomData;

use crate::{InputTypes, Notify, Operator};

/// Apply operator: applies a function to tuple inputs on each tick.
///
/// Unlike [`Map`](super::Map) which takes a single input, `Apply`
/// accepts any tuple of inputs matching an [`InputTypes`] implementation.
/// The function receives the full refs tuple and must return the output
/// by value.
pub struct Apply<I, T, F>
where
    I: InputTypes + Send + 'static,
    T: Clone + Send + 'static,
    F: for<'a> Fn(<I as InputTypes>::Refs<'a>) -> T + Send + 'static,
{
    f: F,
    _phantom: PhantomData<(I, T)>,
}

impl<I, T, F> Apply<I, T, F>
where
    I: InputTypes + Send + 'static,
    T: Clone + Send + 'static,
    F: for<'a> Fn(<I as InputTypes>::Refs<'a>) -> T + Send + 'static,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<I, T, F> Operator for Apply<I, T, F>
where
    I: InputTypes + Send + 'static,
    T: Clone + Send + 'static,
    F: for<'a> Fn(<I as InputTypes>::Refs<'a>) -> T + Send + 'static,
{
    type State = Self;
    type Inputs = I;
    type Output = T;

    fn init(self, inputs: <I as InputTypes>::Refs<'_>, _timestamp: i64) -> (Self, T) {
        let output = (self.f)(inputs);
        (self, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: <I as InputTypes>::Refs<'_>,
        output: &mut T,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        *output = (state.f)(inputs);
        true
    }
}

/// In-place apply operator: reads tuple inputs, writes into output `T`
/// via a mutable reference, and returns whether to propagate.
///
/// Unlike [`Apply`], the function does not allocate a new output — it
/// receives the inputs and `&mut T` and mutates the existing output.
/// The return value controls downstream propagation.
pub struct ApplyInplace<I, T, F>
where
    I: InputTypes + Send + 'static,
    T: Clone + Send + 'static,
    F: for<'a> Fn(<I as InputTypes>::Refs<'a>, &mut T) -> bool + Send + 'static,
{
    f: F,
    initial: T,
    _phantom: PhantomData<I>,
}

impl<I, T, F> ApplyInplace<I, T, F>
where
    I: InputTypes + Send + 'static,
    T: Clone + Send + 'static,
    F: for<'a> Fn(<I as InputTypes>::Refs<'a>, &mut T) -> bool + Send + 'static,
{
    pub fn new(f: F, initial: T) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

impl<I, T, F> Operator for ApplyInplace<I, T, F>
where
    I: InputTypes + Send + 'static,
    T: Clone + Send + 'static,
    F: for<'a> Fn(<I as InputTypes>::Refs<'a>, &mut T) -> bool + Send + 'static,
{
    type State = Self;
    type Inputs = I;
    type Output = T;

    fn init(self, inputs: <I as InputTypes>::Refs<'_>, _timestamp: i64) -> (Self, T) {
        let mut output = self.initial.clone();
        (self.f)(inputs, &mut output);
        (self, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: <I as InputTypes>::Refs<'_>,
        output: &mut T,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        (state.f)(inputs, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    fn apply_two_inputs_add() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let b = Array::from_vec(&[3], vec![10.0_f64, 20.0, 30.0]);

        let (mut s, mut o) = Apply::<(Array<f64>, Array<f64>), _, _>::new(|(a, b)| {
            let mut out = a.clone();
            for (o, &v) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
                *o += v;
            }
            out
        })
        .init((&a, &b), i64::MIN);

        assert_eq!(o.as_slice(), &[11.0, 22.0, 33.0]);

        let a2 = Array::from_vec(&[3], vec![100.0, 200.0, 300.0]);
        Apply::compute(&mut s, (&a2, &b), &mut o, 1, &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[110.0, 220.0, 330.0]);
    }

    #[test]
    fn apply_three_inputs() {
        let a = Array::scalar(2.0_f64);
        let b = Array::scalar(3.0_f64);
        let c = Array::scalar(4.0_f64);

        let (mut s, mut o) =
            Apply::<(Array<f64>, Array<f64>, Array<f64>), _, _>::new(|(a, b, c)| {
                Array::scalar(a[0] * b[0] + c[0])
            })
            .init((&a, &b, &c), i64::MIN);

        assert_eq!(o.as_slice(), &[10.0]); // 2*3 + 4

        let a2 = Array::scalar(5.0);
        Apply::compute(&mut s, (&a2, &b, &c), &mut o, 1, &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[19.0]); // 5*3 + 4
    }

    #[test]
    fn apply_inplace_two_inputs() {
        let a = Array::scalar(5.0_f64);
        let b = Array::scalar(3.0_f64);

        let (mut s, mut o) = ApplyInplace::<(Array<f64>, Array<f64>), _, _>::new(
            |(a, b), out| {
                out[0] = a[0] + b[0];
                true
            },
            Array::scalar(0.0),
        )
        .init((&a, &b), i64::MIN);

        assert_eq!(o.as_slice(), &[8.0]);

        let a2 = Array::scalar(10.0);
        assert!(ApplyInplace::compute(
            &mut s,
            (&a2, &b),
            &mut o,
            1,
            &Notify::new(&[], 0),
        ));
        assert_eq!(o.as_slice(), &[13.0]);
    }

    #[test]
    fn apply_inplace_conditional() {
        let a = Array::scalar(1.0_f64);
        let b = Array::scalar(2.0_f64);

        let (mut s, mut o) = ApplyInplace::<(Array<f64>, Array<f64>), _, _>::new(
            |(a, b), out| {
                let sum = a[0] + b[0];
                if sum > 5.0 {
                    out[0] = sum;
                    true
                } else {
                    false
                }
            },
            Array::scalar(0.0),
        )
        .init((&a, &b), i64::MIN);

        // 1 + 2 = 3 ≤ 5 → false
        let a2 = Array::scalar(1.0);
        assert!(!ApplyInplace::compute(
            &mut s,
            (&a2, &b),
            &mut o,
            1,
            &Notify::new(&[], 0),
        ));

        // 10 + 2 = 12 > 5 → true
        let a3 = Array::scalar(10.0);
        assert!(ApplyInplace::compute(
            &mut s,
            (&a3, &b),
            &mut o,
            2,
            &Notify::new(&[], 0),
        ));
        assert_eq!(o.as_slice(), &[12.0]);
    }
}
