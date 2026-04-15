//! Map operators — apply functions to transform inputs into outputs.
//!
//! - [`Map`] — allocating: `Fn(&S) -> T`.
//! - [`MapInplace`] — in-place: `Fn(&S, &mut T) -> bool`.

use crate::data::Instant;
use crate::{Input, Notify, Operator};

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
    type Inputs = (Input<S>,);
    type Output = T;

    fn init(self, inputs: (&S,), _timestamp: Instant) -> (Self, T) {
        let output = (self.f)(inputs.0);
        (self, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&S,),
        output: &mut T,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        *output = (state.f)(inputs.0);
        true
    }
}

/// In-place map operator: reads input `S`, writes into output `T` via
/// a mutable reference, and returns whether to propagate.
///
/// Unlike [`Map`], the function does not allocate a new output — it
/// receives `(&S, &mut T)` and mutates the existing output in place.
/// The return value controls downstream propagation.
pub struct MapInplace<S, T, F>
where
    S: Clone + Send + 'static,
    T: Clone + Send + 'static,
    F: Fn(&S, &mut T) -> bool + Send + 'static,
{
    f: F,
    initial: T,
    _phantom: std::marker::PhantomData<S>,
}

impl<S, T, F> MapInplace<S, T, F>
where
    S: Clone + Send + 'static,
    T: Clone + Send + 'static,
    F: Fn(&S, &mut T) -> bool + Send + 'static,
{
    pub fn new(f: F, initial: T) -> Self {
        Self {
            f,
            initial,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S, T, F> Operator for MapInplace<S, T, F>
where
    S: Clone + Send + 'static,
    T: Clone + Send + 'static,
    F: Fn(&S, &mut T) -> bool + Send + 'static,
{
    type State = Self;
    type Inputs = (Input<S>,);
    type Output = T;

    fn init(self, inputs: (&S,), _timestamp: Instant) -> (Self, T) {
        let mut output = self.initial.clone();
        (self.f)(inputs.0, &mut output);
        (self, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&S,),
        output: &mut T,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        (state.f)(inputs.0, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Array;
    use crate::operator::Operator;
    use crate::data::Instant;

    fn ts(n: i64) -> Instant { Instant::from_nanos(n) }

    #[test]
    fn map_scalar_double() {
        let a = Array::scalar(5.0_f64);
        let (mut s, mut o) = Map::new(|a: &Array<f64>| {
            let mut out = a.clone();
            out[0] *= 2.0;
            out
        })
        .init((&a,), Instant::MIN);
        assert_eq!(o.as_slice(), &[10.0]);

        let b = Array::scalar(3.0_f64);
        Map::compute(&mut s, (&b,), &mut o, ts(1), &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[6.0]);
    }

    #[test]
    fn map_type_change() {
        // Array<f64> → String
        let a = Array::scalar(42.0_f64);
        let (mut s, mut o) =
            Map::new(|a: &Array<f64>| format!("{:.0}", a[0])).init((&a,), Instant::MIN);
        assert_eq!(o, "42");

        let b = Array::scalar(99.0_f64);
        Map::compute(&mut s, (&b,), &mut o, ts(1), &Notify::new(&[], 0));
        assert_eq!(o, "99");
    }

    #[test]
    fn map_vector_sum() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let (mut s, mut o) = Map::new(|a: &Array<f64>| {
            Array::scalar(a.as_slice().iter().sum::<f64>())
        })
        .init((&a,), Instant::MIN);
        assert_eq!(o.as_slice(), &[6.0]);

        let b = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        Map::compute(&mut s, (&b,), &mut o, ts(1), &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[60.0]);
    }

    #[test]
    fn map_inplace_double() {
        let a = Array::scalar(5.0_f64);
        let (mut s, mut o) =
            MapInplace::new(
                |inp: &Array<f64>, out: &mut Array<f64>| {
                    out[0] = inp[0] * 2.0;
                    true
                },
                Array::scalar(0.0),
            )
            .init((&a,), Instant::MIN);
        assert_eq!(o.as_slice(), &[10.0]);

        let b = Array::scalar(3.0);
        MapInplace::compute(&mut s, (&b,), &mut o, ts(1), &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[6.0]);
    }

    #[test]
    fn map_inplace_conditional() {
        // Only propagate when input > 3.
        let a = Array::scalar(1.0_f64);
        let (mut s, mut o) =
            MapInplace::new(
                |inp: &Array<f64>, out: &mut Array<f64>| {
                    if inp[0] > 3.0 {
                        out[0] = inp[0];
                        true
                    } else {
                        false
                    }
                },
                Array::scalar(0.0),
            )
            .init((&a,), Instant::MIN);

        // Input <= 3 → returns false.
        let b = Array::scalar(2.0);
        assert!(!MapInplace::compute(
            &mut s,
            (&b,),
            &mut o,
            ts(1),
            &Notify::new(&[], 0)
        ));

        // Input > 3 → returns true.
        let c = Array::scalar(5.0);
        assert!(MapInplace::compute(
            &mut s,
            (&c,),
            &mut o,
            ts(2),
            &Notify::new(&[], 0)
        ));
        assert_eq!(o.as_slice(), &[5.0]);
    }
}
