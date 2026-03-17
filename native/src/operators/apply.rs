//! Apply operator — applies a closure to the latest values of N input
//! observables.
//!
//! This is the workhorse operator for element-wise arithmetic.  The closure
//! receives `&[&[T]]` (one slice per input, each of length = stride) and
//! writes into `&mut [T]`.

use std::marker::PhantomData;
use std::ops;

use crate::observable::Observable;
use crate::operator::Operator;

/// Stateless operator that applies `F` to the latest values of its inputs.
pub struct Apply<T: Copy, F: Fn(&[&[T]], &mut [T])> {
    func: F,
    _phantom: PhantomData<T>,
}

impl<T: Copy, F: Fn(&[&[T]], &mut [T])> Apply<T, F> {
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy, F: Fn(&[&[T]], &mut [T])> Operator for Apply<T, F> {
    type Inputs<'a>
        = &'a [&'a Observable<T>]
    where
        Self: 'a;
    type Output = T;

    #[inline(always)]
    fn compute(
        &mut self,
        _timestamp: i64,
        inputs: &[&Observable<T>],
        out: &mut [T],
    ) -> bool {
        // Observables always have values — no emptiness check needed.
        // Gather last-value slices.  Stack-allocate up to 8 inputs.
        let mut buf = [&[] as &[T]; 8];
        if inputs.len() <= 8 {
            for (i, obs) in inputs.iter().enumerate() {
                buf[i] = obs.last();
            }
            (self.func)(&buf[..inputs.len()], out);
        } else {
            let v: Vec<&[T]> = inputs.iter().map(|o| o.last()).collect();
            (self.func)(&v, out);
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Factory functions for common arithmetic operators
// ---------------------------------------------------------------------------

/// Element-wise addition of two observables.
pub fn add<T>() -> Apply<T, impl Fn(&[&[T]], &mut [T])>
where
    T: Copy + ops::Add<Output = T>,
{
    Apply::new(|inputs: &[&[T]], out: &mut [T]| {
        let (a, b) = (inputs[0], inputs[1]);
        for i in 0..out.len() {
            out[i] = a[i] + b[i];
        }
    })
}

/// Element-wise subtraction of two observables.
pub fn subtract<T>() -> Apply<T, impl Fn(&[&[T]], &mut [T])>
where
    T: Copy + ops::Sub<Output = T>,
{
    Apply::new(|inputs: &[&[T]], out: &mut [T]| {
        let (a, b) = (inputs[0], inputs[1]);
        for i in 0..out.len() {
            out[i] = a[i] - b[i];
        }
    })
}

/// Element-wise multiplication of two observables.
pub fn multiply<T>() -> Apply<T, impl Fn(&[&[T]], &mut [T])>
where
    T: Copy + ops::Mul<Output = T>,
{
    Apply::new(|inputs: &[&[T]], out: &mut [T]| {
        let (a, b) = (inputs[0], inputs[1]);
        for i in 0..out.len() {
            out[i] = a[i] * b[i];
        }
    })
}

/// Element-wise division of two observables.
pub fn divide<T>() -> Apply<T, impl Fn(&[&[T]], &mut [T])>
where
    T: Copy + ops::Div<Output = T>,
{
    Apply::new(|inputs: &[&[T]], out: &mut [T]| {
        let (a, b) = (inputs[0], inputs[1]);
        for i in 0..out.len() {
            out[i] = a[i] / b[i];
        }
    })
}

/// Element-wise negation of an observable.
pub fn negate<T>() -> Apply<T, impl Fn(&[&[T]], &mut [T])>
where
    T: Copy + ops::Neg<Output = T>,
{
    Apply::new(|inputs: &[&[T]], out: &mut [T]| {
        let a = inputs[0];
        for i in 0..out.len() {
            out[i] = -a[i];
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observable;

    fn make_pair() -> (Observable<f64>, Observable<f64>) {
        let a = Observable::new(&[], &[20.0]);
        let b = Observable::new(&[], &[7.0]);
        (a, b)
    }

    #[test]
    fn test_add() {
        let (a, b) = make_pair();
        let mut op = add::<f64>();
        let mut out = [0.0];
        assert!(op.compute(2, &[&a, &b], &mut out));
        assert_eq!(out, [27.0]); // 20 + 7
    }

    #[test]
    fn test_subtract() {
        let (a, b) = make_pair();
        let mut op = subtract::<f64>();
        let mut out = [0.0];
        assert!(op.compute(2, &[&a, &b], &mut out));
        assert_eq!(out, [13.0]); // 20 - 7
    }

    #[test]
    fn test_always_produces_output() {
        // Observables always have values, so compute always returns true.
        let a = Observable::<f64>::new(&[], &[0.0]);
        let b = Observable::<f64>::new(&[], &[0.0]);
        let mut op = add::<f64>();
        let mut out = [0.0];
        assert!(op.compute(0, &[&a, &b], &mut out));
        assert_eq!(out, [0.0]); // 0.0 + 0.0
    }

    #[test]
    fn test_strided_add() {
        let a = Observable::new(&[2], &[1.0, 2.0]);
        let b = Observable::new(&[2], &[10.0, 20.0]);
        let mut op = add::<f64>();
        let mut out = [0.0, 0.0];
        assert!(op.compute(1, &[&a, &b], &mut out));
        assert_eq!(out, [11.0, 22.0]);
    }
}
