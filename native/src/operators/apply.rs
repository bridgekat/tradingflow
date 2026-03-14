//! Apply operator — applies a closure to the latest values of N input series.
//!
//! This is the workhorse operator for element-wise arithmetic.  The closure
//! receives `&[&[T]]` (one slice per input, each of length = stride) and
//! writes into `&mut [T]`.

use std::marker::PhantomData;
use std::ops;

use crate::operator::Operator;
use crate::series::Series;

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
        = &'a [&'a Series<T>]
    where
        Self: 'a;
    type Output = T;

    #[inline(always)]
    fn compute(
        &mut self,
        _timestamp: i64,
        inputs: &[&Series<T>],
        out: &mut [T],
    ) -> bool {
        // All inputs must have at least one element.
        for s in inputs {
            if s.is_empty() {
                return false;
            }
        }
        // Gather last-value slices.  Stack-allocate up to 8 inputs.
        let mut buf = [&[] as &[T]; 8];
        if inputs.len() <= 8 {
            for (i, s) in inputs.iter().enumerate() {
                buf[i] = s.last();
            }
            (self.func)(&buf[..inputs.len()], out);
        } else {
            let v: Vec<&[T]> = inputs.iter().map(|s| s.last()).collect();
            (self.func)(&v, out);
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Factory functions for common arithmetic operators
// ---------------------------------------------------------------------------

/// Element-wise addition of two series.
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

/// Element-wise subtraction of two series.
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

/// Element-wise multiplication of two series.
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

/// Element-wise division of two series.
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

/// Element-wise negation of a series.
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
    use crate::series::Series;

    fn make_pair() -> (Series<f64>, Series<f64>) {
        let mut a = Series::new(&[]);
        let mut b = Series::new(&[]);
        a.append_unchecked(1, &[10.0]);
        a.append_unchecked(2, &[20.0]);
        b.append_unchecked(1, &[3.0]);
        b.append_unchecked(2, &[7.0]);
        (a, b)
    }

    #[test]
    fn test_add() {
        let (a, b) = make_pair();
        let mut op = add::<f64>();
        let mut out = [0.0];
        assert!(op.compute(2, &[&a, &b], &mut out));
        assert_eq!(out, [27.0]); // last values: 20 + 7
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
    fn test_empty_input_returns_false() {
        let a = Series::<f64>::new(&[]);
        let b = Series::<f64>::new(&[]);
        let mut op = add::<f64>();
        let mut out = [0.0];
        assert!(!op.compute(0, &[&a, &b], &mut out));
    }

    #[test]
    fn test_strided_add() {
        let mut a = Series::new(&[2]);
        let mut b = Series::new(&[2]);
        a.append_unchecked(1, &[1.0, 2.0]);
        b.append_unchecked(1, &[10.0, 20.0]);
        let mut op = add::<f64>();
        let mut out = [0.0, 0.0];
        assert!(op.compute(1, &[&a, &b], &mut out));
        assert_eq!(out, [11.0, 22.0]);
    }
}
