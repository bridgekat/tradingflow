//! Apply operator and built-in element-wise operators.
//!
//! * [`Apply<T, F>`] — closure-based stateless operator with homogeneous
//!   observable inputs.  Registered via [`Scenario::add_slice_operator`].
//! * [`Add<T>`], [`Subtract<T>`], [`Multiply<T>`], [`Divide<T>`],
//!   [`Negate<T>`] — concrete operators with typed tuple inputs.  Registered
//!   via [`Scenario::add_operator`] with the appropriate [`InputTuple`].

use std::marker::PhantomData;
use std::ops;

use crate::observable::Observable;
use crate::operator::Operator;

// ---------------------------------------------------------------------------
// Apply<T, F> — homogeneous closure operator
// ---------------------------------------------------------------------------

/// Stateless operator that applies `F` to N homogeneous observable inputs.
///
/// `F` receives `&[&[T]]` (one flat value slice per input) and writes into
/// `&mut [T]` (the output buffer).  It always produces output.
///
/// Register via [`Scenario::add_slice_operator`].
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
// Concrete element-wise operators (InputTuple-based, heterogeneous-capable)
// ---------------------------------------------------------------------------

/// Element-wise addition: `out[i] = a[i] + b[i]`.
pub struct Add<T: Copy>(PhantomData<T>);

impl<T: Copy + ops::Add<Output = T>> Operator for Add<T> {
    type Inputs<'a>
        = (&'a Observable<T>, &'a Observable<T>)
    where
        Self: 'a;
    type Output = T;

    #[inline(always)]
    fn compute(&mut self, _ts: i64, inputs: Self::Inputs<'_>, out: &mut [T]) -> bool {
        let (a, b) = inputs;
        let (a, b) = (a.last(), b.last());
        for i in 0..out.len() {
            out[i] = a[i] + b[i];
        }
        true
    }
}

/// Element-wise subtraction: `out[i] = a[i] - b[i]`.
pub struct Subtract<T: Copy>(PhantomData<T>);

impl<T: Copy + ops::Sub<Output = T>> Operator for Subtract<T> {
    type Inputs<'a>
        = (&'a Observable<T>, &'a Observable<T>)
    where
        Self: 'a;
    type Output = T;

    #[inline(always)]
    fn compute(&mut self, _ts: i64, inputs: Self::Inputs<'_>, out: &mut [T]) -> bool {
        let (a, b) = inputs;
        let (a, b) = (a.last(), b.last());
        for i in 0..out.len() {
            out[i] = a[i] - b[i];
        }
        true
    }
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
pub struct Multiply<T: Copy>(PhantomData<T>);

impl<T: Copy + ops::Mul<Output = T>> Operator for Multiply<T> {
    type Inputs<'a>
        = (&'a Observable<T>, &'a Observable<T>)
    where
        Self: 'a;
    type Output = T;

    #[inline(always)]
    fn compute(&mut self, _ts: i64, inputs: Self::Inputs<'_>, out: &mut [T]) -> bool {
        let (a, b) = inputs;
        let (a, b) = (a.last(), b.last());
        for i in 0..out.len() {
            out[i] = a[i] * b[i];
        }
        true
    }
}

/// Element-wise division: `out[i] = a[i] / b[i]`.
pub struct Divide<T: Copy>(PhantomData<T>);

impl<T: Copy + ops::Div<Output = T>> Operator for Divide<T> {
    type Inputs<'a>
        = (&'a Observable<T>, &'a Observable<T>)
    where
        Self: 'a;
    type Output = T;

    #[inline(always)]
    fn compute(&mut self, _ts: i64, inputs: Self::Inputs<'_>, out: &mut [T]) -> bool {
        let (a, b) = inputs;
        let (a, b) = (a.last(), b.last());
        for i in 0..out.len() {
            out[i] = a[i] / b[i];
        }
        true
    }
}

/// Element-wise negation: `out[i] = -a[i]`.
pub struct Negate<T: Copy>(PhantomData<T>);

impl<T: Copy + ops::Neg<Output = T>> Operator for Negate<T> {
    type Inputs<'a>
        = (&'a Observable<T>,)
    where
        Self: 'a;
    type Output = T;

    #[inline(always)]
    fn compute(&mut self, _ts: i64, inputs: Self::Inputs<'_>, out: &mut [T]) -> bool {
        let (a,) = inputs;
        let a = a.last();
        for i in 0..out.len() {
            out[i] = -a[i];
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Create an [`Add`] operator.
pub fn add<T: Copy + ops::Add<Output = T>>() -> Add<T> {
    Add(PhantomData)
}

/// Create a [`Subtract`] operator.
pub fn subtract<T: Copy + ops::Sub<Output = T>>() -> Subtract<T> {
    Subtract(PhantomData)
}

/// Create a [`Multiply`] operator.
pub fn multiply<T: Copy + ops::Mul<Output = T>>() -> Multiply<T> {
    Multiply(PhantomData)
}

/// Create a [`Divide`] operator.
pub fn divide<T: Copy + ops::Div<Output = T>>() -> Divide<T> {
    Divide(PhantomData)
}

/// Create a [`Negate`] operator.
pub fn negate<T: Copy + ops::Neg<Output = T>>() -> Negate<T> {
    Negate(PhantomData)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::Observable;

    #[test]
    fn test_add() {
        let a = Observable::new(&[], &[20.0]);
        let b = Observable::new(&[], &[7.0]);
        let mut op = add::<f64>();
        let mut out = [0.0];
        assert!(op.compute(2, (&a, &b), &mut out));
        assert_eq!(out, [27.0]);
    }

    #[test]
    fn test_subtract() {
        let a = Observable::new(&[], &[20.0]);
        let b = Observable::new(&[], &[7.0]);
        let mut op = subtract::<f64>();
        let mut out = [0.0];
        assert!(op.compute(2, (&a, &b), &mut out));
        assert_eq!(out, [13.0]);
    }

    #[test]
    fn test_multiply() {
        let a = Observable::new(&[], &[4.0]);
        let b = Observable::new(&[], &[5.0]);
        let mut op = multiply::<f64>();
        let mut out = [0.0];
        assert!(op.compute(1, (&a, &b), &mut out));
        assert_eq!(out, [20.0]);
    }

    #[test]
    fn test_divide() {
        let a = Observable::new(&[], &[20.0]);
        let b = Observable::new(&[], &[4.0]);
        let mut op = divide::<f64>();
        let mut out = [0.0];
        assert!(op.compute(1, (&a, &b), &mut out));
        assert_eq!(out, [5.0]);
    }

    #[test]
    fn test_negate() {
        let a = Observable::new(&[], &[7.0]);
        let mut op = negate::<f64>();
        let mut out = [0.0];
        assert!(op.compute(1, (&a,), &mut out));
        assert_eq!(out, [-7.0]);
    }

    #[test]
    fn test_strided_add() {
        let a = Observable::new(&[2], &[1.0, 2.0]);
        let b = Observable::new(&[2], &[10.0, 20.0]);
        let mut op = add::<f64>();
        let mut out = [0.0, 0.0];
        assert!(op.compute(1, (&a, &b), &mut out));
        assert_eq!(out, [11.0, 22.0]);
    }

    #[test]
    fn test_apply_closure() {
        let a = Observable::new(&[], &[3.0]);
        let b = Observable::new(&[], &[4.0]);
        let mut op = Apply::new(|inputs: &[&[f64]], out: &mut [f64]| {
            // Pythagorean: sqrt(a^2 + b^2)
            out[0] = (inputs[0][0] * inputs[0][0] + inputs[1][0] * inputs[1][0]).sqrt();
        });
        let mut out = [0.0];
        assert!(op.compute(1, &[&a, &b], &mut out));
        assert_eq!(out, [5.0]);
    }

    #[test]
    fn test_always_produces_output() {
        let a = Observable::<f64>::new(&[], &[0.0]);
        let b = Observable::<f64>::new(&[], &[0.0]);
        let mut op = add::<f64>();
        let mut out = [0.0];
        assert!(op.compute(0, (&a, &b), &mut out));
        assert_eq!(out, [0.0]);
    }
}
