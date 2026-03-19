//! Apply operator and built-in element-wise operators.
//!
//! * [`Apply<T, F>`] — closure-based stateless operator with homogeneous
//!   observable inputs.  Registered via [`Scenario::add_slice_operator`].
//! * [`Elementwise2`] / [`Elementwise1`] — element-wise binary/unary ops.
//!   Factory functions: [`add`], [`subtract`], [`multiply`], [`divide`],
//!   [`negate`].

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
/// The output shape is determined by `shape_fn` from the input shapes.
///
/// Register via [`Scenario::add_slice_operator`].
pub struct Apply<T: Copy, F: Fn(&[&[T]], &mut [T]), S: Fn(&[&[usize]]) -> Box<[usize]>> {
    func: F,
    shape_fn: S,
    _phantom: PhantomData<T>,
}

impl<T: Copy, F: Fn(&[&[T]], &mut [T]), S: Fn(&[&[usize]]) -> Box<[usize]>> Apply<T, F, S> {
    pub fn new(shape_fn: S, func: F) -> Self {
        Self {
            func,
            shape_fn,
            _phantom: PhantomData,
        }
    }
}

impl<
    T: Copy + 'static,
    F: Fn(&[&[T]], &mut [T]) + 'static,
    S: Fn(&[&[usize]]) -> Box<[usize]> + 'static,
> Operator for Apply<T, F, S>
{
    type Inputs<'a>
        = Box<[&'a Observable<T>]>
    where
        Self: 'a;
    type Scalar = T;

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Box<[usize]> {
        (self.shape_fn)(input_shapes)
    }

    #[inline(always)]
    fn compute(&mut self, _timestamp: i64, inputs: Box<[&Observable<T>]>, out: &mut [T]) -> bool {
        // Gather last-value slices.  Stack-allocate up to 8 inputs.
        let mut buf = [&[] as &[T]; 8];
        if inputs.len() <= 8 {
            for (i, obs) in inputs.iter().enumerate() {
                buf[i] = obs.current();
            }
            (self.func)(&buf[..inputs.len()], out);
        } else {
            let v: Vec<&[T]> = inputs.iter().map(|o| o.current()).collect();
            (self.func)(&v, out);
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Element-wise binary operator
// ---------------------------------------------------------------------------

/// Element-wise binary operator: `out[i] = op(a[i], b[i])`.
///
/// Shape-preserving: output shape equals input shape (inputs must match).
pub struct Elementwise2<T: Copy, Op: Fn(T, T) -> T> {
    op: Op,
    _phantom: PhantomData<T>,
}

impl<T: Copy + 'static, Op: Fn(T, T) -> T + 'static> Operator for Elementwise2<T, Op> {
    type Inputs<'a>
        = (&'a Observable<T>, &'a Observable<T>)
    where
        Self: 'a;
    type Scalar = T;

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Box<[usize]> {
        input_shapes[0].into()
    }

    #[inline(always)]
    fn compute(
        &mut self,
        _ts: i64,
        inputs: (&Observable<T>, &Observable<T>),
        out: &mut [T],
    ) -> bool {
        let (a, b) = inputs;
        let (a, b) = (a.current(), b.current());
        for i in 0..out.len() {
            out[i] = (self.op)(a[i], b[i]);
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Element-wise unary operator
// ---------------------------------------------------------------------------

/// Element-wise unary operator: `out[i] = op(a[i])`.
///
/// Shape-preserving: output shape equals input shape.
pub struct Elementwise1<T: Copy, Op: Fn(T) -> T> {
    op: Op,
    _phantom: PhantomData<T>,
}

impl<T: Copy + 'static, Op: Fn(T) -> T + 'static> Operator for Elementwise1<T, Op> {
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
        let (a,) = inputs;
        let a = a.current();
        for i in 0..out.len() {
            out[i] = (self.op)(a[i]);
        }
        true
    }
}

/// Type aliases.
pub type Add<T> = Elementwise2<T, fn(T, T) -> T>;
pub type Subtract<T> = Elementwise2<T, fn(T, T) -> T>;
pub type Multiply<T> = Elementwise2<T, fn(T, T) -> T>;
pub type Divide<T> = Elementwise2<T, fn(T, T) -> T>;
pub type Negate<T> = Elementwise1<T, fn(T) -> T>;

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Create an element-wise addition operator.
pub fn add<T: Copy + ops::Add<Output = T>>() -> Add<T> {
    Elementwise2 {
        op: |a, b| a + b,
        _phantom: PhantomData,
    }
}

/// Create an element-wise subtraction operator.
pub fn subtract<T: Copy + ops::Sub<Output = T>>() -> Subtract<T> {
    Elementwise2 {
        op: |a, b| a - b,
        _phantom: PhantomData,
    }
}

/// Create an element-wise multiplication operator.
pub fn multiply<T: Copy + ops::Mul<Output = T>>() -> Multiply<T> {
    Elementwise2 {
        op: |a, b| a * b,
        _phantom: PhantomData,
    }
}

/// Create an element-wise division operator.
pub fn divide<T: Copy + ops::Div<Output = T>>() -> Divide<T> {
    Elementwise2 {
        op: |a, b| a / b,
        _phantom: PhantomData,
    }
}

/// Create an element-wise negation operator.
pub fn negate<T: Copy + ops::Neg<Output = T>>() -> Negate<T> {
    Elementwise1 {
        op: |a| -a,
        _phantom: PhantomData,
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
        let mut op = Apply::new(
            |shapes: &[&[usize]]| shapes[0].into(),
            |inputs: &[&[f64]], out: &mut [f64]| {
                out[0] = (inputs[0][0] * inputs[0][0] + inputs[1][0] * inputs[1][0]).sqrt();
            },
        );
        let mut out = [0.0];
        assert!(op.compute(1, vec![&a, &b].into_boxed_slice(), &mut out));
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

    #[test]
    fn test_output_shape() {
        let op = add::<f64>();
        assert_eq!(&*op.output_shape(&[&[3], &[3]]), &[3]);
        assert_eq!(&*op.output_shape(&[&[2, 3], &[2, 3]]), &[2, 3]);

        let op = negate::<f64>();
        assert_eq!(&*op.output_shape(&[&[5]]), &[5]);
    }
}
