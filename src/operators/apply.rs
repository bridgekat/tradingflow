//! Built-in element-wise operators.
//!
//! * [`Apply2`] / [`Apply1`] — element-wise binary/unary ops.
//!   Factory functions: [`add`], [`subtract`], [`multiply`], [`divide`],
//!   [`negate`].

use std::marker::PhantomData;
use std::ops;

use crate::{Array, Operator, Scalar};

// ---------------------------------------------------------------------------
// Element-wise unary operator
// ---------------------------------------------------------------------------

/// Element-wise unary operator: `out[i] = op(a[i])`.
///
/// Shape-preserving: output shape equals input shape.
pub struct Apply1<T: Scalar, Op: Fn(T) -> T> {
    op: Op,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, Op: Fn(T) -> T + Send + 'static> Operator for Apply1<T, Op> {
    type State = Self;
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: i64) -> (Self, Array<T>) {
        (self, Array::default(inputs.0.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        let a = inputs.0.as_slice();
        let out = output.as_slice_mut();
        for i in 0..out.len() {
            out[i] = (state.op)(a[i].clone());
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
pub struct Apply2<T: Scalar, Op: Fn(T, T) -> T> {
    op: Op,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, Op: Fn(T, T) -> T + Send + 'static> Operator for Apply2<T, Op> {
    type State = Self;
    type Inputs = (Array<T>, Array<T>);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>, &Array<T>), _timestamp: i64) -> (Self, Array<T>) {
        (self, Array::default(inputs.0.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&Array<T>, &Array<T>),
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        let a = inputs.0.as_slice();
        let b = inputs.1.as_slice();
        let out = output.as_slice_mut();
        for i in 0..out.len() {
            out[i] = (state.op)(a[i].clone(), b[i].clone());
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

pub type Add<T> = Apply2<T, fn(T, T) -> T>;
pub type Subtract<T> = Apply2<T, fn(T, T) -> T>;
pub type Multiply<T> = Apply2<T, fn(T, T) -> T>;
pub type Divide<T> = Apply2<T, fn(T, T) -> T>;
pub type Negate<T> = Apply1<T, fn(T) -> T>;

/// Create an element-wise addition operator.
pub fn add<T: Scalar + ops::Add<Output = T>>() -> Add<T> {
    Apply2 {
        op: |a, b| a + b,
        _phantom: PhantomData,
    }
}

/// Create an element-wise subtraction operator.
pub fn subtract<T: Scalar + ops::Sub<Output = T>>() -> Subtract<T> {
    Apply2 {
        op: |a, b| a - b,
        _phantom: PhantomData,
    }
}

/// Create an element-wise multiplication operator.
pub fn multiply<T: Scalar + ops::Mul<Output = T>>() -> Multiply<T> {
    Apply2 {
        op: |a, b| a * b,
        _phantom: PhantomData,
    }
}

/// Create an element-wise division operator.
pub fn divide<T: Scalar + ops::Div<Output = T>>() -> Divide<T> {
    Apply2 {
        op: |a, b| a / b,
        _phantom: PhantomData,
    }
}

/// Create an element-wise negation operator.
pub fn negate<T: Scalar + ops::Neg<Output = T>>() -> Negate<T> {
    Apply1 {
        op: |a| -a,
        _phantom: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::operator::Operator;

    #[test]
    fn add_scalar() {
        let a = Array::scalar(10.0_f64);
        let b = Array::scalar(3.0);
        let (mut s, mut o) = add::<f64>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o.as_slice(), &[13.0]);
    }

    #[test]
    fn add_vector() {
        let a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        let (mut s, mut o) = add::<f64>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_i32() {
        let a = Array::from_vec(&[3], vec![1_i32, 2, 3]);
        let b = Array::from_vec(&[3], vec![10, 20, 30]);
        let (mut s, mut o) = add::<i32>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o.as_slice(), &[11, 22, 33]);
    }

    #[test]
    fn subtract_scalar() {
        let a = Array::scalar(20.0_f64);
        let b = Array::scalar(7.0);
        let (mut s, mut o) = subtract::<f64>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o.as_slice(), &[13.0]);
    }

    #[test]
    fn multiply_scalar() {
        let a = Array::scalar(4.0_f64);
        let b = Array::scalar(5.0);
        let (mut s, mut o) = multiply::<f64>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o.as_slice(), &[20.0]);
    }

    #[test]
    fn divide_scalar() {
        let a = Array::scalar(20.0_f64);
        let b = Array::scalar(4.0);
        let (mut s, mut o) = divide::<f64>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o.as_slice(), &[5.0]);
    }

    #[test]
    fn negate_vector() {
        let a = Array::from_vec(&[3], vec![1.0_f64, -2.0, 3.0]);
        let (mut s, mut o) = negate::<f64>().init((&a,), i64::MIN);
        Apply1::compute(&mut s, (&a,), &mut o, 1);
        assert_eq!(o.as_slice(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn multi_step() {
        let mut a = Array::scalar(0.0_f64);
        let mut b = Array::scalar(0.0);
        let (mut s, mut o) = add::<f64>().init((&a, &b), i64::MIN);
        a[0] = 10.0;
        b[0] = 3.0;
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o[0], 13.0);
        a[0] = 20.0;
        b[0] = 7.0;
        Apply2::compute(&mut s, (&a, &b), &mut o, 2);
        assert_eq!(o[0], 27.0);
    }

    #[test]
    fn preserves_shape() {
        let a = Array::from_vec(&[2, 3], vec![0.0_f64; 6]);
        let (_, o) = add::<f64>().init((&a, &a), i64::MIN);
        assert_eq!(o.shape(), &[2, 3]);
        let (_, o) = negate::<f64>().init((&a,), i64::MIN);
        assert_eq!(o.shape(), &[2, 3]);
    }
}
