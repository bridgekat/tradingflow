//! Built-in element-wise operators.
//!
//! * [`Elementwise2`] / [`Elementwise1`] — element-wise binary/unary ops.
//!   Factory functions: [`add`], [`subtract`], [`multiply`], [`divide`],
//!   [`negate`].

use std::marker::PhantomData;
use std::ops;

use crate::operator::Operator;
use crate::store::{ElementViewMut, Store};
use crate::types::Scalar;

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
    type Inputs = (Store<T>,);
    type Scalar = T;

    fn window_sizes(&self, _: &[&[usize]]) -> (usize,) {
        (1,)
    }

    fn default(&self, input_shapes: &[&[usize]]) -> (Box<[usize]>, Box<[T]>) {
        let shape: Box<[usize]> = input_shapes[0].into();
        let stride = shape.iter().product::<usize>();
        (shape, vec![T::default(); stride].into())
    }

    fn init(self) -> Self {
        self
    }

    #[inline(always)]
    fn compute(state: &mut Self, inputs: (&Store<T>,), output: ElementViewMut<'_, T>) -> bool {
        let a = inputs.0.current();
        let out = output.values;
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
    type Inputs = (Store<T>, Store<T>);
    type Scalar = T;

    fn window_sizes(&self, _: &[&[usize]]) -> (usize, usize) {
        (1, 1)
    }

    fn default(&self, input_shapes: &[&[usize]]) -> (Box<[usize]>, Box<[T]>) {
        let shape: Box<[usize]> = input_shapes[0].into();
        let stride = shape.iter().product::<usize>();
        (shape, vec![T::default(); stride].into())
    }

    fn init(self) -> Self {
        self
    }

    #[inline(always)]
    fn compute(
        state: &mut Self,
        inputs: (&Store<T>, &Store<T>),
        output: ElementViewMut<'_, T>,
    ) -> bool {
        let (a, b) = (inputs.0.current(), inputs.1.current());
        let out = output.values;
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    #[test]
    fn test_add() {
        let a = Store::element(&[], &[20.0]);
        let b = Store::element(&[], &[7.0]);
        let mut state = add::<f64>().init();
        let mut out = Store::element(&[], &[0.0]);
        out.push_default(1);
        Apply2::compute(&mut state, (&a, &b), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[27.0]);
    }

    #[test]
    fn test_subtract() {
        let a = Store::element(&[], &[20.0]);
        let b = Store::element(&[], &[7.0]);
        let mut state = subtract::<f64>().init();
        let mut out = Store::element(&[], &[0.0]);
        out.push_default(1);
        Apply2::compute(&mut state, (&a, &b), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[13.0]);
    }

    #[test]
    fn test_multiply() {
        let a = Store::element(&[], &[4.0]);
        let b = Store::element(&[], &[5.0]);
        let mut state = multiply::<f64>().init();
        let mut out = Store::element(&[], &[0.0]);
        out.push_default(1);
        Apply2::compute(&mut state, (&a, &b), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[20.0]);
    }

    #[test]
    fn test_divide() {
        let a = Store::element(&[], &[20.0]);
        let b = Store::element(&[], &[4.0]);
        let mut state = divide::<f64>().init();
        let mut out = Store::element(&[], &[0.0]);
        out.push_default(1);
        Apply2::compute(&mut state, (&a, &b), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[5.0]);
    }

    #[test]
    fn test_negate() {
        let a = Store::element(&[], &[7.0]);
        let mut state = negate::<f64>().init();
        let mut out = Store::element(&[], &[0.0]);
        out.push_default(1);
        Apply1::compute(&mut state, (&a,), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[-7.0]);
    }

    #[test]
    fn test_strided_add() {
        let a = Store::element(&[2], &[1.0, 2.0]);
        let b = Store::element(&[2], &[10.0, 20.0]);
        let mut state = add::<f64>().init();
        let mut out = Store::element(&[2], &[0.0, 0.0]);
        out.push_default(1);
        Apply2::compute(&mut state, (&a, &b), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[11.0, 22.0]);
    }

    #[test]
    fn test_always_produces_output() {
        let a = Store::<f64>::element(&[], &[0.0]);
        let b = Store::<f64>::element(&[], &[0.0]);
        let mut state = add::<f64>().init();
        let mut out = Store::element(&[], &[0.0]);
        out.push_default(1);
        Apply2::compute(&mut state, (&a, &b), out.current_view_mut());
        out.commit();
        assert_eq!(out.current(), &[0.0]);
    }

    #[test]
    fn test_output_shape() {
        let op = add::<f64>();
        assert_eq!(&*op.default(&[&[3], &[3]]).0, &[3]);
        assert_eq!(&*op.default(&[&[2, 3], &[2, 3]]).0, &[2, 3]);

        let op = negate::<f64>();
        assert_eq!(&*op.default(&[&[5]]).0, &[5]);
    }
}
