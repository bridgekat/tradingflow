//! Built-in element-wise operators.
//!
//! * [`Apply2`] / [`Apply1`] — element-wise binary/unary ops.
//!   Factory functions: [`add`], [`subtract`], [`multiply`], [`divide`],
//!   [`negate`].

use std::marker::PhantomData;
use std::ops;

use crate::array::Array;
use crate::operator::Operator;
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
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: i64) -> (Self, Array<T>) {
        (self, Array::zeros(inputs.0.shape()))
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
        (self, Array::zeros(inputs.0.shape()))
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
