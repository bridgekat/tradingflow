//! Built-in element-wise operators.
//!
//! * [`Apply1`] / [`Apply2`] — generic unary/binary element-wise operators.
//!
//! # Arithmetic
//!
//! [`add`], [`subtract`], [`multiply`], [`divide`], [`negate`].
//!
//! # Math (float-only)
//!
//! [`log`], [`log2`], [`log10`], [`exp`], [`exp2`], [`sqrt`],
//! [`ceil`], [`floor`], [`round`], [`recip`].
//!
//! # Math (signed)
//!
//! [`abs`], [`sign`].
//!
//! # Parameterized
//!
//! [`pow`], [`scale`], [`shift`], [`clamp`], [`nan_to_num`].
//!
//! # Binary math
//!
//! [`min`], [`max`].

use std::marker::PhantomData;
use std::ops;

use num_traits::{Float, Signed};

use crate::{Array, Operator, Scalar};

// ===========================================================================
// Element-wise unary operator
// ===========================================================================

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

// ===========================================================================
// Element-wise binary operator
// ===========================================================================

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

// ===========================================================================
// Arithmetic factory functions
// ===========================================================================

pub type Add<T> = Apply2<T, fn(T, T) -> T>;
pub type Subtract<T> = Apply2<T, fn(T, T) -> T>;
pub type Multiply<T> = Apply2<T, fn(T, T) -> T>;
pub type Divide<T> = Apply2<T, fn(T, T) -> T>;
pub type Negate<T> = Apply1<T, fn(T) -> T>;

/// Element-wise addition.
pub fn add<T: Scalar + ops::Add<Output = T>>() -> Add<T> {
    Apply2 {
        op: |a, b| a + b,
        _phantom: PhantomData,
    }
}

/// Element-wise subtraction.
pub fn subtract<T: Scalar + ops::Sub<Output = T>>() -> Subtract<T> {
    Apply2 {
        op: |a, b| a - b,
        _phantom: PhantomData,
    }
}

/// Element-wise multiplication.
pub fn multiply<T: Scalar + ops::Mul<Output = T>>() -> Multiply<T> {
    Apply2 {
        op: |a, b| a * b,
        _phantom: PhantomData,
    }
}

/// Element-wise division.
pub fn divide<T: Scalar + ops::Div<Output = T>>() -> Divide<T> {
    Apply2 {
        op: |a, b| a / b,
        _phantom: PhantomData,
    }
}

/// Element-wise negation.
pub fn negate<T: Scalar + ops::Neg<Output = T>>() -> Negate<T> {
    Apply1 {
        op: |a| -a,
        _phantom: PhantomData,
    }
}

// ===========================================================================
// Float unary factory functions
// ===========================================================================

pub type Log<T> = Apply1<T, fn(T) -> T>;
pub type Log2<T> = Apply1<T, fn(T) -> T>;
pub type Log10<T> = Apply1<T, fn(T) -> T>;
pub type Exp<T> = Apply1<T, fn(T) -> T>;
pub type Exp2<T> = Apply1<T, fn(T) -> T>;
pub type Sqrt<T> = Apply1<T, fn(T) -> T>;
pub type Ceil<T> = Apply1<T, fn(T) -> T>;
pub type Floor<T> = Apply1<T, fn(T) -> T>;
pub type Round<T> = Apply1<T, fn(T) -> T>;
pub type Recip<T> = Apply1<T, fn(T) -> T>;

/// Element-wise natural logarithm.
pub fn log<T: Scalar + Float>() -> Log<T> {
    Apply1 { op: |x| x.ln(), _phantom: PhantomData }
}

/// Element-wise base-2 logarithm.
pub fn log2<T: Scalar + Float>() -> Log2<T> {
    Apply1 { op: |x| x.log2(), _phantom: PhantomData }
}

/// Element-wise base-10 logarithm.
pub fn log10<T: Scalar + Float>() -> Log10<T> {
    Apply1 { op: |x| x.log10(), _phantom: PhantomData }
}

/// Element-wise exponential.
pub fn exp<T: Scalar + Float>() -> Exp<T> {
    Apply1 { op: |x| x.exp(), _phantom: PhantomData }
}

/// Element-wise base-2 exponential.
pub fn exp2<T: Scalar + Float>() -> Exp2<T> {
    Apply1 { op: |x| x.exp2(), _phantom: PhantomData }
}

/// Element-wise square root.
pub fn sqrt<T: Scalar + Float>() -> Sqrt<T> {
    Apply1 { op: |x| x.sqrt(), _phantom: PhantomData }
}

/// Element-wise ceiling.
pub fn ceil<T: Scalar + Float>() -> Ceil<T> {
    Apply1 { op: |x| x.ceil(), _phantom: PhantomData }
}

/// Element-wise floor.
pub fn floor<T: Scalar + Float>() -> Floor<T> {
    Apply1 { op: |x| x.floor(), _phantom: PhantomData }
}

/// Element-wise rounding.
pub fn round<T: Scalar + Float>() -> Round<T> {
    Apply1 { op: |x| x.round(), _phantom: PhantomData }
}

/// Element-wise reciprocal (`1/x`).
pub fn recip<T: Scalar + Float>() -> Recip<T> {
    Apply1 { op: |x| x.recip(), _phantom: PhantomData }
}

// ===========================================================================
// Signed unary factory functions
// ===========================================================================

pub type Abs<T> = Apply1<T, fn(T) -> T>;
pub type Sign<T> = Apply1<T, fn(T) -> T>;

/// Element-wise absolute value.
pub fn abs<T: Scalar + Signed>() -> Abs<T> {
    Apply1 { op: |x| x.abs(), _phantom: PhantomData }
}

/// Element-wise signum (−1, 0, or +1).
pub fn sign<T: Scalar + Signed>() -> Sign<T> {
    Apply1 { op: |x| x.signum(), _phantom: PhantomData }
}

// ===========================================================================
// Parameterized unary factory functions
// ===========================================================================

/// Element-wise power: `x.powf(n)`.
pub fn pow<T: Scalar + Float>(n: T) -> Apply1<T, impl Fn(T) -> T + Send + 'static> {
    Apply1 { op: move |x| x.powf(n), _phantom: PhantomData }
}

/// Element-wise scale: `x * c`.
pub fn scale<T: Scalar + ops::Mul<Output = T>>(c: T) -> Apply1<T, impl Fn(T) -> T + Send + 'static> {
    Apply1 { op: move |x| x * c.clone(), _phantom: PhantomData }
}

/// Element-wise shift: `x + c`.
pub fn shift<T: Scalar + ops::Add<Output = T>>(c: T) -> Apply1<T, impl Fn(T) -> T + Send + 'static> {
    Apply1 { op: move |x| x + c.clone(), _phantom: PhantomData }
}

/// Element-wise clamp to `[lo, hi]`.
pub fn clamp<T: Scalar + Float>(lo: T, hi: T) -> Apply1<T, impl Fn(T) -> T + Send + 'static> {
    Apply1 {
        op: move |x| lo.max(hi.min(x)),
        _phantom: PhantomData,
    }
}

/// Replace NaN with `val`.
pub fn nan_to_num<T: Scalar + Float>(val: T) -> Apply1<T, impl Fn(T) -> T + Send + 'static> {
    Apply1 {
        op: move |x| if x.is_nan() { val } else { x },
        _phantom: PhantomData,
    }
}

// ===========================================================================
// Float binary factory functions
// ===========================================================================

pub type Min<T> = Apply2<T, fn(T, T) -> T>;
pub type Max<T> = Apply2<T, fn(T, T) -> T>;

/// Element-wise minimum (IEEE 754: returns non-NaN if one operand is NaN).
pub fn min<T: Scalar + Float>() -> Min<T> {
    Apply2 { op: |a, b| a.min(b), _phantom: PhantomData }
}

/// Element-wise maximum (IEEE 754: returns non-NaN if one operand is NaN).
pub fn max<T: Scalar + Float>() -> Max<T> {
    Apply2 { op: |a, b| a.max(b), _phantom: PhantomData }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::operator::Operator;

    // -- Existing arithmetic tests -------------------------------------------

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

    // -- Float unary tests ---------------------------------------------------

    fn apply1_values<Op: Fn(f64) -> f64 + Send + 'static>(
        op: Apply1<f64, Op>,
        input: &[f64],
    ) -> Vec<f64> {
        let a = Array::from_vec(&[input.len()], input.to_vec());
        let (mut s, mut o) = op.init((&a,), i64::MIN);
        Apply1::compute(&mut s, (&a,), &mut o, 1);
        o.into_vec()
    }

    #[test]
    fn test_log() {
        let out = apply1_values(log(), &[1.0, std::f64::consts::E, 10.0]);
        assert!((out[0] - 0.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_exp() {
        let out = apply1_values(exp(), &[0.0, 1.0]);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_sqrt() {
        let out = apply1_values(sqrt(), &[4.0, 9.0, 16.0]);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_abs_and_sign() {
        let out = apply1_values(abs(), &[-3.0, 0.0, 5.0]);
        assert_eq!(out, vec![3.0, 0.0, 5.0]);

        let out = apply1_values(sign(), &[-3.0, 0.0, 5.0]);
        assert_eq!(out, vec![-1.0, 1.0, 1.0]); // f64::signum(+0.0) = 1.0 (IEEE 754)
    }

    #[test]
    fn test_recip() {
        let out = apply1_values(recip(), &[2.0, 4.0, 0.5]);
        assert_eq!(out, vec![0.5, 0.25, 2.0]);
    }

    #[test]
    fn test_ceil_floor_round() {
        let out = apply1_values(ceil(), &[1.1, 1.9, -1.1]);
        assert_eq!(out, vec![2.0, 2.0, -1.0]);

        let out = apply1_values(floor(), &[1.1, 1.9, -1.1]);
        assert_eq!(out, vec![1.0, 1.0, -2.0]);

        let out = apply1_values(round(), &[1.4, 1.5, -1.5]);
        assert_eq!(out, vec![1.0, 2.0, -2.0]);
    }

    // -- Parameterized unary tests -------------------------------------------

    #[test]
    fn test_pow() {
        let out = apply1_values(pow(2.0), &[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_scale_and_shift() {
        let out = apply1_values(scale(3.0), &[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![3.0, 6.0, 9.0]);

        let out = apply1_values(shift(10.0), &[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_clamp() {
        let out = apply1_values(clamp(2.0, 5.0), &[1.0, 3.0, 7.0]);
        assert_eq!(out, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_nan_to_num() {
        let out = apply1_values(nan_to_num(0.0), &[1.0, f64::NAN, 3.0]);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 3.0);
    }

    // -- NaN propagation tests -----------------------------------------------

    #[test]
    fn test_log_nan_and_edge() {
        let out = apply1_values(log(), &[f64::NAN, 0.0, -1.0]);
        assert!(out[0].is_nan());
        assert!(out[1].is_infinite() && out[1] < 0.0); // log(0) = -inf
        assert!(out[2].is_nan()); // log(-1) = NaN
    }

    #[test]
    fn test_sqrt_negative() {
        let out = apply1_values(sqrt(), &[-1.0]);
        assert!(out[0].is_nan());
    }

    #[test]
    fn test_recip_zero() {
        let out = apply1_values(recip(), &[0.0]);
        assert!(out[0].is_infinite());
    }

    // -- Binary float tests --------------------------------------------------

    #[test]
    fn test_min_max() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 5.0, 3.0]);
        let b = Array::from_vec(&[3], vec![2.0, 4.0, 6.0]);

        let (mut s, mut o) = min::<f64>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o.as_slice(), &[1.0, 4.0, 3.0]);

        let (mut s, mut o) = max::<f64>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        assert_eq!(o.as_slice(), &[2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_min_nan() {
        let a = Array::from_vec(&[2], vec![f64::NAN, 1.0]);
        let b = Array::from_vec(&[2], vec![1.0, f64::NAN]);
        let (mut s, mut o) = min::<f64>().init((&a, &b), i64::MIN);
        Apply2::compute(&mut s, (&a, &b), &mut o, 1);
        // Float::min returns the non-NaN value
        assert_eq!(o[0], 1.0);
        assert_eq!(o[1], 1.0);
    }
}
