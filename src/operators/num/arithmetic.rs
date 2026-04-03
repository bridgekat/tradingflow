//! Non-parameterized element-wise arithmetic operators.

use std::marker::PhantomData;
use std::ops;

use num_traits::{Float, Signed};

use crate::operator::Notify;
use crate::{Array, Operator, Scalar};

// ===========================================================================
// Unary
// ===========================================================================

macro_rules! define_unary_op {
    ($(#[$meta:meta])* $Name:ident [$($bounds:tt)*], |$x:ident| $body:expr) => {
        $(#[$meta])*
        pub struct $Name<T: Scalar>(PhantomData<T>);

        impl<T: Scalar + $($bounds)*> $Name<T> {
            /// Create a new instance.
            pub fn new() -> Self { Self(PhantomData) }
        }

        impl<T: Scalar + $($bounds)*> Default for $Name<T> {
            fn default() -> Self { Self::new() }
        }

        impl<T: Scalar + $($bounds)*> Operator for $Name<T> {
            type State = ();
            type Inputs = (Array<T>,);
            type Output = Array<T>;

            fn init(self, inputs: (&Array<T>,), _timestamp: i64) -> ((), Array<T>) {
                ((), Array::zeros(inputs.0.shape()))
            }

            #[inline(always)]
            fn compute(
                _state: &mut (),
                inputs: (&Array<T>,),
                output: &mut Array<T>,
                _timestamp: i64,
                _notify: &Notify<'_>,
            ) -> bool {
                let a = inputs.0.as_slice();
                let out = output.as_mut_slice();
                for i in 0..out.len() {
                    let $x = a[i].clone();
                    out[i] = $body;
                }
                true
            }
        }
    };
}

define_unary_op!(
    /// Element-wise negation: `-a`.
    Negate [ops::Neg<Output = T>], |x| -x
);

define_unary_op!(
    /// Element-wise natural logarithm.
    Log [Float], |x| x.ln()
);

define_unary_op!(
    /// Element-wise base-2 logarithm.
    Log2 [Float], |x| x.log2()
);

define_unary_op!(
    /// Element-wise base-10 logarithm.
    Log10 [Float], |x| x.log10()
);

define_unary_op!(
    /// Element-wise exponential.
    Exp [Float], |x| x.exp()
);

define_unary_op!(
    /// Element-wise base-2 exponential.
    Exp2 [Float], |x| x.exp2()
);

define_unary_op!(
    /// Element-wise square root.
    Sqrt [Float], |x| x.sqrt()
);

define_unary_op!(
    /// Element-wise ceiling.
    Ceil [Float], |x| x.ceil()
);

define_unary_op!(
    /// Element-wise floor.
    Floor [Float], |x| x.floor()
);

define_unary_op!(
    /// Element-wise rounding.
    Round [Float], |x| x.round()
);

define_unary_op!(
    /// Element-wise reciprocal: `1/x`.
    Recip [Float], |x| x.recip()
);

define_unary_op!(
    /// Element-wise absolute value.
    Abs [Signed], |x| x.abs()
);

define_unary_op!(
    /// Element-wise signum (−1, 0, or +1).
    Sign [Signed], |x| x.signum()
);

// ===========================================================================
// Binary
// ===========================================================================

macro_rules! define_binary_op {
    ($(#[$meta:meta])* $Name:ident [$($bounds:tt)*], |$a:ident, $b:ident| $body:expr) => {
        $(#[$meta])*
        pub struct $Name<T: Scalar>(PhantomData<T>);

        impl<T: Scalar + $($bounds)*> $Name<T> {
            /// Create a new instance.
            pub fn new() -> Self { Self(PhantomData) }
        }

        impl<T: Scalar + $($bounds)*> Default for $Name<T> {
            fn default() -> Self { Self::new() }
        }

        impl<T: Scalar + $($bounds)*> Operator for $Name<T> {
            type State = ();
            type Inputs = (Array<T>, Array<T>);
            type Output = Array<T>;

            fn init(self, inputs: (&Array<T>, &Array<T>), _timestamp: i64) -> ((), Array<T>) {
                ((), Array::zeros(inputs.0.shape()))
            }

            #[inline(always)]
            fn compute(
                _state: &mut (),
                inputs: (&Array<T>, &Array<T>),
                output: &mut Array<T>,
                _timestamp: i64,
                _notify: &Notify<'_>,
            ) -> bool {
                let a_sl = inputs.0.as_slice();
                let b_sl = inputs.1.as_slice();
                let out = output.as_mut_slice();
                for i in 0..out.len() {
                    let $a = a_sl[i].clone();
                    let $b = b_sl[i].clone();
                    out[i] = $body;
                }
                true
            }
        }
    };
}

define_binary_op!(
    /// Element-wise addition: `a + b`.
    Add [ops::Add<Output = T>], |a, b| a + b
);

define_binary_op!(
    /// Element-wise subtraction: `a - b`.
    Subtract [ops::Sub<Output = T>], |a, b| a - b
);

define_binary_op!(
    /// Element-wise multiplication: `a * b`.
    Multiply [ops::Mul<Output = T>], |a, b| a * b
);

define_binary_op!(
    /// Element-wise division: `a / b`.
    Divide [ops::Div<Output = T>], |a, b| a / b
);

define_binary_op!(
    /// Element-wise minimum (IEEE 754: returns non-NaN if one operand is NaN).
    Min [Float], |a, b| a.min(b)
);

define_binary_op!(
    /// Element-wise maximum (IEEE 754: returns non-NaN if one operand is NaN).
    Max [Float], |a, b| a.max(b)
);

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::operator::Operator;

    fn unary_values<O: Operator<Inputs = (Array<f64>,), Output = Array<f64>, State = ()>>(
        op: O,
        input: &[f64],
    ) -> Vec<f64> {
        let a = Array::from_vec(&[input.len()], input.to_vec());
        let (mut s, mut o) = op.init((&a,), i64::MIN);
        O::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], &[]));
        o.as_slice().to_vec()
    }

    // -- Unary ---------------------------------------------------------------

    #[test]
    fn negate_vector() {
        let a = Array::from_vec(&[3], vec![1.0_f64, -2.0, 3.0]);
        let (mut s, mut o) = Negate::<f64>::new().init((&a,), i64::MIN);
        Negate::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_log() {
        let out = unary_values(Log::new(), &[1.0, std::f64::consts::E, 10.0]);
        assert!((out[0] - 0.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_exp() {
        let out = unary_values(Exp::new(), &[0.0, 1.0]);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_sqrt() {
        let out = unary_values(Sqrt::new(), &[4.0, 9.0, 16.0]);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_abs_and_sign() {
        let out = unary_values(Abs::new(), &[-3.0, 0.0, 5.0]);
        assert_eq!(out, vec![3.0, 0.0, 5.0]);

        let out = unary_values(Sign::new(), &[-3.0, 0.0, 5.0]);
        assert_eq!(out, vec![-1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_recip() {
        let out = unary_values(Recip::new(), &[2.0, 4.0, 0.5]);
        assert_eq!(out, vec![0.5, 0.25, 2.0]);
    }

    #[test]
    fn test_ceil_floor_round() {
        let out = unary_values(Ceil::new(), &[1.1, 1.9, -1.1]);
        assert_eq!(out, vec![2.0, 2.0, -1.0]);
        let out = unary_values(Floor::new(), &[1.1, 1.9, -1.1]);
        assert_eq!(out, vec![1.0, 1.0, -2.0]);
        let out = unary_values(Round::new(), &[1.4, 1.5, -1.5]);
        assert_eq!(out, vec![1.0, 2.0, -2.0]);
    }

    #[test]
    fn test_log_nan_and_edge() {
        let out = unary_values(Log::new(), &[f64::NAN, 0.0, -1.0]);
        assert!(out[0].is_nan());
        assert!(out[1].is_infinite() && out[1] < 0.0);
        assert!(out[2].is_nan());
    }

    #[test]
    fn test_sqrt_negative() {
        let out = unary_values(Sqrt::new(), &[-1.0]);
        assert!(out[0].is_nan());
    }

    #[test]
    fn test_recip_zero() {
        let out = unary_values(Recip::new(), &[0.0]);
        assert!(out[0].is_infinite());
    }

    #[test]
    fn preserves_shape() {
        let a = Array::from_vec(&[2, 3], vec![0.0_f64; 6]);
        let (_, o) = Negate::<f64>::new().init((&a,), i64::MIN);
        assert_eq!(o.shape(), &[2, 3]);
        let (_, o) = Add::<f64>::new().init((&a, &a), i64::MIN);
        assert_eq!(o.shape(), &[2, 3]);
    }

    // -- Binary --------------------------------------------------------------

    #[test]
    fn add_scalar() {
        let a = Array::scalar(10.0_f64);
        let b = Array::scalar(3.0);
        let (mut s, mut o) = Add::<f64>::new().init((&a, &b), i64::MIN);
        Add::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[13.0]);
    }

    #[test]
    fn add_vector() {
        let a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        let (mut s, mut o) = Add::<f64>::new().init((&a, &b), i64::MIN);
        Add::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_i32() {
        let a = Array::from_vec(&[3], vec![1_i32, 2, 3]);
        let b = Array::from_vec(&[3], vec![10, 20, 30]);
        let (mut s, mut o) = Add::<i32>::new().init((&a, &b), i64::MIN);
        Add::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[11, 22, 33]);
    }

    #[test]
    fn subtract_scalar() {
        let a = Array::scalar(20.0_f64);
        let b = Array::scalar(7.0);
        let (mut s, mut o) = Subtract::<f64>::new().init((&a, &b), i64::MIN);
        Subtract::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[13.0]);
    }

    #[test]
    fn multiply_scalar() {
        let a = Array::scalar(4.0_f64);
        let b = Array::scalar(5.0);
        let (mut s, mut o) = Multiply::<f64>::new().init((&a, &b), i64::MIN);
        Multiply::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[20.0]);
    }

    #[test]
    fn divide_scalar() {
        let a = Array::scalar(20.0_f64);
        let b = Array::scalar(4.0);
        let (mut s, mut o) = Divide::<f64>::new().init((&a, &b), i64::MIN);
        Divide::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[5.0]);
    }

    #[test]
    fn multi_step() {
        let mut a = Array::scalar(0.0_f64);
        let mut b = Array::scalar(0.0);
        let (mut s, mut o) = Add::<f64>::new().init((&a, &b), i64::MIN);
        a[0] = 10.0;
        b[0] = 3.0;
        Add::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o[0], 13.0);
        a[0] = 20.0;
        b[0] = 7.0;
        Add::compute(&mut s, (&a, &b), &mut o, 2, &Notify::new(&[], &[]));
        assert_eq!(o[0], 27.0);
    }

    #[test]
    fn test_min_max() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 5.0, 3.0]);
        let b = Array::from_vec(&[3], vec![2.0, 4.0, 6.0]);

        let (mut s, mut o) = Min::<f64>::new().init((&a, &b), i64::MIN);
        Min::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[1.0, 4.0, 3.0]);

        let (mut s, mut o) = Max::<f64>::new().init((&a, &b), i64::MIN);
        Max::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_min_nan() {
        let a = Array::from_vec(&[2], vec![f64::NAN, 1.0]);
        let b = Array::from_vec(&[2], vec![1.0, f64::NAN]);
        let (mut s, mut o) = Min::<f64>::new().init((&a, &b), i64::MIN);
        Min::compute(&mut s, (&a, &b), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o[0], 1.0);
        assert_eq!(o[1], 1.0);
    }
}
