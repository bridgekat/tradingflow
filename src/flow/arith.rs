//! Element-wise numeric operators — direct port of [`crate::operators::num`]'s
//! `arithmetic` module onto the new [`Operator`](super::op::Operator) trait.
//! The `compute` bodies are unchanged; only `Input`→`Port` and the `produced`
//! type differ.

use std::marker::PhantomData;
use std::ops;

use num_traits::{Float, Signed};

use flowgraph::typed::Port;

use super::op::Operator;
use crate::{Array, Scalar};

// ===========================================================================
// Unary
// ===========================================================================

macro_rules! define_unary_op {
    ($(#[$meta:meta])* $Name:ident [$($bounds:tt)*], |$x:ident| $body:expr) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $Name<T: Scalar>(PhantomData<T>);

        impl<T: Scalar + $($bounds)*> $Name<T> {
            pub fn new() -> Self { Self(PhantomData) }
        }

        impl<T: Scalar + $($bounds)*> Default for $Name<T> {
            fn default() -> Self { Self::new() }
        }

        impl<T: Scalar + $($bounds)*> Operator for $Name<T> {
            type State = ();
            type Inputs = Port<Array<T>>;
            type Output = Array<T>;

            fn init(&self, inputs: &Array<T>) -> ((), Array<T>) {
                ((), Array::zeros(inputs.shape()))
            }

            #[inline(always)]
            fn compute(
                _state: &mut (),
                inputs: &Array<T>,
                output: &mut Array<T>,
                _produced: bool,
            ) -> bool {
                let a = inputs.as_slice();
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

define_unary_op!(/// Element-wise negation: `-a`.
    Negate [ops::Neg<Output = T>], |x| -x);
define_unary_op!(/// Element-wise natural logarithm.
    Log [Float], |x| x.ln());
define_unary_op!(/// Element-wise base-2 logarithm.
    Log2 [Float], |x| x.log2());
define_unary_op!(/// Element-wise base-10 logarithm.
    Log10 [Float], |x| x.log10());
define_unary_op!(/// Element-wise exponential.
    Exp [Float], |x| x.exp());
define_unary_op!(/// Element-wise base-2 exponential.
    Exp2 [Float], |x| x.exp2());
define_unary_op!(/// Element-wise square root.
    Sqrt [Float], |x| x.sqrt());
define_unary_op!(/// Element-wise ceiling.
    Ceil [Float], |x| x.ceil());
define_unary_op!(/// Element-wise floor.
    Floor [Float], |x| x.floor());
define_unary_op!(/// Element-wise rounding.
    Round [Float], |x| x.round());
define_unary_op!(/// Element-wise reciprocal: `1/x`.
    Recip [Float], |x| x.recip());
define_unary_op!(/// Element-wise absolute value.
    Abs [Signed], |x| x.abs());
define_unary_op!(/// Element-wise signum (−1, 0, or +1).
    Sign [Signed], |x| x.signum());

// ===========================================================================
// Binary
// ===========================================================================

macro_rules! define_binary_op {
    ($(#[$meta:meta])* $Name:ident [$($bounds:tt)*], |$a:ident, $b:ident| $body:expr) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $Name<T: Scalar>(PhantomData<T>);

        impl<T: Scalar + $($bounds)*> $Name<T> {
            pub fn new() -> Self { Self(PhantomData) }
        }

        impl<T: Scalar + $($bounds)*> Default for $Name<T> {
            fn default() -> Self { Self::new() }
        }

        impl<T: Scalar + $($bounds)*> Operator for $Name<T> {
            type State = ();
            type Inputs = (Port<Array<T>>, Port<Array<T>>);
            type Output = Array<T>;

            fn init(&self, inputs: (&Array<T>, &Array<T>)) -> ((), Array<T>) {
                ((), Array::zeros(inputs.0.shape()))
            }

            #[inline(always)]
            fn compute(
                _state: &mut (),
                inputs: (&Array<T>, &Array<T>),
                output: &mut Array<T>,
                _produced: (bool, bool),
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

define_binary_op!(/// Element-wise addition: `a + b`.
    Add [ops::Add<Output = T>], |a, b| a + b);
define_binary_op!(/// Element-wise subtraction: `a - b`.
    Subtract [ops::Sub<Output = T>], |a, b| a - b);
define_binary_op!(/// Element-wise multiplication: `a * b`.
    Multiply [ops::Mul<Output = T>], |a, b| a * b);
define_binary_op!(/// Element-wise division: `a / b`.
    Divide [ops::Div<Output = T>], |a, b| a / b);
define_binary_op!(/// Element-wise minimum (IEEE 754).
    Min [Float], |a, b| a.min(b));
define_binary_op!(/// Element-wise maximum (IEEE 754).
    Max [Float], |a, b| a.max(b));

// ===========================================================================
// Parameterized unary
// ===========================================================================

/// Element-wise power: `x.powf(n)`.
#[derive(Clone)]
pub struct Pow<T: Scalar> {
    n: T,
}

impl<T: Scalar + Float> Pow<T> {
    pub fn new(n: T) -> Self {
        Self { n }
    }
}

impl<T: Scalar + Float> Operator for Pow<T> {
    type State = T;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>) -> (T, Array<T>) {
        (self.n, Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut T,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _produced: bool,
    ) -> bool {
        let n = *state;
        let a = inputs.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = a[i].powf(n);
        }
        true
    }
}
