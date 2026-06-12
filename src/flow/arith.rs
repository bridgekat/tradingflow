//! Element-wise numeric operators, implemented directly on
//! [`flowgraph::typed::Operator`]. The loop bodies are unchanged from the
//! legacy port; the output buffer lives in the operator state and is sized on
//! the `init` build call.

use std::marker::PhantomData;
use std::ops;

use num_traits::{Float, Signed};

use flowgraph::typed::{Operator, RefPort};

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
            type Inputs = RefPort<Array<T>>;
            type Outputs = RefPort<Array<T>>;
            type State = Array<T>;

            fn init(self) -> Array<T> {
                Array::zeros(&[0])
            }

            #[inline(always)]
            fn compute<'a, 'b: 'a>(
                (_, a): (bool, &'a Array<T>),
                out: &'b mut Array<T>,
                init: bool,
            ) -> (bool, &'a Array<T>) {
                if init {
                    *out = Array::zeros(a.shape());
                    return (false, &*out);
                }
                let src = a.as_slice();
                let dst = out.as_mut_slice();
                for i in 0..dst.len() {
                    let $x = src[i].clone();
                    dst[i] = $body;
                }
                (true, &*out)
            }

            #[inline(always)]
            fn passthrough<'a, 'b: 'a>(
                _: (bool, &'a Array<T>),
                out: &'b Array<T>,
            ) -> (bool, &'a Array<T>) {
                (false, out)
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
            type Inputs = (RefPort<Array<T>>, RefPort<Array<T>>);
            type Outputs = RefPort<Array<T>>;
            type State = Array<T>;

            fn init(self) -> Array<T> {
                Array::zeros(&[0])
            }

            #[inline(always)]
            fn compute<'a, 'b: 'a>(
                ((_, a), (_, b)): ((bool, &'a Array<T>), (bool, &'a Array<T>)),
                out: &'b mut Array<T>,
                init: bool,
            ) -> (bool, &'a Array<T>) {
                if init {
                    *out = Array::zeros(a.shape());
                    return (false, &*out);
                }
                let a_sl = a.as_slice();
                let b_sl = b.as_slice();
                let dst = out.as_mut_slice();
                for i in 0..dst.len() {
                    let $a = a_sl[i].clone();
                    let $b = b_sl[i].clone();
                    dst[i] = $body;
                }
                (true, &*out)
            }

            #[inline(always)]
            fn passthrough<'a, 'b: 'a>(
                _: ((bool, &'a Array<T>), (bool, &'a Array<T>)),
                out: &'b Array<T>,
            ) -> (bool, &'a Array<T>) {
                (false, out)
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

/// Runtime state for [`Pow`]: the exponent plus the output buffer.
pub struct PowState<T: Scalar> {
    n: T,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Pow<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = PowState<T>;

    fn init(self) -> PowState<T> {
        PowState {
            n: self.n,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut PowState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            state.out = Array::zeros(a.shape());
            return (false, &state.out);
        }
        let n = state.n;
        let src = a.as_slice();
        let dst = state.out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = src[i].powf(n);
        }
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b PowState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}
