//! Element-wise numeric operators.
//!
//! The whole family collapses to two operators parameterized by a callable —
//! [`Unary<T, N, F>`] (`Fn(T) -> T`) and [`Binary<T, N, F>`] (`Fn(T, T) -> T`) —
//! over the strided [`ArrayView`] currency, sharing the one contiguous-fast /
//! strided-slow elementwise core ([`apply_unary`](crate::data::array::apply_unary)
//! / `apply_binary`). The named constructors below ([`add`], [`log`], …) are
//! thin wrappers that pin the callable, so `add()` / `log()` read like the old
//! `Add` / `Log` nodes while inferring the rank `N` from the wiring.

use std::marker::PhantomData;
use std::ops;

use num_traits::{Float, Signed};

use flowgraph::typed::{Operator, ViewPort};

use crate::data::array::{apply_binary, apply_unary};
use crate::operators::op::ArrayValue;
use crate::{Array, ArrayView, Scalar};

// ===========================================================================
// Unary — one elementwise operator, parameterized by a callable.
// ===========================================================================

/// Element-wise `out[i] = f(x[i])` over a rank-`N` [`ArrayView`].
pub struct Unary<T: Scalar, const N: usize, F> {
    f: F,
    _p: PhantomData<T>,
}

impl<T: Scalar, const N: usize, F: Fn(T) -> T + Send + Sync + 'static> Unary<T, N, F> {
    pub fn new(f: F) -> Self {
        Self { f, _p: PhantomData }
    }
}

/// Runtime state for [`Unary`]: the callable plus the output buffer.
pub struct UnaryState<T: Scalar, const N: usize, F> {
    f: F,
    out: Array<T, N>,
}

impl<T: Scalar, const N: usize, F: Fn(T) -> T + Send + Sync + 'static> Operator for Unary<T, N, F> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = UnaryState<T, N, F>;

    fn init(self) -> Self::State {
        UnaryState {
            f: self.f,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        // The build call seeds the output with the transformed build value (not
        // zeros — a fabricated finite observation would leak through carry
        // readers) but does not notify.
        if init {
            state.out = Array::zeros(x.extents());
        }
        apply_unary(&mut state.out, &x, &state.f);
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ===========================================================================
// Binary — the two-input elementwise operator.
// ===========================================================================

/// Element-wise `out[i] = f(a[i], b[i])` over rank-`N` [`ArrayView`]s sharing
/// extents.
pub struct Binary<T: Scalar, const N: usize, F> {
    f: F,
    _p: PhantomData<T>,
}

impl<T: Scalar, const N: usize, F: Fn(T, T) -> T + Send + Sync + 'static> Binary<T, N, F> {
    pub fn new(f: F) -> Self {
        Self { f, _p: PhantomData }
    }
}

/// Runtime state for [`Binary`]: the callable plus the output buffer.
pub struct BinaryState<T: Scalar, const N: usize, F> {
    f: F,
    out: Array<T, N>,
}

impl<T: Scalar, const N: usize, F: Fn(T, T) -> T + Send + Sync + 'static> Operator
    for Binary<T, N, F>
{
    type Inputs = (ViewPort<ArrayValue<T, N>>, ViewPort<ArrayValue<T, N>>);
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = BinaryState<T, N, F>;

    fn init(self) -> Self::State {
        BinaryState {
            f: self.f,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.out = Array::zeros(a.extents());
        }
        apply_binary(&mut state.out, &a, &b, &state.f);
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ===========================================================================
// Named constructors — the legacy `Add`/`Log`/… nodes as thin wrappers.
// ===========================================================================

/// A [`Unary`] whose callable is a plain function pointer (the type of every
/// non-capturing named constructor below — nameable, `Copy`, `Send + Sync`).
pub type UnaryFn<T, const N: usize> = Unary<T, N, fn(T) -> T>;
/// A [`Binary`] whose callable is a plain function pointer.
pub type BinaryFn<T, const N: usize> = Binary<T, N, fn(T, T) -> T>;

macro_rules! define_unary {
    ($(#[$meta:meta])* $name:ident [$($bounds:tt)*], |$x:ident| $body:expr) => {
        $(#[$meta])*
        pub fn $name<T: Scalar + $($bounds)*, const N: usize>() -> UnaryFn<T, N> {
            Unary::new(|$x| $body)
        }
    };
}

macro_rules! define_binary {
    ($(#[$meta:meta])* $name:ident [$($bounds:tt)*], |$a:ident, $b:ident| $body:expr) => {
        $(#[$meta])*
        pub fn $name<T: Scalar + $($bounds)*, const N: usize>() -> BinaryFn<T, N> {
            Binary::new(|$a, $b| $body)
        }
    };
}

define_unary!(/// Element-wise negation: `-a`.
    negate [ops::Neg<Output = T>], |x| -x);
define_unary!(/// Element-wise natural logarithm.
    log [Float], |x| x.ln());
define_unary!(/// Element-wise base-2 logarithm.
    log2 [Float], |x| x.log2());
define_unary!(/// Element-wise base-10 logarithm.
    log10 [Float], |x| x.log10());
define_unary!(/// Element-wise exponential.
    exp [Float], |x| x.exp());
define_unary!(/// Element-wise base-2 exponential.
    exp2 [Float], |x| x.exp2());
define_unary!(/// Element-wise square root.
    sqrt [Float], |x| x.sqrt());
define_unary!(/// Element-wise ceiling.
    ceil [Float], |x| x.ceil());
define_unary!(/// Element-wise floor.
    floor [Float], |x| x.floor());
define_unary!(/// Element-wise rounding.
    round [Float], |x| x.round());
define_unary!(/// Element-wise reciprocal: `1/x`.
    recip [Float], |x| x.recip());
define_unary!(/// Element-wise absolute value.
    abs [Signed], |x| x.abs());
define_unary!(/// Element-wise signum (−1, 0, or +1).
    sign [Signed], |x| x.signum());

define_binary!(/// Element-wise addition: `a + b`.
    add [ops::Add<Output = T>], |a, b| a + b);
define_binary!(/// Element-wise subtraction: `a - b`.
    subtract [ops::Sub<Output = T>], |a, b| a - b);
define_binary!(/// Element-wise multiplication: `a * b`.
    multiply [ops::Mul<Output = T>], |a, b| a * b);
define_binary!(/// Element-wise division: `a / b`.
    divide [ops::Div<Output = T>], |a, b| a / b);
define_binary!(/// Element-wise minimum (IEEE 754).
    min [Float], |a, b| a.min(b));
define_binary!(/// Element-wise maximum (IEEE 754).
    max [Float], |a, b| a.max(b));

/// Element-wise power: `x.powf(n)` (the one parameterized unary — its callable
/// captures the exponent, so it is not a plain function pointer).
pub fn pow<T: Scalar + Float, const N: usize>(n: T) -> Unary<T, N, impl Fn(T) -> T + Send + Sync> {
    Unary::new(move |x: T| x.powf(n))
}
