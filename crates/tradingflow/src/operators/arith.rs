//! Element-wise numeric, comparison and logical operators.
//!
//! The whole family collapses to two operators parameterized by a callable —
//! [`UnaryMap<T, U, N, F>`] (`Fn(T) -> U`) and [`BinaryMap<T, U, N, F>`]
//! (`Fn(T, T) -> U`) — over the strided [`ArrayView`] currency, sharing the one
//! contiguous-fast / strided-slow elementwise core
//! ([`apply_unary`](tradingflow_data::array::apply_unary) / `apply_binary`).
//!
//! Fixing the output scalar `U` names the three sub-families:
//!
//! | `U` | Alias | Family | Constructors |
//! | --- | --- | --- | --- |
//! | `T` | [`Unary`] / [`Binary`] | arithmetic | [`add`], [`log`], … |
//! | `bool` | [`Predicate`] / [`Compare`] | comparison | [`greater`], [`greater_than`], [`is_finite`], … |
//! | `T`, at `T = bool` | [`Unary`] / [`Binary`] | logical | [`and`], [`or`], [`not`], … |
//!
//! and [`Indicator`] (`bool -> T`) / [`Choose`] (`(bool, T, T) -> T`) read a
//! mask back into the numeric currency. The named constructors are thin wrappers
//! that pin the callable, so `add()` / `log()` read like the old `Add` / `Log`
//! nodes while inferring the rank `N` from the wiring.
//!
//! A worked signal — `MA(x, 10) - MA(x, 5) > 0 AND NOT LAG(that, 1) > 0`, over
//! a live array handle via the self-recording formula constructors
//! ([`ma`](super::ma), [`lag`](super::lag), …; every generic is inferred from
//! the one parameter annotation):
//!
//! ```ignore
//! tradingflow::segment!(|x: ArrayPort<f64, 1>| -> ArrayPort<bool, 1> {
//!     let d = subtract() @ (ma(10) @ x, ma(5) @ x);
//!     and() @ (greater_than(0.0) @ d, not() @ (greater_than(0.0) @ lag(1) @ d))
//! })
//! ```
//!
//! Comparisons follow IEEE 754: a `NaN` operand compares `false` under every
//! ordering predicate (and `true` under [`not_equal`] / [`not_equal_to`]). Use
//! [`is_finite`] to test for missing data explicitly rather than relying on a
//! comparison to filter it.

use std::marker::PhantomData;
use std::ops;

use num_traits::{Float, Signed};

use crate::data::array::{apply_binary, apply_unary};
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::operators::op::ArrayPort;

// ===========================================================================
// UnaryMap — one elementwise operator, parameterized by a callable.
// ===========================================================================

/// Element-wise `out[i] = f(x[i])` over a rank-`N` [`ArrayView`], mapping the
/// input scalar `T` to the output scalar `U`.
pub struct UnaryMap<T: Scalar, U: Scalar, const N: usize, F> {
    f: F,
    _p: PhantomData<(T, U)>,
}

impl<T: Scalar, U: Scalar, const N: usize, F: Fn(T) -> U + Send + Sync + 'static>
    UnaryMap<T, U, N, F>
{
    pub fn new(f: F) -> Self {
        Self { f, _p: PhantomData }
    }
}

/// Runtime state for [`UnaryMap`]: the callable plus the output buffer.
pub struct UnaryMapState<T: Scalar, U: Scalar, const N: usize, F> {
    f: F,
    out: Array<U, N>,
    _p: PhantomData<T>,
}

impl<T: Scalar, U: Scalar, const N: usize, F: Fn(T) -> U + Send + Sync + 'static> Operator
    for UnaryMap<T, U, N, F>
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<U, N>;
    type Context = Instant;
    type State = UnaryMapState<T, U, N, F>;

    fn init(self) -> Self::State {
        UnaryMapState {
            f: self.f,
            out: Array::zeros([0; N]),
            _p: PhantomData,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, U, N>) {
        // The build call seeds the output with the transformed build value (not
        // zeros — a fabricated finite observation would leak through carry
        // readers) but does not notify.
        if init {
            state.out = Array::zeros(x.extents());
        }
        apply_unary(&mut state.out, x, &state.f);
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, U, N>) {
        (false, state.out.view())
    }
}

// ===========================================================================
// BinaryMap — the two-input elementwise operator.
// ===========================================================================

/// Element-wise `out[i] = f(a[i], b[i])` over rank-`N` [`ArrayView`]s sharing
/// extents, mapping the input scalar `T` to the output scalar `U`.
pub struct BinaryMap<T: Scalar, U: Scalar, const N: usize, F> {
    f: F,
    _p: PhantomData<(T, U)>,
}

impl<T: Scalar, U: Scalar, const N: usize, F: Fn(T, T) -> U + Send + Sync + 'static>
    BinaryMap<T, U, N, F>
{
    pub fn new(f: F) -> Self {
        Self { f, _p: PhantomData }
    }
}

/// Runtime state for [`BinaryMap`]: the callable plus the output buffer.
pub struct BinaryMapState<T: Scalar, U: Scalar, const N: usize, F> {
    f: F,
    out: Array<U, N>,
    _p: PhantomData<T>,
}

impl<T: Scalar, U: Scalar, const N: usize, F: Fn(T, T) -> U + Send + Sync + 'static> Operator
    for BinaryMap<T, U, N, F>
{
    type Inputs = (ArrayPort<T, N>, ArrayPort<T, N>);
    type Outputs = ArrayPort<U, N>;
    type Context = Instant;
    type State = BinaryMapState<T, U, N, F>;

    fn init(self) -> Self::State {
        BinaryMapState {
            f: self.f,
            out: Array::zeros([0; N]),
            _p: PhantomData,
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, U, N>) {
        if init {
            state.out = Array::zeros(a.extents());
        }
        apply_binary(&mut state.out, a, b, &state.f);
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, U, N>) {
        (false, state.out.view())
    }
}

// ===========================================================================
// Aliases naming the three sub-families.
// ===========================================================================

/// Arithmetic unary: `T -> T` (also the logical `not` at `T = bool`).
pub type Unary<T, const N: usize, F> = UnaryMap<T, T, N, F>;
/// Arithmetic binary: `(T, T) -> T` (also `and`/`or`/`xor` at `T = bool`).
pub type Binary<T, const N: usize, F> = BinaryMap<T, T, N, F>;
/// Comparison against a constant: `T -> bool`.
pub type Predicate<T, const N: usize, F> = UnaryMap<T, bool, N, F>;
/// Comparison of two arrays: `(T, T) -> bool`.
pub type Compare<T, const N: usize, F> = BinaryMap<T, bool, N, F>;

/// A [`Unary`] whose callable is a plain function pointer (the type of every
/// non-capturing named constructor below — nameable, `Copy`, `Send + Sync`).
pub type UnaryFn<T, const N: usize> = Unary<T, N, fn(T) -> T>;
/// A [`Binary`] whose callable is a plain function pointer.
pub type BinaryFn<T, const N: usize> = Binary<T, N, fn(T, T) -> T>;
/// A [`Predicate`] whose callable is a plain function pointer.
pub type PredicateFn<T, const N: usize> = Predicate<T, N, fn(T) -> bool>;
/// A [`Compare`] whose callable is a plain function pointer.
pub type CompareFn<T, const N: usize> = Compare<T, N, fn(T, T) -> bool>;

// ===========================================================================
// Named constructors — arithmetic.
// ===========================================================================

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

// ===========================================================================
// Named constructors — comparison.
// ===========================================================================

macro_rules! define_compare {
    ($(#[$meta:meta])* $name:ident, |$a:ident, $b:ident| $body:expr) => {
        $(#[$meta])*
        pub fn $name<T: Scalar + PartialOrd, const N: usize>() -> CompareFn<T, N> {
            Compare::new(|$a, $b| $body)
        }
    };
}

define_compare!(/// Element-wise `a > b`; a `NaN` operand yields `false`.
    greater, |a, b| a > b);
define_compare!(/// Element-wise `a >= b`; a `NaN` operand yields `false`.
    greater_equal, |a, b| a >= b);
define_compare!(/// Element-wise `a < b`; a `NaN` operand yields `false`.
    less, |a, b| a < b);
define_compare!(/// Element-wise `a <= b`; a `NaN` operand yields `false`.
    less_equal, |a, b| a <= b);
define_compare!(/// Element-wise `a == b`; a `NaN` operand yields `false`.
    equal, |a, b| a == b);
define_compare!(/// Element-wise `a != b`; a `NaN` operand yields `true`.
    not_equal, |a, b| a != b);

macro_rules! define_predicate {
    ($(#[$meta:meta])* $name:ident, |$x:ident, $v:ident| $body:expr) => {
        $(#[$meta])*
        pub fn $name<T: Scalar + PartialOrd, const N: usize>(
            $v: T,
        ) -> Predicate<T, N, impl Fn(T) -> bool + Send + Sync> {
            Predicate::new(move |$x: T| $body)
        }
    };
}

define_predicate!(/// Element-wise `x > v`; `NaN` yields `false`.
    greater_than, |x, v| x > v);
define_predicate!(/// Element-wise `x >= v`; `NaN` yields `false`.
    at_least, |x, v| x >= v);
define_predicate!(/// Element-wise `x < v`; `NaN` yields `false`.
    less_than, |x, v| x < v);
define_predicate!(/// Element-wise `x <= v`; `NaN` yields `false`.
    at_most, |x, v| x <= v);
define_predicate!(/// Element-wise `x == v`; `NaN` yields `false`.
    equal_to, |x, v| x == v);
define_predicate!(/// Element-wise `x != v`; `NaN` yields `true`.
    not_equal_to, |x, v| x != v);

/// Element-wise `x.is_finite()` — test for missing data explicitly, rather than
/// relying on an ordering comparison to filter `NaN`s.
pub fn is_finite<T: Scalar + Float, const N: usize>() -> PredicateFn<T, N> {
    Predicate::new(|x: T| x.is_finite())
}

/// Element-wise `x.is_nan()`.
pub fn is_nan<T: Scalar + Float, const N: usize>() -> PredicateFn<T, N> {
    Predicate::new(|x: T| x.is_nan())
}

// ===========================================================================
// Named constructors — logical connectives (`Unary`/`Binary` at `T = bool`).
// ===========================================================================

/// Element-wise logical AND. Both masks are always evaluated — a graph edge
/// carries a whole cross-section, so there is nothing to short-circuit.
pub fn and<const N: usize>() -> BinaryFn<bool, N> {
    Binary::new(|a, b| a && b)
}

/// Element-wise logical OR. Both masks are always evaluated.
pub fn or<const N: usize>() -> BinaryFn<bool, N> {
    Binary::new(|a, b| a || b)
}

/// Element-wise logical XOR.
pub fn xor<const N: usize>() -> BinaryFn<bool, N> {
    Binary::new(|a, b| a ^ b)
}

/// Element-wise logical NOT.
pub fn not<const N: usize>() -> UnaryFn<bool, N> {
    Unary::new(|a| !a)
}

// ===========================================================================
// Mask consumers — back from `bool` into the numeric currency.
// ===========================================================================

/// Read a mask into the numeric currency: `if mask[i] { on } else { off }`.
/// `indicator(1.0, 0.0)` is the 0/1 indicator; `indicator(1.0, f64::NAN)` is the
/// NaN-masking universe filter.
pub fn indicator<T: Scalar, const N: usize>(
    on: T,
    off: T,
) -> UnaryMap<bool, T, N, impl Fn(bool) -> T + Send + Sync> {
    UnaryMap::new(move |m: bool| if m { on.clone() } else { off.clone() })
}

/// Element-wise `if cond[i] { a[i] } else { b[i] }` — the three-input selector.
pub struct Choose<T: Scalar, const N: usize> {
    _p: PhantomData<T>,
}

impl<T: Scalar, const N: usize> Choose<T, N> {
    pub fn new() -> Self {
        Self { _p: PhantomData }
    }
}

impl<T: Scalar, const N: usize> Default for Choose<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Operator for Choose<T, N> {
    type Inputs = (ArrayPort<bool, N>, ArrayPort<T, N>, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self) -> Self::State {
        Array::zeros([0; N])
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        ((_, cond), (_, a), (_, b)): (
            (bool, ArrayView<'a, bool, N>),
            (bool, ArrayView<'a, T, N>),
            (bool, ArrayView<'a, T, N>),
        ),
        _: &Instant,
        out: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            *out = Array::zeros(cond.extents());
        }
        let (cs, as_, bs) = (cond.to_contiguous(), a.to_contiguous(), b.to_contiguous());
        let dst = out.data_mut();
        for i in 0..dst.len() {
            dst[i] = if cs[i] { as_[i].clone() } else { bs[i].clone() };
        }
        (!init, out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (
            (bool, ArrayView<'a, bool, N>),
            (bool, ArrayView<'a, T, N>),
            (bool, ArrayView<'a, T, N>),
        ),
        _: &Instant,
        out: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

/// Element-wise `if cond[i] { a[i] } else { b[i] }` — the three-input selector.
pub fn choose<T: Scalar, const N: usize>() -> Choose<T, N> {
    Choose::new()
}
