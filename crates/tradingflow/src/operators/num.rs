//! Numeric operators over the strided [`ArrayView`] currency: the element-wise
//! arithmetic / comparison / logical family, plus the cross-tick and
//! cross-sectional statistics built on it.
//!
//! # Element-wise
//!
//! The whole element-wise family collapses to two operators parameterized by a
//! callable — [`UnaryMap<T, U, N, F>`] (`Fn(T) -> U`) and
//! [`BinaryMap<T, U, N, F>`] (`Fn(T, T) -> U`) — sharing the one
//! contiguous-fast / strided-slow elementwise core
//! ([`apply_unary`] / `apply_binary`).
//!
//! Fixing the output scalar `U` names the three sub-families:
//!
//! | `U` | Alias | Family | Constructors |
//! | --- | --- | --- | --- |
//! | `T` | [`Unary`] / [`Binary`] | arithmetic | [`add`], [`log`], … |
//! | `bool` | [`Predicate`] / [`Compare`] | comparison | [`greater`], [`greater_than`], [`is_finite`], … |
//! | `T`, at `T = bool` | [`Unary`] / [`Binary`] | logical | [`and`], [`or`], [`not`], … |
//!
//! and [`indicator`] (`bool -> T`) / [`Choose`] (`(bool, T, T) -> T`) read a
//! mask back into the numeric currency. The named constructors are thin wrappers
//! that pin the callable, so `add()` / `log()` read like the old `Add` / `Log`
//! nodes while inferring the rank `N` from the wiring.
//!
//! A worked signal — `MA(x, 10) - MA(x, 5) > 0 AND NOT LAG(that, 1) > 0`, over
//! a live array handle via the self-recording formula constructors
//! ([`ma`](super::formula::ma), [`lag`](super::formula::lag), …; every generic is
//! inferred from the one parameter annotation):
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
//!
//! # Statistics
//!
//! The cross-tick / cross-sectional operators each home an output
//! [`Array<T, N>`] in their state, read the input through
//! [`to_contiguous`](ArrayView::to_contiguous) (zero-copy when the view is
//! contiguous, materialized only when strided), and lend a `ViewPort` view of
//! the output.
//!
//! The build (`init`) call seeds the output with the faithful transform of the
//! build value (not zeros — a fabricated finite observation would leak through
//! carry readers) without notifying; the cross-tick operators ([`Diff`],
//! [`PctChange`], [`ForwardFill`]) instead seed NaN and run no per-tick state
//! update on the build call.

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops;

use num_traits::{Float, Signed};

use crate::data::array::{apply_binary, apply_unary};
use crate::data::{Array, ArrayView, Instant, Layout, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

// ===========================================================================
// Element-wise — the UnaryMap / BinaryMap core and its named constructors.
// ===========================================================================

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

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        // The build call seeds the output with the transformed build value (not
        // zeros — a fabricated finite observation would leak through carry
        // readers); the initial render does not notify.
        let mut state = UnaryMapState {
            f: self.f,
            out: Array::zeros(x.extents()),
            _p: PhantomData,
        };
        apply_unary(&mut state.out, x, &state.f);
        state
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, N>) {
        apply_unary(&mut state.out, x, &state.f);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
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

    fn init(
        self,
        ((_, a), (_, b)): ((bool, ArrayView<'_, T, N>), (bool, ArrayView<'_, T, N>)),
    ) -> Self::State {
        let mut state = BinaryMapState {
            f: self.f,
            out: Array::zeros(a.extents()),
            _p: PhantomData,
        };
        apply_binary(&mut state.out, a, b, &state.f);
        state
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, N>) {
        apply_binary(&mut state.out, a, b, &state.f);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
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

    fn init(
        self,
        ((_, cond), (_, a), (_, b)): (
            (bool, ArrayView<'_, bool, N>),
            (bool, ArrayView<'_, T, N>),
            (bool, ArrayView<'_, T, N>),
        ),
    ) -> Self::State {
        let mut out = Array::zeros(cond.extents());
        let (cs, as_, bs) = (cond.to_contiguous(), a.to_contiguous(), b.to_contiguous());
        let dst = out.data_mut();
        for i in 0..dst.len() {
            dst[i] = if cs[i] { as_[i].clone() } else { bs[i].clone() };
        }
        out
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        ((_, cond), (_, a), (_, b)): (
            (bool, ArrayView<'a, bool, N>),
            (bool, ArrayView<'a, T, N>),
            (bool, ArrayView<'a, T, N>),
        ),
        out: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let (cs, as_, bs) = (cond.to_contiguous(), a.to_contiguous(), b.to_contiguous());
        let dst = out.data_mut();
        for i in 0..dst.len() {
            dst[i] = if cs[i] { as_[i].clone() } else { bs[i].clone() };
        }
        (true, out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (
            (bool, ArrayView<'a, bool, N>),
            (bool, ArrayView<'a, T, N>),
            (bool, ArrayView<'a, T, N>),
        ),
        out: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

/// Element-wise `if cond[i] { a[i] } else { b[i] }` — the three-input selector.
pub fn choose<T: Scalar, const N: usize>() -> Choose<T, N> {
    Choose::new()
}

// ===========================================================================
// Statistics — cross-tick and cross-sectional operators.
// ===========================================================================

// ---------------------------------------------------------------------------
// Clamp
// ---------------------------------------------------------------------------

/// Element-wise clamp to `[lo, hi]`.
#[derive(Clone)]
pub struct Clamp<T: Scalar, const N: usize> {
    lo: T,
    hi: T,
}

impl<T: Scalar + Float, const N: usize> Clamp<T, N> {
    pub fn new(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }
}

/// Runtime state for [`Clamp`]: the bounds plus the output buffer.
pub struct ClampState<T: Scalar, const N: usize> {
    lo: T,
    hi: T,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Clamp<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ClampState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = ClampState {
            lo: self.lo,
            hi: self.hi,
            out: Array::zeros(x.extents()),
        };
        let (lo, hi) = (state.lo, state.hi);
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        for i in 0..dst.len() {
            dst[i] = lo.max(hi.min(src[i]));
        }
        state
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let (lo, hi) = (state.lo, state.hi);
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        for i in 0..dst.len() {
            dst[i] = lo.max(hi.min(src[i]));
        }
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Fillna
// ---------------------------------------------------------------------------

/// Element-wise NaN replacement with a constant.
#[derive(Clone)]
pub struct Fillna<T: Scalar, const N: usize> {
    val: T,
}

impl<T: Scalar + Float, const N: usize> Fillna<T, N> {
    pub fn new(val: T) -> Self {
        Self { val }
    }
}

/// Runtime state for [`Fillna`]: the fill value plus the output buffer.
pub struct FillnaState<T: Scalar, const N: usize> {
    val: T,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Fillna<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = FillnaState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = FillnaState {
            val: self.val,
            out: Array::zeros(x.extents()),
        };
        let val = state.val;
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        for i in 0..dst.len() {
            dst[i] = if src[i].is_nan() { val } else { src[i] };
        }
        state
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let val = state.val;
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        for i in 0..dst.len() {
            dst[i] = if src[i].is_nan() { val } else { src[i] };
        }
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// ForwardFill
// ---------------------------------------------------------------------------

/// Forward-fills NaN with the last valid observation (per element position).
#[derive(Clone)]
pub struct ForwardFill<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> ForwardFill<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for ForwardFill<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float, const N: usize> Operator for ForwardFill<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    // The output buffer doubles as the fill memory: cells keep their last
    // non-NaN value across ticks because the state persists.
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        Array::full(x.extents(), T::nan())
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = out.data_mut();
        for i in 0..dst.len() {
            if !src[i].is_nan() {
                dst[i] = src[i];
            }
        }
        (true, out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

// ---------------------------------------------------------------------------
// Diff (cross-tick first difference)
// ---------------------------------------------------------------------------

/// Element-wise first difference across ticks: `input - input_prev`.
#[derive(Clone)]
pub struct Diff<T: Scalar + Float, const N: usize>(PhantomData<T>);

impl<T: Scalar + Float, const N: usize> Diff<T, N> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float, const N: usize> Default for Diff<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Diff`]: the previous input (NaN-initialised) plus the
/// output buffer.
pub struct DiffState<T: Scalar + Float, const N: usize> {
    prev: Vec<T>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Diff<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = DiffState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        DiffState {
            prev: vec![T::nan(); x.layout().len()],
            out: Array::full(x.extents(), T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        for i in 0..dst.len() {
            dst[i] = src[i] - state.prev[i];
        }
        state.prev.copy_from_slice(src);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// PctChange (cross-tick linear return)
// ---------------------------------------------------------------------------

/// Element-wise one-step linear return: `input / input_prev - 1`.
#[derive(Clone)]
pub struct PctChange<T: Scalar + Float, const N: usize>(PhantomData<T>);

impl<T: Scalar + Float, const N: usize> PctChange<T, N> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float, const N: usize> Default for PctChange<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`PctChange`]: the previous input (NaN-initialised) plus
/// the output buffer.
pub struct PctChangeState<T: Scalar + Float, const N: usize> {
    prev: Vec<T>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for PctChange<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = PctChangeState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        PctChangeState {
            prev: vec![T::nan(); x.layout().len()],
            out: Array::full(x.extents(), T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        let one = T::one();
        for i in 0..dst.len() {
            dst[i] = src[i] / state.prev[i] - one;
        }
        state.prev.copy_from_slice(src);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Gaussianize (cross-sectional rank → Gaussian)
// ---------------------------------------------------------------------------

/// Cross-sectional rank-to-Gaussian transform over the flat cross-section.
#[derive(Clone)]
pub struct Gaussianize<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Gaussianize<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Gaussianize<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Gaussianize`]: the index sort scratch buffer plus the
/// output buffer.
pub struct GaussianizeState<T: Scalar + Float, const N: usize> {
    scratch: Vec<usize>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Gaussianize<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = GaussianizeState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = GaussianizeState {
            scratch: vec![0; x.layout().len()],
            out: Array::zeros(x.extents()),
        };
        // Seed the output with the faithful transform of the build value; the
        // initial render does not notify.
        gaussianize_into(&mut state, x);
        state
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        gaussianize_into(state, x);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The per-call body of [`Gaussianize`]: rank the finite entries of `x` into
/// `state.out` through the inverse normal CDF (NaN elsewhere).
#[inline(always)]
fn gaussianize_into<T: Scalar + Float, const N: usize>(
    state: &mut GaussianizeState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;

    let mut n_valid = 0usize;
    for (i, v) in src.iter().enumerate() {
        if !v.is_nan() {
            state.scratch[n_valid] = i;
            n_valid += 1;
        }
    }
    state.scratch[..n_valid]
        .sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

    let dst = state.out.data_mut();
    let nan = T::nan();
    for slot in dst.iter_mut() {
        *slot = nan;
    }
    if n_valid > 0 {
        let denom = n_valid as f64;
        for rank in 0..n_valid {
            let p = (rank as f64 + 0.5) / denom;
            let z = norm_inv(p);
            dst[state.scratch[rank]] = T::from(z).unwrap_or(nan);
        }
    }
}

/// Inverse standard-normal CDF via Acklam's rational approximation.
#[inline]
fn norm_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const PLOW: f64 = 0.02425;
    const PHIGH: f64 = 1.0 - PLOW;

    if p < PLOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= PHIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Percentile (cross-sectional rank → percentile)
// ---------------------------------------------------------------------------

/// Cross-sectional rank-to-percentile transform over the flat cross-section.
#[derive(Clone)]
pub struct Percentile<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Percentile<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Percentile<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Percentile`]: the index sort scratch buffer plus the
/// output buffer.
pub struct PercentileState<T: Scalar + Float, const N: usize> {
    scratch: Vec<usize>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Percentile<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = PercentileState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = PercentileState {
            scratch: vec![0; x.layout().len()],
            out: Array::zeros(x.extents()),
        };
        // Seed the output with the faithful transform of the build value; the
        // initial render does not notify.
        percentile_into(&mut state, x);
        state
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        percentile_into(state, x);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The per-call body of [`Percentile`]: rank the finite entries of `x` into
/// `state.out` as percentiles in `[0, 1]` (NaN elsewhere).
#[inline(always)]
fn percentile_into<T: Scalar + Float, const N: usize>(
    state: &mut PercentileState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;

    let mut n_valid = 0usize;
    for (i, v) in src.iter().enumerate() {
        if !v.is_nan() {
            state.scratch[n_valid] = i;
            n_valid += 1;
        }
    }
    state.scratch[..n_valid]
        .sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

    let dst = state.out.data_mut();
    let nan = T::nan();
    for slot in dst.iter_mut() {
        *slot = nan;
    }
    if n_valid > 0 {
        let denom = T::from(n_valid as f64).unwrap_or(nan);
        let half = T::from(0.5).unwrap_or(nan);
        for rank in 0..n_valid {
            let p = (T::from(rank as f64).unwrap_or(nan) + half) / denom;
            dst[state.scratch[rank]] = p;
        }
    }
}

// ---------------------------------------------------------------------------
// Standardize (cross-sectional z-score)
// ---------------------------------------------------------------------------

/// Cross-sectional z-score (population std; NaN if < 2 finite or σ = 0).
#[derive(Clone)]
pub struct Standardize<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Standardize<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Standardize<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float, const N: usize> Operator for Standardize<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut out = Array::zeros(x.extents());
        // Seed the output with the faithful transform of the build value; the
        // initial render does not notify.
        standardize_into(&mut out, x);
        out
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        standardize_into(out, x);
        (true, out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

/// The per-call body of [`Standardize`]: z-score the cross-section of `x` into
/// `out` (all NaN if < 2 finite or σ = 0).
#[inline(always)]
fn standardize_into<T: Scalar + Float, const N: usize>(
    out: &mut Array<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;
    let n = src.len();
    let nan = T::nan();

    let mut n_valid = 0usize;
    let mut sum = T::zero();
    for &v in src.iter() {
        if !v.is_nan() {
            n_valid += 1;
            sum = sum + v;
        }
    }

    let dst = out.data_mut();
    if n_valid < 2 {
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        return;
    }

    let mean = sum / T::from(n_valid).unwrap();

    let mut ssd = T::zero();
    for &v in src.iter() {
        if !v.is_nan() {
            let d = v - mean;
            ssd = ssd + d * d;
        }
    }
    let variance = ssd / T::from(n_valid).unwrap();
    let std = variance.sqrt();

    if std <= T::zero() {
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        return;
    }

    for i in 0..n {
        let v = src[i];
        dst[i] = if v.is_nan() { nan } else { (v - mean) / std };
    }
}

// ---------------------------------------------------------------------------
// Winsorize (cross-sectional percentile clipping)
// ---------------------------------------------------------------------------

/// Cross-sectional winsorization: clip non-NaN values to the `[p, 1-p]`
/// quantile range of the cross-section.
#[derive(Clone)]
pub struct Winsorize<T: Scalar + Float, const N: usize> {
    p: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Winsorize<T, N> {
    pub fn new(p: T) -> Self {
        assert!(p >= T::zero(), "Winsorize requires p >= 0");
        assert!(p < T::from(0.5).unwrap(), "Winsorize requires p < 0.5");
        Self {
            p,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Winsorize`]: the quantile, a scratch sort buffer and the
/// output buffer.
pub struct WinsorizeState<T: Scalar + Float, const N: usize> {
    p: T,
    sort_buf: Vec<T>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Winsorize<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = WinsorizeState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = WinsorizeState {
            p: self.p,
            sort_buf: vec![T::zero(); x.layout().len()],
            out: Array::zeros(x.extents()),
        };
        // Seed the output with the faithful transform of the build value; the
        // initial render does not notify.
        winsorize_into(&mut state, x);
        state
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        winsorize_into(state, x);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The per-call body of [`Winsorize`]: clip the cross-section of `x` to its
/// `[p, 1-p]` quantile range into `state.out` (all NaN if nothing finite).
#[inline(always)]
fn winsorize_into<T: Scalar + Float, const N: usize>(
    state: &mut WinsorizeState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;
    let n = src.len();

    let mut n_valid = 0usize;
    for &v in src.iter() {
        if !v.is_nan() {
            state.sort_buf[n_valid] = v;
            n_valid += 1;
        }
    }
    state.sort_buf[..n_valid].sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let dst = state.out.data_mut();
    let nan = T::nan();

    if n_valid == 0 {
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        return;
    }

    let p_f = state.p.to_f64().unwrap_or(0.0);
    let k = ((p_f * n_valid as f64).floor() as usize).min(n_valid - 1);
    let lo = state.sort_buf[k];
    let hi = state.sort_buf[n_valid - 1 - k];

    for i in 0..n {
        let v = src[i];
        if v.is_nan() {
            dst[i] = nan;
        } else if v < lo {
            dst[i] = lo;
        } else if v > hi {
            dst[i] = hi;
        } else {
            dst[i] = v;
        }
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Element-wise clamp into `[lo, hi]` (`NaN` passes through).
pub fn clamp<T: Scalar + Float, const N: usize>(lo: T, hi: T) -> Clamp<T, N> {
    Clamp::new(lo, hi)
}

/// Replace every non-finite element with `val`.
pub fn fillna<T: Scalar + Float, const N: usize>(val: T) -> Fillna<T, N> {
    Fillna::new(val)
}

/// Carry the last finite value forward across ticks.
pub fn forward_fill<T: Scalar + Float, const N: usize>() -> ForwardFill<T, N> {
    ForwardFill::new()
}

/// Element-wise first difference across ticks: `x − x₋₁`. The `n`-tick
/// generalization over a live handle is [`change`](super::formula::change).
pub fn diff<T: Scalar + Float, const N: usize>() -> Diff<T, N> {
    Diff::new()
}

/// Element-wise one-step linear return: `x / x₋₁ − 1`. The `n`-tick
/// generalization over a live handle is [`growth`](super::formula::growth).
pub fn pct_change<T: Scalar + Float, const N: usize>() -> PctChange<T, N> {
    PctChange::new()
}

/// Cross-sectional rank, mapped through the inverse normal CDF.
pub fn gaussianize<T: Scalar + Float, const N: usize>() -> Gaussianize<T, N> {
    Gaussianize::new()
}

/// Cross-sectional percentile rank into `[0, 1]`.
pub fn percentile<T: Scalar + Float, const N: usize>() -> Percentile<T, N> {
    Percentile::new()
}

/// Cross-sectional z-score: `(x − mean) / std`.
pub fn standardize<T: Scalar + Float, const N: usize>() -> Standardize<T, N> {
    Standardize::new()
}

/// Cross-sectionally clip the tails at the `p` / `1 − p` quantiles.
pub fn winsorize<T: Scalar + Float, const N: usize>(p: T) -> Winsorize<T, N> {
    Winsorize::new(p)
}
