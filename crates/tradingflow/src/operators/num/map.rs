//! The element-wise core — [`UnaryMap`] / [`BinaryMap`] — and the aliases
//! naming the arithmetic, comparison and logical sub-families.

use std::marker::PhantomData;

use crate::data::array::{map, map_binary, map_binary_into, map_into};
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

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
        UnaryMapState {
            out: map(x, &self.f),
            f: self.f,
            _p: PhantomData,
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, N>) {
        map_into(state.out.data_mut(), x, &state.f);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, U, N>) {
        (false, state.out.view())
    }
}

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
        BinaryMapState {
            out: map_binary(a, b, &self.f),
            f: self.f,
            _p: PhantomData,
        }
    }

    fn compute<'a, 'b: 'a>(
        ((_, a), (_, b)): ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, N>) {
        map_binary_into(state.out.data_mut(), a, b, &state.f);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, ArrayView<'a, T, N>), (bool, ArrayView<'a, T, N>)),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, U, N>) {
        (false, state.out.view())
    }
}

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
