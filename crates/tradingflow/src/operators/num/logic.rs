//! Named constructors — logical connectives (`Unary`/`Binary` at `T = bool`)
//! and [`indicator`], the mask consumer reading `bool` back into `T`.

use super::map::{Binary, BinaryFn, Unary, UnaryFn, UnaryMap};
use crate::data::Scalar;

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

/// Read a mask into the numeric currency: `if mask[i] { on } else { off }`.
/// `indicator(1.0, 0.0)` is the 0/1 indicator; `indicator(1.0, f64::NAN)` is the
/// NaN-masking universe filter.
pub fn indicator<T: Scalar, const N: usize>(
    on: T,
    off: T,
) -> UnaryMap<bool, T, N, impl Fn(bool) -> T + Send + Sync> {
    UnaryMap::new(move |m: bool| if m { on.clone() } else { off.clone() })
}
