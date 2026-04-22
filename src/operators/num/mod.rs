//! Built-in element-wise numeric operators.
//!
//! Every operator in this module implements [`Operator`](crate::Operator) with
//! stateless computation (no rolling window). Inputs and outputs are
//! [`Array<T>`](crate::Array) where `T: Scalar`. Operators are generic over
//! the scalar type; trait bounds on `T` vary by category (see below).
//!
//! # Unary arithmetic
//!
//! [`Negate`] — requires `Neg`.
//!
//! # Binary arithmetic
//!
//! [`Add`], [`Subtract`], [`Multiply`], [`Divide`] — require the
//! corresponding `std::ops` trait. These accept two input arrays of the same
//! shape and produce one output array.
//!
//! # Unary math (float-only, `T: Float`)
//!
//! [`Log`], [`Log2`], [`Log10`], [`Exp`], [`Exp2`], [`Sqrt`],
//! [`Ceil`], [`Floor`], [`Round`], [`Recip`].
//!
//! # Unary math (signed, `T: Signed`)
//!
//! [`Abs`], [`Sign`].
//!
//! # Binary math (float-only, `T: Float`)
//!
//! [`Min`], [`Max`] — element-wise minimum/maximum of two arrays
//! (IEEE 754 semantics: returns the non-NaN operand when one is NaN).
//!
//! # Parameterized unary (one input array, constructor takes a constant)
//!
//! - [`Pow`] (`T: Float`) — `x.powf(n)`.
//! - [`Scale`] (`T: Mul`) — `x * c`.
//! - [`Shift`] (`T: Add`) — `x + c`.
//! - [`Clamp`] (`T: Float`) — clamp to `[lo, hi]`.
//! - [`Fillna`] (`T: Float`) — replace NaN with a constant.
//! - [`ForwardFill`] (`T: Float`) — replace NaN with the last valid observation.
//!
//! # Ranking (float-only, `T: Float`, 1-D input)
//!
//! - [`Rank`] — 0-based rank of each element (smallest → 0).
//! - [`ArgSort`] — indices that would sort the array (smallest first).
//!
//! Both treat NaN as larger than any real value, so NaNs end up at the
//! highest ranks / indices.
//!
//! # Distribution shaping (float-only, `T: Float`, 1-D input)
//!
//! - [`Gaussianize`] — cross-sectional rank-to-Gaussian transform: map
//!   each non-NaN element to `Φ⁻¹((rank + 0.5) / n_valid)`.  NaN inputs
//!   are preserved as NaN outputs.

mod arithmetic;
mod clamp;
mod ffill;
mod fillna;
mod gaussianize;
mod pow;
mod rank;
mod scale;
mod shift;

pub use arithmetic::{
    Abs, Add, Ceil, Divide, Exp, Exp2, Floor, Log, Log2, Log10, Max, Min, Multiply, Negate, Recip,
    Round, Sign, Sqrt, Subtract,
};
pub use clamp::Clamp;
pub use ffill::ForwardFill;
pub use fillna::Fillna;
pub use gaussianize::Gaussianize;
pub use pow::Pow;
pub use rank::{ArgSort, Rank};
pub use scale::Scale;
pub use shift::Shift;
