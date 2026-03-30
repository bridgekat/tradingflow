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

mod arithmetic;
mod clamp;
mod fillna;
mod pow;
mod scale;
mod shift;

pub use arithmetic::{
    Abs, Add, Ceil, Divide, Exp, Exp2, Floor, Log, Log2, Log10, Max, Min, Multiply, Negate, Recip,
    Round, Sign, Sqrt, Subtract,
};
pub use clamp::Clamp;
pub use fillna::Fillna;
pub use pow::Pow;
pub use scale::Scale;
pub use shift::Shift;
