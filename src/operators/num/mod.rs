//! Built-in element-wise numeric operators.
//!
//! Every operator in this module implements [`Operator`](crate::Operator) with
//! stateless computation unless otherwise noted. Inputs and outputs are
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
//! - [`Clamp`] (`T: Float`) — clamp to `[lo, hi]`.
//! - [`Fillna`] (`T: Float`) — replace NaN with a constant.
//! - [`ForwardFill`] (`T: Float`) — replace NaN with the last valid observation.
//!
//! # Cross-tick (stateful, `T: Float`)
//!
//! These maintain a ring buffer of the last `offset` input arrays.
//!
//! - [`Diff`] — element-wise first difference: `input - input_{offset back}`.
//! - [`PctChange`] — element-wise linear return: `input / input_{offset back} - 1`.
//!
//! Combining these with the unary math operators yields the standard
//! return conventions: linear returns are `PctChange`, while log returns
//! are `Log -> Diff`.
//!
//! # Distribution shaping (float-only, `T: Float`, 1-D input)
//!
//! Cross-sectional rank statistics that sort and handle NaN internally:
//! non-NaN entries are ranked ascending (denominator is `n_valid`, not
//! `n`) and NaN inputs propagate to NaN outputs, so downstream
//! `is_finite` masks still filter missing entries.
//!
//! - [`Gaussianize`] — cross-sectional rank-to-Gaussian: map each
//!   non-NaN element to `Φ⁻¹((rank + 0.5) / n_valid)`.
//! - [`Percentile`] — cross-sectional rank-to-percentile: map each
//!   non-NaN element to `(rank + 0.5) / n_valid ∈ (0, 1)`.  Same sort
//!   and NaN logic as `Gaussianize`, just without the `Φ⁻¹` step.

mod arithmetic;
mod clamp;
mod diff;
mod ffill;
mod fillna;
mod gaussianize;
mod pct_change;
mod percentile;

pub use arithmetic::{
    Abs, Add, Ceil, Divide, Exp, Exp2, Floor, Log, Log2, Log10, Max, Min, Multiply, Negate, Pow,
    Recip, Round, Sign, Sqrt, Subtract,
};
pub use clamp::Clamp;
pub use diff::Diff;
pub use ffill::ForwardFill;
pub use fillna::Fillna;
pub use gaussianize::Gaussianize;
pub use pct_change::PctChange;
pub use percentile::Percentile;
