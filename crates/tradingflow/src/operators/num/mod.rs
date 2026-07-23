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
//! ([`map`] / [`map_binary`] and their in-place `_into` variants).
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
//! [`to_contiguous`](crate::data::ArrayView::to_contiguous) (zero-copy when the
//! view is contiguous, materialized only when strided), and lend a `ViewPort`
//! view of the output.
//!
//! The build (`init`) call seeds the output with the faithful transform of the
//! build value (not zeros — a fabricated finite observation would leak through
//! carry readers) without notifying; the cross-tick operators ([`Diff`],
//! [`PctChange`], [`ForwardFill`]) instead seed NaN and run no per-tick state
//! update on the build call.
//!
//! [`ArrayView`]: crate::data::ArrayView
//! [`Array<T, N>`]: crate::data::Array
//! [`map`]: crate::data::array::map
//! [`map_binary`]: crate::data::array::map_binary

mod arithmetic;
mod choose;
mod clamp;
mod compare;
mod diff;
mod fillna;
mod forward_fill;
mod gaussianize;
mod logic;
mod map;
mod pct_change;
mod percentile;
mod rank;
mod standardize;
mod winsorize;

pub use arithmetic::*;
pub use choose::*;
pub use clamp::*;
pub use compare::*;
pub use diff::*;
pub use fillna::*;
pub use forward_fill::*;
pub use gaussianize::*;
pub use logic::*;
pub use map::*;
pub use pct_change::*;
pub use percentile::*;
pub use standardize::*;
pub use winsorize::*;
