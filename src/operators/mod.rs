//! Built-in operators for the DAG runtime.
//!
//! Every operator in this module implements [`Operator`](crate::Operator).
//! Operators are registered into a [`Scenario`](crate::Scenario) via
//! [`Scenario::add_operator`](crate::Scenario::add_operator).
//!
//! # Structural operators
//!
//! - [`Const`] — 0-input node holding a fixed initial value; output can be
//!   mutated externally via [`Scenario::value_mut`](crate::Scenario::value_mut).
//! - [`Id`] — identity passthrough (`T → T`); useful as a trigger-gated node.
//! - [`Filter`] — passes or drops the entire input `Array<T>` based on a
//!   predicate closure.
//! - [`Where`] — element-wise conditional replacement: keeps values where the
//!   condition holds, fills with a constant otherwise.
//!
//! # Array reshape / selection operators
//!
//! - [`Select`] — index selection along an axis (precomputed flat index map).
//! - [`Concat`] — concatenate N arrays along an existing axis (variadic input).
//! - [`Stack`] — stack N arrays along a new axis (variadic input).
//! - [`Cast`] — element-wise type conversion between `Array<S>` and `Array<T>`
//!   via `num_traits::AsPrimitive`.
//!
//! # Series operators
//!
//! - [`Record`] — records each `Array<T>` tick into a `Series<T>`.
//! - [`Last`] — extracts the most recent element from a `Series<T>` as an
//!   `Array<T>`. Two-sided inverse of `Record`.
//! - [`Lag`] — outputs the value from N steps ago in a `Series<T>`, with a fill
//!   value for insufficient history.
//!
//! # Element-wise numeric operators ([`num`])
//!
//! Stateless, element-wise arithmetic and math on `Array<T>`:
//!
//! - Unary arithmetic: [`Negate`].
//! - Binary arithmetic: [`Add`], [`Subtract`], [`Multiply`], [`Divide`].
//! - Unary math (float): [`Log`], [`Log2`], [`Log10`], [`Exp`], [`Exp2`],
//!   [`Sqrt`], [`Ceil`], [`Floor`], [`Round`], [`Recip`].
//! - Unary math (signed): [`Abs`], [`Sign`].
//! - Binary math (float): [`Min`], [`Max`].
//! - Parameterized unary: [`Pow`], [`Scale`], [`Shift`], [`Clamp`], [`Fillna`].
//!
//! # Rolling (windowed) operators ([`rolling`])
//!
//! Stateful, incremental operators on `Series<T>` with O(1) per-tick cost:
//!
//! - [`rolling::RollingSum`], [`rolling::RollingMean`],
//!   [`rolling::RollingVariance`], [`rolling::RollingCovariance`].
//! - [`rolling::Ema`] — exponential moving average.
//! - [`rolling::ForwardFill`] — replaces NaN with last valid observation.

pub mod cast;
pub mod concat;
pub mod r#const;
pub mod filter;
pub mod id;
pub mod lag;
pub mod last;
pub mod num;
pub mod record;
pub mod rolling;
pub mod select;
pub mod stack;
pub mod r#where;

pub use cast::Cast;
pub use concat::Concat;
pub use r#const::Const;
pub use filter::Filter;
pub use id::Id;
pub use lag::Lag;
pub use last::Last;
pub use num::{
    Abs, Add, Ceil, Clamp, Divide, Exp, Exp2, Fillna, Floor, Log, Log2, Log10, Max, Min, Multiply,
    Negate, Pow, Recip, Round, Scale, Shift, Sign, Sqrt, Subtract,
};
pub use record::Record;
pub use select::Select;
pub use stack::Stack;
pub use r#where::Where;
