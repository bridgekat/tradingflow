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
//! - [`Map`] — applies a function `S → T` to transform input into output.
//! - [`MapInplace`] — applies a function `(&S, &mut T) → bool` in place.
//! - [`Apply`] — applies a function `Inputs → T` to transform tuple inputs
//!   into output.
//! - [`ApplyInplace`] — applies a function `(Inputs, &mut T) → bool` in place.
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
//! # Sub-modules
//!
//! - [`metrics`] — clock-driven financial metrics.
//! - [`num`] — element-wise numeric operators.
//! - [`rolling`] — rolling (windowed) operators.
//! - [`stocks`] — stock-specific operators.

pub mod apply;
pub mod cast;
pub mod clocked;
pub mod concat;
pub mod metrics;
pub mod r#const;
pub mod filter;
pub mod id;
pub mod lag;
pub mod map;
pub mod last;
pub mod num;
pub mod record;
pub mod rolling;
pub mod select;
pub mod stack;
pub mod stocks;
pub mod r#where;

pub use cast::Cast;
pub use clocked::Clocked;
pub use concat::Concat;
pub use r#const::Const;
pub use filter::Filter;
pub use id::Id;
pub use lag::Lag;
pub use apply::{Apply, ApplyInplace};
pub use map::{Map, MapInplace};
pub use last::Last;
pub use record::Record;
pub use select::Select;
pub use stack::Stack;
pub use r#where::Where;
