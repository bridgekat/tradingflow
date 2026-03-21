//! Concrete operator implementations.
//!
//! * [`Apply`] — closure-based homogeneous operator (variable arity).
//! * [`Add`], [`Subtract`], [`Multiply`], [`Divide`], [`Negate`] —
//!   element-wise arithmetic (typed tuple inputs via [`InputTuple`]).
//! * [`Select`] — index selection along an axis.
//! * [`Filter`] — whole-element predicate filter.
//! * [`Where`] — element-wise conditional replacement.
//! * [`Concat`] — concatenation along an existing axis (variable arity).
//! * [`Stack`] — stacking along a new axis (variable arity).

mod apply;
mod concat;
mod filter;
mod select;
mod stack;
mod r#where;

pub use apply::{
    Add, Apply1, Apply2, Divide, Multiply, Negate, Subtract, add, divide, multiply, negate,
    subtract,
};
pub use concat::Concat;
pub use filter::Filter;
pub use select::Select;
pub use stack::Stack;
pub use r#where::Where;
