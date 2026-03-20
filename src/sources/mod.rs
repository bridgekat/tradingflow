//! Concrete [`Source`](crate::source::Source) implementations.
//!
//! * [`ArraySource`] — historical-only source backed by pre-loaded arrays.

mod array_source;

pub use array_source::ArraySource;
