//! Layer 0: untyped graph and builder.

mod cell;
mod error;
mod graph;
mod segment;

pub use cell::ErasedCell;
pub use error::Error;
pub use graph::{Adjacency, Builder, Graph};
pub use segment::{ComputeFn, Segment};
