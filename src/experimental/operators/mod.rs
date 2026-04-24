//! PoC operators for the wavefront execution model.
//!
//! Re-implements a small set of operators with the
//! [`experimental::Operator`](crate::experimental::Operator) trait.

pub mod add;
pub mod diff;
pub mod id;
pub mod multiply;

pub use add::Add;
pub use diff::Diff;
pub use id::Id;
pub use multiply::Multiply;
