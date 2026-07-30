//! [`Schema`] and [`Retention`] helper structs.

mod retention;
mod schema;

pub use retention::Retention;
pub use schema::{Axis, Schema};

#[cfg(feature = "arrow")]
pub mod arrow;
