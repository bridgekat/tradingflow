//! Basic operators on clock signal arrays.

mod boolean;
mod map;
mod on_clock;

pub use boolean::{always, and, any, filter, or};
pub use map::{Clockify, as_clock_map};
pub use on_clock::on_clock;
