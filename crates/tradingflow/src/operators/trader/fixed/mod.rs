//! Operators that simulate trading at a fixed price (e.g. closing price)
//! per period. Prices are given as [`f64`] to ensure precision.

mod base;
mod benchmark;
mod random;
mod simple;

pub use base::{Exec, Fixed, FixedState};
pub use benchmark::benchmark;
pub use random::random;
pub use simple::simple;
