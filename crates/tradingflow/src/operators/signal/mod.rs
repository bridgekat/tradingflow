//! Basic operators on signal arrays.
//!
//! A signal array is a boolean array that is reset to `false` at the end
//! of each generation. It is used to indicate one-off events occurring at
//! specific instants of time, rather than persistent states. Plain arrays
//! can be used alongside signal arrays to carry payloads (values) for the
//! events, in which case they should be ignored when the signal is `false`.

mod boolean;
mod map;
mod on_signal;

pub use boolean::{always, and, any, filter, or};
pub use map::{Signify, as_signal_map};
pub use on_signal::on_signal;
