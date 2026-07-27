//! Basic event-semantics operators on arrays.
//!
//! These are operators which treats a non-NaN array element as carrying a
//! one-time "event", and resets all outputs to NaN in their reset methods
//! so that each "event" cannot be processed multiple times by downstream
//! operators. Arrays with such interpretation are called "event arrays"
//! in contrast with "state arrays".

mod as_clock;
mod as_event;
mod eventify;
mod filter;
mod sample;

pub use as_clock::as_clock;
pub use as_event::as_event;
pub use eventify::eventify;
pub use filter::filter;
pub use sample::sample;
