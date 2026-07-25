//! Basic event-semantics operators.
//!
//! These are operators with non-trivial behavior in terms of the notification
//! flags of their inputs or outputs: a wire with `notify = true` is considered
//! as carrying a one-time "event", and a wire with `notify = false` is
//! typically ignored. See module-level docs of [`crate::graph`] for an
//! overview of notification flags.

mod clock;
mod concat;
mod filter;
mod resample;

pub use clock::clock;
pub use concat::{concat_sync, stack_sync};
pub use filter::filter;
pub use resample::resample;
