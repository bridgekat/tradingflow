//! Basic event-semantics operators on arrays.
//!
//! These are operators with non-trivial behavior in terms of the notification
//! flags or NaN status of their inputs or outputs:
//!
//! - A non-NaN element in an array with `notify = true` is considered
//!   as carrying a one-time "event";
//! - A NaN element or an element in an array with `notify = false` is
//!   typically ignored.
//!
//! For non-floating scalar types inside arrays, `Option<T>` is used to
//! represent the presence or absence of an event, where `None` has the same
//! meaning as NaN.

mod clock;
mod concat;
mod filter;
mod resample;

pub use clock::clock;
pub use concat::{concat_sync, stack_sync};
pub use filter::filter;
pub use resample::resample;
