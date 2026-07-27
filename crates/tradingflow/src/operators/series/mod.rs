//! Basic series operators.

mod constant;
mod last;
mod record;
mod shift;
mod view;

pub use constant::{constant, empty, from_parts};
pub use last::{last, last_or};
pub use record::{buffer, record, record_all, record_clocked};
pub use shift::shift;
pub use view::{derive_view, pad_ndim, slice, slice_reshape, transpose};
