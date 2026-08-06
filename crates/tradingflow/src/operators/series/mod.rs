//! Basic operators on series.

mod constant;
mod last;
mod record;
mod shift;
mod view;

pub use constant::constant;
pub use last::{last, last_or};
pub use record::{buffer, record_all, record_on};
pub use shift::shift;
pub use view::{derive_view, move_axis, pad_ndim, permute_axes, slice, slice_reshape, swap_axes};
