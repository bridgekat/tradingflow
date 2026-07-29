//! Basic operators on arrays.

mod concat;
mod constant;
mod map;
mod reshape;
mod select;
mod split;
mod view;

pub use concat::{concat, stack};
pub use constant::{constant, from_parts, full, scalar, zeros};
pub use map::{
    array_binary_map, array_binary_map_inplace, array_map, array_map_inplace, array_ternary_map,
    array_ternary_map_inplace, binary_map, map, ternary_map,
};
pub use reshape::reshape;
pub use select::{select, select_at, select_mask};
pub use split::{split, unstack};
pub use view::{derive_view, pad_ndim, slice, slice_reshape, transpose};
