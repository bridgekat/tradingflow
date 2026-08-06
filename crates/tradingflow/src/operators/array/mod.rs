//! Basic operators on arrays.

mod concat;
mod constant;
mod map;
mod reduce;
mod reshape;
mod select;
mod split;
mod view;

pub use concat::{concat, stack};
pub use constant::constant;
pub use map::{
    array_binary_map, array_binary_map_inplace, array_map, array_map_inplace, array_ternary_map,
    array_ternary_map_inplace, binary_map, map, ternary_map,
};
pub use reduce::{inner_reduce, outer_reduce, reduce_along_axis};
pub use reshape::reshape;
pub use select::{select, select_at, select_mask};
pub use split::{split, unstack};
pub use view::{derive_view, move_axis, pad_ndim, permute_axes, slice, slice_reshape, swap_axes};
