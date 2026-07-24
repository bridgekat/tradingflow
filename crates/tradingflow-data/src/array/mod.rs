//! [`Array`] and [`ArrayView`].

mod concat;
mod iter;
mod map;
mod owned;
mod select;
mod split;
mod view;

pub use concat::{concat, concat_into, stack, stack_into};
pub use iter::{ArrayIntoIter, ArrayIter};
pub use map::{
    binary_for_each, binary_map, binary_map_into, for_each, map, map_into, ternary_for_each,
    ternary_map, ternary_map_into,
};
pub use owned::Array;
pub use select::{select, select_into, select_mask, select_mask_into};
pub use split::{split, unstack};
pub use view::ArrayView;
