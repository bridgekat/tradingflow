//! [`Array`] and [`ArrayView`].

mod apply;
mod iter;
mod owned;
mod view;

pub use apply::{apply_binary, apply_unary};
pub use iter::{ArrayIntoIter, ArrayIter};
pub use owned::Array;
pub use view::ArrayView;
