//! The [`Layout`] trait and concrete layout policies.

mod base;
mod impls;
mod into;
mod iter;
mod slice;

pub use base::Layout;
pub use impls::{ColMajor, RowMajor, Strided};
pub use into::{IntoSliceReshapes, IntoSlices};
pub use iter::Offsets;
pub use slice::{Slice, SliceReshape};
