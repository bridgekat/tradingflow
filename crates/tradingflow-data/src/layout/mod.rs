//! The [`Layout`] trait and concrete layout policies.

mod base;
mod impls;
mod iter;
mod slice;

pub use base::Layout;
pub use impls::{ColMajor, RowMajor, Strided};
pub use iter::Offsets;
pub use slice::{IntoSliceReshapes, IntoSlices, Slice, SliceReshape};
