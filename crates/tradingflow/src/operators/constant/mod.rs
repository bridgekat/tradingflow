//! Constant cells: input-free segments that hold a value for the lifetime of
//! the graph and re-emit it on every tick, always notifying.
//!
//! One operator per output shape — `ConstVal` (`Val` copy), `ConstRef` (`Ref`
//! borrow), `ConstArray` and `ConstSeries` (strided [`ArrayView`] /
//! [`SeriesView`] currency).
//!
//! [`ArrayView`]: crate::data::ArrayView
//! [`SeriesView`]: crate::data::SeriesView

mod array;
mod reference;
mod series;
mod val;

pub use array::*;
pub use reference::*;
pub use series::*;
pub use val::*;
