//! Function / selection / lag operators over the strided
//! [`ArrayView`](crate::data::ArrayView) currency — `Map`/`MapInplace`,
//! `Apply`/`ApplyInplace` (closure compute), `Select` (materializing
//! selection), and `Lag` (a `Series` element from N steps ago).
//!
//! The closure operators receive the values-only views tree (notify flags
//! stripped via [`StripNotify`](crate::ports::StripNotify)) and return an owned
//! [`Array`](crate::data::Array), which is homed in `State` and lent as a
//! `ViewPort` view — so a closure reads strided inputs and the result composes
//! as the same view currency.

mod apply;
mod apply_inplace;
mod lag;
mod map;
mod map_inplace;
mod select;

pub use apply::*;
pub use apply_inplace::*;
pub use lag::*;
pub use map::*;
pub use map_inplace::*;
pub use select::*;
