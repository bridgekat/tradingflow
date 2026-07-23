//! [`Series`] and [`SeriesView`].

mod iter;
mod owned;
mod shift;
mod view;

pub use iter::{SeriesIntoIter, SeriesIter};
pub use owned::Series;
pub use shift::shift;
pub use view::SeriesView;
