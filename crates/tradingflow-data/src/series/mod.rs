//! [`Series`] and [`SeriesView`].

mod iter;
mod owned;
mod retention;
mod view;

pub use iter::{SeriesIntoIter, SeriesIter};
pub use owned::Series;
pub use retention::Retention;
pub use view::SeriesView;
