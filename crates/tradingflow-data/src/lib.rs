//! Simple N-dimensional arrays and time series.
//!
//! > Compared to the `ndarray` crate, we use a const generic parameter
//! > `N: usize` for the number of array dimensions. This could support an
//! > arbitrary number of dimensions, while the array view type remains [`Copy`]
//! > (in particular, has trivial [`Drop`] implementation) so it interacts
//! > better with the computation graph executor.
//!
//! # Arrays
//!
//! An [`Array<T, N>`] is an `N`-dimensional array of scalars `T`, which owns
//! its data.
//!
//! An [`ArrayView<'a, T, N>`] is a borrowed, possibly strided view of such an
//! array (with lifetime `'a`).
//!
//! The two types are convertible via [`Array::view`] and
//! [`ArrayView::to_array`]`.
//!
//! # Series
//!
//! A [`Series<T, N>`] is a time series whose elements are `N`-dimensional
//! arrays of scalars `T`, which owns its data. Its history may be bounded by a
//! [`Retention`] policy.
//!
//! A [`SeriesView<'a, T, N>`] is a borrowed, possibly strided view of such a
//! series (with lifetime `'a`).
//!
//! The two types are convertible via [`Series::view`] and
//! [`SeriesView::to_series`].
//!
//! In addition, [`Series::to_array`] converts
//! a series into an array with `N + 1` dimensions, and
//! [`SeriesView::to_array_view`] does the similar for a series view.
//!
//! # Timestamps
//!
//! Each element in a time series is associated with an [`Instant`] timestamp,
//! which is a [`Duration`] time span relative to some globally chosen epoch.
//!
//! > The main reason of using them instead of [`std::time`] is to have an
//! > identical memory layout as NumPy `datetime64[ns]`.
//!
//! The time types are "naive" in that no particular time scale (TAI, UTC etc.)
//! is assumed — the interpretation is up to the user. However, a consistent
//! choice should be maintained across a single program.

pub mod array;
pub mod layout;
pub mod scalar;
pub mod schema;
pub mod series;
pub mod time;

pub use array::{Array, ArrayView};
pub use layout::{Layout, Offsets};
pub use scalar::Scalar;
pub use schema::Schema;
pub use series::{Retention, Series, SeriesView};
pub use time::{Duration, Instant};
