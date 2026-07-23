//! Simple N-dimensional arrays and time series.
//!
//! > Compared to the `ndarray` crate, we use a const generic parameter
//! > `N: usize` for the number of array dimensions. This could support an
//! > arbitrary number of dimensions, while the array view type remains [`Copy`]
//! > (in particular, has trivial [`Drop`] implementation) so it interacts
//! > better with the computation graph executor.
//!
//! # Examples
//!
//! Basic operations on arrays:
//!
//! ```rust
//! use tradingflow::data::*;
//!
//! // Create an owned row-major contiguous array.
//! let mut a = Array::zeros([2, 3]);
//!
//! // Element indexing and assignment.
//! a[[0, 2]] = 3;
//! a[[1, 1]] = 5;
//! a[[1, 2]] = 6;
//!
//! // Reshape it into a 1-dimensional array.
//! let a = a.reshape([6]);
//! assert_eq!(a[[2]], 3);
//! assert_eq!(a[[4]], 5);
//! assert_eq!(a[[5]], 6);
//!
//! // Reshape back, create a view.
//! // Owned arrays support reshaping; views support slicing.
//! let a = a.reshape([2, 3]);
//! let view = a.view();
//!
//! // The slicing syntax is similar to NumPy.
//! // Note that the slicing specifiers are inside a tuple instead of an
//! // array, as they may have different types.
//! assert_eq!(view.slice((0..2, 1..3)).extents(), [2, 2]);
//! assert_eq!(view.slice((.., 1..3)).extents(), [2, 2]);
//! assert_eq!(view.slice((.., 1..)).extents(), [2, 2]);
//!
//! // We can add new axes to, or remove axes from, a view.
//! // However, the resulting number of dimensions must be written
//! // explicitly (will panic at runtime if incorrect).
//! let row = view.slice_reshape::<1, _>((0, ..));
//! let ext = view.slice_reshape::<4, _>((.., NewAxis, .., NewAxis));
//! assert_eq!(row.extents(), [3]);
//! assert_eq!(ext.extents(), [2, 1, 3, 1]);
//!
//! // To freely reshape a view, the elements need to be deep-cloned
//! // into an owned array first.
//! let b = view.slice((.., 1..)).to_array().reshape([4]);
//! assert_eq!(b, Array::from_parts([4], [0, 3, 5, 6].into()));
//! ```
//!
//! Basic operations on time series:
//!
//! ```rust
//! use tradingflow::data::*;
//!
//! // Create a time series of owned row-major contiguous arrays.
//! let mut a = Series::new([2, 3]);
//!
//! // Add elements to the series.
//! let ts = Instant::from_offset(Duration::from_nanos(0));
//! let val = Array::from_parts([2, 3], [0, 1, 2, 3, 4, 5].into());
//! a.push(ts, val.view());
//! a.push(ts, val.view());
//! a.push(ts, val.view());
//!
//! // Element indexing (direct assignment is not supported yet).
//! assert_eq!(a.at(0), (ts, val.view()));
//!
//! // Element reshaping.
//! let a = a.reshape([6]);
//! assert_eq!(a.at(0).1.extents(), [6]);
//!
//! // Reshape back, create a view.
//! let a = a.reshape([2, 3]);
//! let view = a.view();
//!
//! // Windowing and element-wise slicing.
//! let windowed = view.window(1..3);
//! let sliced = view.slice((.., 1..));
//! assert_eq!(windowed.len(), 2);
//! assert_eq!(windowed.extents(), [2, 3]);
//! assert_eq!(sliced.len(), 3);
//! assert_eq!(sliced.extents(), [2, 2]);
//!
//! // Element-wise slicing works similarly to array views.
//! let row = view.slice_reshape::<1, _>((0, ..));
//! let ext = view.slice_reshape::<4, _>((.., NewAxis, .., NewAxis));
//! assert_eq!(row.extents(), [3]);
//! assert_eq!(ext.extents(), [2, 1, 3, 1]);
//!
//! // Series and views can be converted into `N + 1`-dimensional arrays and
//! // views, but again the `M = N + 1` must be explicitly written down.
//! let arr_view = view.to_array_view::<3>();
//! assert_eq!(arr_view.extents(), [3, 2, 3]);
//! let arr = a.to_array::<3>();
//! assert_eq!(arr.extents(), [3, 2, 3]);
//! ```
//!
//! Basic operations on schemas:
//!
//! ```rust
//! use tradingflow::data::*;
//!
//! // Create a schema of names.
//! let s = Schema::new(["a", "b", "c"]);
//!
//! // Look up names and indices.
//! assert_eq!(s.name(0), "a");
//! assert_eq!(s.index("c"), 2);
//! assert_eq!(s.indices(["c", "a"]), vec![2, 0]);
//!
//! // Select a subset of names by a list of unique indices.
//! let subset = s.select(&[0, 2]);
//! assert_eq!(subset.names(), &["a", "c"]);
//!
//! // Take the union of two schemas. Names must be unique.
//! let t = Schema::new(["ddd", "eee"]);
//! let all = subset.union(&t);
//! assert_eq!(all.names(), &["a", "c", "ddd", "eee"]);
//! ```
//!
//! # Arrays
//!
//! An [`Array<T, N>`] is an `N`-dimensional array of scalars `T`, which owns
//! its data.
//!
//! An [`ArrayView<'a, T, N>`] is a borrowed, possibly strided view of such an
//! array (with lifetime `'a`).
//!
//! # Series
//!
//! A [`Series<T, N>`] is a time series whose elements are `N`-dimensional
//! arrays of scalars `T`, which owns its data.
//!
//! A [`SeriesView<'a, T, N>`] is a borrowed, possibly strided view of such a
//! series (with lifetime `'a`).
//!
//! > Time series are append-only from the back, and trim-only from the front.
//! > Elements are logically indexed: trimming front elements *does not change*
//! > the logical indices of the remaining elements.
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
//!
//! # Schemas
//!
//! A [`Schema`] is a collection of unique names, which can be used to index
//! into arrays and series. It is useful for representing the columns of a
//! table, or the fields of a record.
//!
//! However, the association between schemas and axes of arrays or series
//! is not enforced; it is up to the user to maintain consistency. The
//! [`Schema`] is no more than a bookkeeping utility.

pub mod array;
pub mod layout;
pub mod scalar;
pub mod schema;
pub mod series;
pub mod time;

pub use array::{Array, ArrayView};
pub use layout::{Layout, Offsets, Slice, SliceReshape};
pub use scalar::Scalar;
pub use schema::Schema;
pub use series::{Series, SeriesView};
pub use time::{Duration, Instant};

pub use SliceReshape::NewAxis;
