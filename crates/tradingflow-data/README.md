# `tradingflow-data`

Simple N-dimensional arrays and time series.

> Compared to the `ndarray` crate, we use a const generic parameter `N: usize` for the number of array dimensions. This could support an arbitrary number of dimensions, while the array view type remains `Copy` (in particular, has trivial `Drop` implementation) so it interacts better with the computation graph scheduler.

## Arrays

The `Array<T, N>` type is an `N`-dimensional array of scalars `T`, which owns its data.

The `ArrayView<'a, T, N>` type is an `N`-dimensional array of scalars `T`, which borrows its data from elsewhere (with lifetime `'a`).

The `Array::view()` method creates an `ArrayView` borrow of its data, and `ArrayView::to_array()` copies the borrowed data to create an owned `Array`.

## Series

The `Series<T, N>` type is a time series whose elements are `N`-dimensional arrays of scalars `T`, which owns its data.

The `SeriesView<'a, T, N>` type is a time series whose elements are `N`-dimensional arrays, which borrows its data from elsewhere (with lifetime `'a`).

The `Series::view()` method creates a `SeriesView` borrow of its data, and `SeriesView::to_series()` copies the borrowed data to create an owned `Series`.

## Timestamps

Each element in a time series is associated with a custom `Instant` timestamp, which contains the number of SI nanoseconds since `1970-01-01 00:00:00 TAI`.

Two timestamps can be subtracted to produce a `Duration` time span, which contains a number of SI nanoseconds.

Calculating time spans on the TAI scale avoids issues caused by UTC leap seconds, but requires a conversion from standard UTC timestamps. We use the `hifitime` crate to perform this conversion under the hood.
