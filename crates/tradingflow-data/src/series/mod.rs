//! Time series with a **compile-time element rank** `N`, append-only
//! semantics, bounded retention, and temporal lookups — the [`Array`](crate::Array)-family
//! container for history.
//!
//! * [`Series<T, N>`] — owned, growable: a row-major contiguous value buffer
//!   plus one timestamp per element, growing along the (implicit) time axis.
//!   Lives in operator `State`, exactly like [`Array`](crate::Array).
//! * [`SeriesView<'a, T, N>`] — borrowed, `Copy`, self-contained window:
//!   `&[Instant]` + `&[T]` + the element [`Shape`]. The [`Series`] analogue of
//!   [`ArrayView`](crate::ArrayView), convertible into a rank-`N + 1`
//!   [`ArrayView`](crate::ArrayView) via
//!   [`as_array_view`](SeriesView::as_array_view) (the time axis becomes
//!   axis 0).
//!
//! The element rank `N` is static; the extents are dynamic. A series is
//! conceptually a `[time, extents…]` tensor whose axis 0 grows on
//! [`push`](Series::push) — but the storage stays `Vec`-backed (an
//! [`Array`](crate::Array) is a fixed-size snapshot; a series needs amortized
//! append and front-trim).
//!
//! # Logical vs physical indexing
//!
//! A series addresses its elements by **logical index** — the absolute
//! position since the series was created (`0` is the first element ever
//! pushed). [`len`](Series::len) is the logical length: the total number of
//! elements pushed, regardless of how many are still physically retained.
//!
//! A series may carry a [`Retention`] bound (a maximum element count and/or a
//! maximum time span). When set, [`push`](Series::push) drops the oldest
//! elements from the front so that physical storage stays bounded. A
//! [`base`](Series::base) offset records how many leading elements have been
//! dropped; positional accessors map a logical index `i` to the physical slot
//! `i - base`. The default retention is **unbounded** — nothing is ever
//! dropped, `base` stays `0`, and logical and physical indices coincide.
//!
//! Positional accessors ([`elem`](Series::elem),
//! [`timestamp`](Series::timestamp), [`window`](Series::window)) take logical
//! indices and panic if asked for an element that has been evicted
//! (`i < base`). The bulk accessors that return the whole retained window
//! ([`timestamps`](Series::timestamps), [`data`](Series::data),
//! [`view`](Series::view), [`tail`](Series::tail)) return only the window
//! `[base, len)`; for an unbounded series that is the full history. A
//! [`SeriesView`] is a plain window snapshot — **its indices are view-local**
//! (`0` is the view's first element), not logical.
//!
//! The view is also where reads are *defined*: every derived read on a
//! [`Series`] delegates to the corresponding [`SeriesView`] read on
//! [`view`](Series::view), after mapping logical indices to the retained
//! window. Within that window, a bounded series therefore reads identically
//! to an unbounded one by construction.
//!
//! # Views vs flat data
//!
//! Accessors hand out [`ArrayView`](crate::ArrayView)/[`SeriesView`] borrows
//! rather than flat slices: [`elem`](Series::elem) and [`last`](Series::last)
//! for one element, [`view`](Series::view)/[`window`](Series::window)/[`tail`](Series::tail)
//! for a range, and [`push`](Series::push) to append one. The methods that
//! name `data` are the flat row-major escape hatches —
//! [`data`](Series::data)/[`data_mut`](Series::data_mut) for the whole
//! retained buffer, [`elem_data`](Series::elem_data) for one element, and
//! [`push_data`](Series::push_data) to append from a slice.

use crate::{Instant, Scalar, Shape};

mod iter;
mod owned;
mod retention;
mod view;

pub use iter::{SeriesIntoIter, SeriesIter};
pub use retention::Retention;

// ===========================================================================
// Series<T, N> — owned, growable, row-major contiguous backing store
// ===========================================================================

/// A time series of rank-`N` array elements in row-major contiguous layout.
///
/// Timestamps are [`Instant`]s (SI nanoseconds since the TAI epoch) in
/// non-decreasing order. See the [module docs](self) for logical vs physical
/// indexing and bounded retention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Series<T: Scalar, const N: usize> {
    /// Physically retained timestamps (one per element).
    timestamps: Vec<Instant>,
    /// Physically retained elements (row-major, `shape.len()` scalars each).
    data: Vec<T>,
    /// Element shape (canonical row-major; the time axis is implicit).
    shape: Shape<N>,
    /// Logical index of physical element `0` — the count of elements dropped
    /// off the front. `0` for an unbounded series.
    base: usize,
    /// Retention bound applied on [`push`](Self::push).
    retention: Retention,
}

// ===========================================================================
// SeriesView<'a, T, N> — borrowed, Copy, self-contained window
// ===========================================================================

/// A borrowed window of a [`Series`]: one timestamp slice, one packed
/// row-major value slice, and the element [`Shape`] — the
/// [`ArrayView`](crate::ArrayView) analogue for history.
///
/// Indices are **view-local** (`0` is the view's first element); a view
/// carries no logical/retention bookkeeping. `Copy` (the payload is
/// references plus plain `usize`s) and fully lifetime-checked — a view cannot
/// outlive the series it borrows from. Convertible into a rank-`N + 1`
/// [`ArrayView`](crate::ArrayView) via [`as_array_view`](Self::as_array_view)
/// (elements are packed, so the conversion is zero-copy).
#[derive(Debug)]
pub struct SeriesView<'a, T: Scalar, const N: usize> {
    /// One timestamp per element, non-decreasing.
    timestamps: &'a [Instant],
    /// Packed row-major elements: `timestamps.len() * shape.len()` scalars.
    data: &'a [T],
    /// Element shape (canonical row-major; the time axis is axis 0 of `data`).
    shape: Shape<N>,
}

// Manual (not derived) `Clone`/`Copy`: the view is references + plain `usize`s,
// so it is `Copy` regardless of whether `T` is. (Derived `Copy` would wrongly
// demand `T: Copy`, which e.g. `String` does not satisfy.)
impl<T: Scalar, const N: usize> Clone for SeriesView<'_, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Scalar, const N: usize> Copy for SeriesView<'_, T, N> {}
