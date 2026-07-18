//! Iteration over a series' `(timestamp, element)` pairs.
//!
//! Borrowed ([`SeriesIter`]): one shared walk backs [`Series::iter`],
//! [`SeriesView::iter`], and the `&Series` / `SeriesView` `IntoIterator`s,
//! yielding `(Instant, ArrayView)` over the retained window with no copy.
//!
//! Owned ([`SeriesIntoIter`]): by-value `IntoIterator for Series` consumes the
//! series and yields `(Instant, Array)`, moving each element's scalars out of
//! the backing buffer — a consumed series can't lend a borrowing view, so the
//! owned form hands back an owned [`Array`].

use super::{Series, SeriesView};
use crate::{Array, ArrayView, Instant, Scalar, Shape};

/// Iterator over a [`SeriesView`] / [`Series`] window, yielding each element's
/// `(timestamp, `[`ArrayView`]`)` pair in chronological order; created by
/// [`SeriesView::iter`] / [`Series::iter`] (or by iterating a `&Series` or a
/// `SeriesView`). Double-ended and exact-size.
#[derive(Clone, Debug)]
pub struct SeriesIter<'a, T: Scalar, const N: usize> {
    /// The as-yet-unyielded timestamps; `len()` is what remains.
    timestamps: &'a [Instant],
    /// The matching packed elements: `timestamps.len() * shape.len()` scalars.
    data: &'a [T],
    /// Element shape, shared by every element (canonical row-major).
    shape: Shape<N>,
}

impl<'a, T: Scalar, const N: usize> Iterator for SeriesIter<'a, T, N> {
    type Item = (Instant, ArrayView<'a, T, N>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (&ts, rest) = self.timestamps.split_first()?;
        self.timestamps = rest;
        let (block, rest_data) = self.data.split_at(self.shape.len());
        self.data = rest_data;
        Some((ts, ArrayView::from_parts(self.shape, block)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.timestamps.len();
        (n, Some(n))
    }
}

impl<'a, T: Scalar, const N: usize> DoubleEndedIterator for SeriesIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (&ts, rest) = self.timestamps.split_last()?;
        self.timestamps = rest;
        let (rest_data, block) = self.data.split_at(self.data.len() - self.shape.len());
        self.data = rest_data;
        Some((ts, ArrayView::from_parts(self.shape, block)))
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for SeriesIter<'_, T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for SeriesIter<'_, T, N> {}

/// Owned iterator over a [`Series`], yielding each element's
/// `(timestamp, `[`Array`]`)` pair in chronological order and moving its
/// scalars out of the backing buffer; created by `Series`' by-value
/// [`IntoIterator`]. Double-ended and exact-size.
#[derive(Clone, Debug)]
pub struct SeriesIntoIter<T: Scalar, const N: usize> {
    /// The as-yet-unyielded timestamps; `len()` is what remains.
    timestamps: std::vec::IntoIter<Instant>,
    /// The matching packed scalars: `timestamps.len() * shape.len()` of them.
    data: std::vec::IntoIter<T>,
    /// Element shape, shared by every element (canonical row-major).
    shape: Shape<N>,
}

impl<T: Scalar, const N: usize> Iterator for SeriesIntoIter<T, N> {
    type Item = (Instant, Array<T, N>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ts = self.timestamps.next()?;
        // The packed invariant keeps exactly `shape.len()` scalars per remaining
        // timestamp, so the next front block is a whole element.
        let block: Vec<T> = self.data.by_ref().take(self.shape.len()).collect();
        Some((ts, Array::from_vec(self.shape.extents(), block)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.timestamps.len();
        (n, Some(n))
    }
}

impl<T: Scalar, const N: usize> DoubleEndedIterator for SeriesIntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let ts = self.timestamps.next_back()?;
        // Pull the trailing element's scalars off the back (they come reversed),
        // then restore row-major order.
        let mut block: Vec<T> = (0..self.shape.len())
            .map(|_| self.data.next_back().expect("packed series invariant"))
            .collect();
        block.reverse();
        Some((ts, Array::from_vec(self.shape.extents(), block)))
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for SeriesIntoIter<T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for SeriesIntoIter<T, N> {}

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// Iterate the window's elements in chronological order, yielding each
    /// `(timestamp, `[`ArrayView`]`)` pair. [`IntoIterator`] for the view does
    /// the same.
    pub fn iter(&self) -> SeriesIter<'a, T, N> {
        SeriesIter {
            timestamps: self.timestamps,
            data: self.data,
            shape: self.shape,
        }
    }
}

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Iterate the retained window's elements in chronological order, yielding
    /// each `(timestamp, `[`ArrayView`]`)` pair.
    pub fn iter(&self) -> SeriesIter<'_, T, N> {
        self.view().iter()
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for &'a Series<T, N> {
    type Item = (Instant, ArrayView<'a, T, N>);
    type IntoIter = SeriesIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.view().iter()
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for SeriesView<'a, T, N> {
    type Item = (Instant, ArrayView<'a, T, N>);
    type IntoIter = SeriesIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Scalar, const N: usize> IntoIterator for Series<T, N> {
    type Item = (Instant, Array<T, N>);
    type IntoIter = SeriesIntoIter<T, N>;

    /// Consume the series, yielding each retained element as an owned
    /// `(timestamp, `[`Array`]`)` pair in chronological order.
    fn into_iter(self) -> Self::IntoIter {
        SeriesIntoIter {
            timestamps: self.timestamps.into_iter(),
            data: self.data.into_iter(),
            shape: self.shape,
        }
    }
}
