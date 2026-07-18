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
//! - [`Array::view`] creates an [`ArrayView`] borrow of its data.
//!
//! An [`ArrayView<'a, T, N>`] is a borrowed, possibly strided view of such an
//! array (with lifetime `'a`).
//!
//! - [`ArrayView::to_array`] copies the viewed data into an owned [`Array`].
//!
//! # Series
//!
//! A [`Series<T, N>`] is a time series whose elements are `N`-dimensional
//! arrays of scalars `T`, which owns its data. Its history may be bounded by a
//! [`Retention`] policy.
//!
//! - [`Series::view`] creates a [`SeriesView`] borrow of its data.
//!
//! A [`SeriesView<'a, T, N>`] is a borrowed, possibly strided view of such a
//! series (with lifetime `'a`).
//!
//! - [`SeriesView::to_series`] copies the viewed data into an owned [`Series`].
//!
//! # Timestamps
//!
//! Each element in a time series is associated with a custom [`Instant`]
//! timestamp, stored as a 64-bit integer of SI nanoseconds since
//! `1970-01-01 00:00:00 TAI`.
//!
//! Two timestamps can be subtracted to produce a [`Duration`] time span,
//! stored as a 64-bit integer of SI nanoseconds.
//!
//! Calculating time spans on the TAI scale avoids issues caused by UTC
//! leap seconds, but requires a conversion from standard UTC timestamps.
//! We use the `hifitime` crate to perform this conversion under the hood.

use std::ops::Range;

pub mod array;
pub mod series;
pub mod time;

pub use array::{Array, ArrayView};
pub use series::{Retention, Series, SeriesView};
pub use time::{Duration, Instant, civil_from_days, days_from_civil, tai_to_utc, utc_to_tai};

/// A permitted array scalar type.
pub trait Scalar: Send + Sync + Clone + Default + 'static {}

impl<T: Send + Sync + Clone + Default + 'static> Scalar for T {}

/// Per-axis extents and strides for a rank-`N` array, all inline (`Copy`).
///
/// `strides[d]` is the flat-buffer step for one increment along axis `d`. A
/// [`row_major`](Self::row_major) shape is C-contiguous; a sliced view carries
/// non-canonical strides over the parent's buffer.
///
/// This is where the crate's index arithmetic lives — see the
/// [module docs](self#shapes) for the tour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape<const N: usize> {
    extents: [usize; N],
    strides: [usize; N],
}

impl<const N: usize> Shape<N> {
    /// Canonical row-major (C-contiguous) strides for `extents`.
    pub fn row_major(extents: [usize; N]) -> Self {
        let mut strides = [0usize; N];
        let mut acc = 1usize;
        let mut d = N;
        while d > 0 {
            d -= 1;
            strides[d] = acc;
            acc *= extents[d];
        }
        Shape { extents, strides }
    }

    /// Build a shape from explicit extents and strides (a strided view).
    pub fn strided(extents: [usize; N], strides: [usize; N]) -> Self {
        Shape { extents, strides }
    }

    /// Per-axis extents.
    #[inline(always)]
    pub fn extents(&self) -> [usize; N] {
        self.extents
    }

    /// Per-axis strides.
    #[inline(always)]
    pub fn strides(&self) -> [usize; N] {
        self.strides
    }

    /// Element count (product of extents). The empty product is `1`, so a
    /// rank-0 shape holds one element.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.extents.iter().product()
    }

    /// Whether the shape holds no elements (some extent is zero).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.extents.contains(&0)
    }

    /// Whether the strides are exactly canonical row-major for the extents —
    /// the predicate that gates the contiguous fast path (mdspan
    /// `is_exhaustive`).
    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        self.strides == Shape::row_major(self.extents).strides
    }

    /// Minimal backing-slice length that contains every addressable element —
    /// `1 + Σ (extents[d] - 1)·strides[d]`, or `0` if the shape is empty
    /// (mdspan `required_span_size`). Equals [`len`](Self::len) for a
    /// contiguous shape.
    #[inline(always)]
    pub fn span(&self) -> usize {
        if self.is_empty() {
            return 0;
        }
        let steps: usize = self
            .extents
            .iter()
            .zip(&self.strides)
            .map(|(e, s)| (e - 1) * s)
            .sum();
        1 + steps
    }
}

// ---------------------------------------------------------------------------
// Element addressing
// ---------------------------------------------------------------------------

impl<const N: usize> Shape<N> {
    /// Whether `index` is in bounds on every axis (`index[d] < extents[d]`).
    ///
    /// Always `true` for a rank-0 shape, which holds its one element at
    /// index `[]`.
    #[inline(always)]
    pub fn contains(&self, index: [usize; N]) -> bool {
        index.iter().zip(&self.extents).all(|(i, e)| i < e)
    }

    /// Flat offset of `index` — `Σ index[d]·strides[d]`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds on any axis.
    #[inline(always)]
    pub fn offset(&self, index: [usize; N]) -> usize {
        match self.offset_checked(index) {
            Some(offset) => offset,
            None => panic!(
                "index {index:?} out of bounds for extents {:?}",
                self.extents,
            ),
        }
    }

    /// Flat offset of `index`, or `None` if it is out of bounds.
    #[inline(always)]
    pub fn offset_checked(&self, index: [usize; N]) -> Option<usize> {
        self.contains(index)
            .then(|| index.iter().zip(&self.strides).map(|(i, s)| i * s).sum())
    }

    /// Every element's flat offset, in row-major order.
    ///
    /// Two shapes with equal extents visit their elements in the same order, so
    /// `a.offsets().zip(b.offsets())` walks a pair of layouts element-wise
    /// whatever their strides.
    #[inline(always)]
    pub fn offsets(&self) -> Offsets<N> {
        Offsets {
            extents: self.extents,
            strides: self.strides,
            index: [0; N],
            offset: 0,
            remaining: self.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Packed blocks
// ---------------------------------------------------------------------------

/// Flat addressing of a **packed sequence of blocks**, each one element of this
/// shape laid out row-major back to back — the layout a [`Series`] stores its
/// elements in, and the one [`stacked`](Self::stacked) describes as a single
/// rank-`N + 1` array.
impl<const N: usize> Shape<N> {
    /// Scalar count of `n` packed blocks — `n · len()`.
    #[inline(always)]
    pub fn blocks_len(&self, n: usize) -> usize {
        n * self.len()
    }

    /// Flat range of block `i` — `i · len() .. (i + 1) · len()`.
    #[inline(always)]
    pub fn block_range(&self, i: usize) -> Range<usize> {
        self.blocks_range(i..i + 1)
    }

    /// Flat range of blocks `range` — `range.start · len() .. range.end · len()`.
    #[inline(always)]
    pub fn blocks_range(&self, range: Range<usize>) -> Range<usize> {
        self.blocks_len(range.start)..self.blocks_len(range.end)
    }
}

// ---------------------------------------------------------------------------
// Rank changes
// ---------------------------------------------------------------------------

impl<const N: usize> Shape<N> {
    /// The rank-`M = N + 1` row-major shape of `n` packed blocks of this shape:
    /// extents `[n, extents…]`. The inverse of [`split_first`](Self::split_first)
    /// for a contiguous shape.
    ///
    /// (`M` is spelled explicitly because stable Rust cannot form `N + 1` in a
    /// type; the relation is asserted at runtime.)
    ///
    /// # Panics
    ///
    /// Panics if `M != N + 1`. The blocks are assumed packed — i.e. this shape
    /// is [contiguous](Self::is_contiguous), checked on debug builds.
    pub fn stacked<const M: usize>(&self, n: usize) -> Shape<M> {
        assert_eq!(M, N + 1, "stacked: M ({M}) must be N + 1 ({})", N + 1);
        debug_assert!(self.is_contiguous(), "stacked: blocks must be contiguous");
        let mut extents = [0usize; M];
        extents[0] = n;
        extents[1..].copy_from_slice(&self.extents);
        Shape::row_major(extents)
    }

    /// Split axis 0 off: the flat step between consecutive slices along it
    /// (`strides[0]`), and the rank-`M = N - 1` shape of one such slice (the
    /// inner axes' extents and strides, unchanged — so slice `i` of a view
    /// starts at flat offset `i · step`).
    ///
    /// # Panics
    ///
    /// Panics if `N == 0` or `M != N - 1`.
    pub fn split_first<const M: usize>(&self) -> (usize, Shape<M>) {
        assert!(N >= 1, "split_first: rank N must be >= 1, got 0");
        assert_eq!(M, N - 1, "split_first: M ({M}) must be N - 1 ({})", N - 1);
        let mut extents = [0usize; M];
        let mut strides = [0usize; M];
        extents.copy_from_slice(&self.extents[1..]);
        strides.copy_from_slice(&self.strides[1..]);
        (self.strides[0], Shape { extents, strides })
    }
}

// ===========================================================================
// Offsets — row-major traversal of a strided layout
// ===========================================================================

/// Iterator over every element's flat offset in a [`Shape`], in row-major
/// order; created by [`Shape::offsets`].
///
/// Each step carries a mixed-radix logical index and keeps the running offset
/// in step with it, so a step costs `O(1)` amortized rather than an `O(N)`
/// recompute of `Σ index[d]·strides[d]`.
#[derive(Clone, Debug)]
pub struct Offsets<const N: usize> {
    extents: [usize; N],
    strides: [usize; N],
    index: [usize; N],
    offset: usize,
    remaining: usize,
}

impl<const N: usize> Iterator for Offsets<N> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.remaining = self.remaining.checked_sub(1)?;
        let offset = self.offset;
        // Increment the logical index from the fastest-varying axis, undoing
        // each axis that wraps; `offset` follows the same carry.
        let mut d = N;
        while d > 0 {
            d -= 1;
            self.index[d] += 1;
            self.offset += self.strides[d];
            if self.index[d] < self.extents[d] {
                break;
            }
            self.index[d] = 0;
            self.offset -= self.extents[d] * self.strides[d];
        }
        Some(offset)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<const N: usize> ExactSizeIterator for Offsets<N> {}
impl<const N: usize> std::iter::FusedIterator for Offsets<N> {}
