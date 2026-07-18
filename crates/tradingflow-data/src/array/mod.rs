//! Dense array with a **compile-time rank** `N` and a strided
//! (`std::mdspan`-like) borrowed view — the unified currency of the operator
//! interface.
//!
//! * [`Array<T, N>`] — owned, row-major **contiguous** backing store; lives in
//!   operator `State`.
//! * [`ArrayView<'a, T, N>`] ≈ `mdspan<T, dextents<usize, N>, layout_stride>`:
//!   `&[T]` (from the view's origin) + a [`Shape`] (per-axis extents and
//!   strides), all inline.
//!   It is `Copy`, `Send`, `Sync`, and self-contained — there is **no**
//!   by-reference shape, so view-emitting operators need neither in-state shape
//!   storage nor a per-generation arena.
//!
//! The rank `N` is static (known at compile time); the extents and strides are
//! dynamic (runtime). A zero-copy slice of an array (a column, a squeezed axis)
//! is just another `ArrayView` with an origin-advanced `&[T]` and adjusted
//! `strides` over the same buffer; the elementwise core
//! ([`apply_unary`]/[`apply_binary`]) has a contiguous fast path and a strided
//! fallback, so a strided view feeds directly into the next operator with no
//! materialization.
//!
//! # Views vs flat data
//!
//! Both types address scalars per-axis — `a[[i, j]]`, `a[[]]` for rank 0 —
//! resolving the index through [`Shape::offset`], so a strided view reads the
//! same scalars as its parent. [`Array::assign`] likewise takes a view.
//!
//! The methods that name `data` are the flat row-major escape hatches:
//! [`Array::data`]/[`Array::data_mut`] over the owned buffer, and
//! [`ArrayView::data`] over the backing slice from the view's origin. A view is
//! only guaranteed flat when it is contiguous, so materializing one goes
//! through [`as_slice`](ArrayView::as_slice) (borrow iff already contiguous),
//! [`to_contiguous`](ArrayView::to_contiguous) (borrow or copy) or
//! [`to_vec`](ArrayView::to_vec) (always copy).

use crate::{Scalar, Shape};

mod iter;
mod ops;
mod owned;
mod view;

pub use iter::{ArrayIntoIter, ArrayIter};
pub(crate) use ops::write_row_major;
pub use ops::{apply_binary, apply_unary};

// ===========================================================================
// Array<T, N> — owned, row-major contiguous backing store
// ===========================================================================

/// An owned, row-major contiguous, rank-`N` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array<T: Scalar, const N: usize> {
    data: Box<[T]>,
    shape: Shape<N>,
}

// ===========================================================================
// ArrayView<'a, T, N> — borrowed, strided, Copy, self-contained
// ===========================================================================

/// A borrowed, strided (`mdspan`-like), rank-`N` view: the zero-copy edge
/// currency of the operator interface.
///
/// It holds a backing slice (`&[T]`) that **starts at the view's origin**
/// element, plus a [`Shape`]; `strides` address into that slice (the
/// `[0, …, 0]` element is `data[0]`), so a column or a squeezed axis is a view
/// over the **same** buffer (advanced to its origin) with no copy. `Copy` (the
/// payload is references + plain `usize`s) and fully lifetime-checked — a view
/// cannot outlive the array it borrows from. The engine's per-generation
/// contract keeps wire-borne views valid between recomputes.
#[derive(Debug)]
pub struct ArrayView<'a, T: Scalar, const N: usize> {
    /// The backing buffer from the view's origin; `strides` address into it.
    data: &'a [T],
    shape: Shape<N>,
}

// Manual (not derived) `Clone`/`Copy`: the view is references + plain `usize`s,
// so it is `Copy` regardless of whether `T` is. (Derived `Copy` would wrongly
// demand `T: Copy`, which e.g. `String` does not satisfy.)
impl<T: Scalar, const N: usize> Clone for ArrayView<'_, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Scalar, const N: usize> Copy for ArrayView<'_, T, N> {}
