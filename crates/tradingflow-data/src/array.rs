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

use std::borrow::Cow;
use std::ops;

use super::{Scalar, Shape};

// ===========================================================================
// Array<T, N> — owned, row-major contiguous backing store
// ===========================================================================

/// An owned, row-major contiguous, rank-`N` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array<T: Scalar, const N: usize> {
    data: Box<[T]>,
    shape: Shape<N>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<T: Scalar> Array<T, 0> {
    /// Create a rank-0 array holding one scalar.
    pub fn scalar(value: T) -> Self {
        Self {
            data: vec![value].into(),
            shape: Shape::row_major([]),
        }
    }
}

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Create an array filled with `value`.
    pub fn full(extents: [usize; N], value: T) -> Self {
        let shape = Shape::row_major(extents);
        Self {
            data: vec![value; shape.len()].into(),
            shape,
        }
    }

    /// Create an array filled with `T::default()` (0 for numeric types).
    pub fn zeros(extents: [usize; N]) -> Self {
        Self::full(extents, T::default())
    }

    /// Create an array from extents and a flat row-major buffer — the owning
    /// counterpart of [`ArrayView::from_slice`].
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != extents.iter().product()`.
    pub fn from_vec(extents: [usize; N], data: Vec<T>) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(
            data.len(),
            shape.len(),
            "from_vec: extents {:?} expect {} scalars, got {}",
            extents,
            shape.len(),
            data.len(),
        );
        Self {
            data: data.into(),
            shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Array<T, N> {
    /// The shape (per-axis extents and strides; always canonical row-major).
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis extents.
    pub fn extents(&self) -> [usize; N] {
        self.shape.extents()
    }

    /// Number of scalars (product of extents).
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    /// Whether there are no scalars (some extent is zero).
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Bulk access
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Flat immutable slice of all scalars (row-major).
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Flat mutable slice of all scalars (row-major).
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow the whole array as a contiguous [`ArrayView`].
    pub fn view(&self) -> ArrayView<'_, T, N> {
        ArrayView {
            data: &self.data,
            shape: self.shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

/// Index by a per-axis logical index — `a[[i, j]]` for a rank-2 array, `a[[]]`
/// for a rank-0 one. [`data`](Array::data) is the flat row-major escape hatch.
///
/// # Panics
///
/// Panics if the index is out of bounds on any axis.
impl<T: Scalar, const N: usize> ops::Index<[usize; N]> for Array<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        &self.data[self.shape.offset(index)]
    }
}

impl<T: Scalar, const N: usize> ops::IndexMut<[usize; N]> for Array<T, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        &mut self.data[self.shape.offset(index)]
    }
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Copy in the scalars of a rank-`N` view, which may be strided (it is
    /// materialized row-major). [`data_mut`](Self::data_mut) is the flat
    /// counterpart.
    ///
    /// # Panics
    ///
    /// Panics if `value.extents() != self.extents()`.
    pub fn assign(&mut self, value: ArrayView<'_, T, N>) {
        assert_eq!(value.extents(), self.extents(), "assign: extents mismatch");
        write_row_major(&mut self.data, value);
    }

    /// Change the extents in place (same rank), without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different scalar count.
    pub fn reshape(&mut self, extents: [usize; N]) {
        let shape = Shape::row_major(extents);
        assert_eq!(
            self.len(),
            shape.len(),
            "reshape: current len {} != new extents {:?} ({} scalars)",
            self.len(),
            extents,
            shape.len(),
        );
        self.shape = shape;
    }
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

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// View extents and a flat row-major buffer as a contiguous array — the
    /// borrowing counterpart of [`Array::from_vec`].
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != extents.iter().product()`.
    pub fn from_slice(extents: [usize; N], data: &'a [T]) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(
            data.len(),
            shape.len(),
            "from_slice: extents {:?} expect {} scalars, got {}",
            extents,
            shape.len(),
            data.len(),
        );
        Self { data, shape }
    }

    /// Build a strided view from a [`Shape`] and a backing slice whose **first
    /// element is the view's origin** (`[0, …, 0]`).
    ///
    /// # Panics
    ///
    /// Panics if `data` is too short to contain every scalar the shape
    /// addresses (`data.len() < shape.span()`).
    pub fn from_parts(shape: Shape<N>, data: &'a [T]) -> Self {
        assert!(
            data.len() >= shape.span(),
            "from_parts: shape spans {} scalars, got {}",
            shape.span(),
            data.len(),
        );
        Self { data, shape }
    }
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> ArrayView<'_, T, N> {
    /// The shape (per-axis extents and strides; possibly non-canonical).
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis extents.
    pub fn extents(&self) -> [usize; N] {
        self.shape.extents()
    }

    /// Number of scalars (product of extents).
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    /// Whether there are no scalars (some extent is zero).
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Bulk access
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// The backing slice from the view's origin — for callers that walk the
    /// view with explicit stride arithmetic (index `0` is the `[0, …, 0]`
    /// element).
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// The contiguous fast path: `Some(flat slice)` iff the view has canonical
    /// row-major strides. `None` for a strided view (e.g. a column).
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.shape.is_contiguous() {
            Some(&self.data[..self.shape.len()])
        } else {
            None
        }
    }

    /// Borrow the view's scalars as a contiguous flat slice, materializing into
    /// an owned buffer (row-major) only when the view is strided. Zero-copy for
    /// the common contiguous case.
    pub fn to_contiguous(&self) -> Cow<'a, [T]> {
        match self.as_slice() {
            Some(s) => Cow::Borrowed(s),
            None => Cow::Owned(self.to_vec()),
        }
    }

    /// Materialize the view into a fresh row-major `Vec<T>`.
    pub fn to_vec(&self) -> Vec<T> {
        match self.as_slice() {
            Some(s) => s.to_vec(),
            None => self
                .shape
                .offsets()
                .map(|off| self.data[off].clone())
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

/// Index by a per-axis logical index, resolved through the view's strides —
/// `v[[i, j]]` is the same scalar as in the parent array, with no copy.
///
/// # Panics
///
/// Panics if the index is out of bounds on any axis.
impl<T: Scalar, const N: usize> ops::Index<[usize; N]> for ArrayView<'_, T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        &self.data[self.shape.offset(index)]
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> ArrayView<'_, T, N> {
    /// Copy the view into an owned, contiguous [`Array`].
    pub fn to_array(&self) -> Array<T, N> {
        Array::from_vec(self.shape.extents(), self.to_vec())
    }
}

impl<'a, T: Scalar, const N: usize> From<&'a Array<T, N>> for ArrayView<'a, T, N> {
    fn from(a: &'a Array<T, N>) -> Self {
        a.view()
    }
}

impl<T: Scalar, const N: usize> From<ArrayView<'_, T, N>> for Array<T, N> {
    fn from(v: ArrayView<'_, T, N>) -> Self {
        v.to_array()
    }
}

// ===========================================================================
// Elementwise core
//
// Traversal order comes from `Shape::offsets`; these add only the fast path for
// layouts that need no stride arithmetic at all. `pub` because the operator
// library (`tradingflow`'s `arith` module) is the caller: the elementwise
// kernels live with the array they iterate.
// ===========================================================================

/// Element-wise unary `out[i] = f(x[i])`, contiguous-fast / strided-slow. The
/// output scalar `U` may differ from the input's (a predicate maps `T` to
/// `bool`); `U == T` for the arithmetic operators.
pub fn apply_unary<T: Scalar, U: Scalar, const N: usize>(
    out: &mut Array<U, N>,
    x: ArrayView<T, N>,
    f: impl Fn(T) -> U,
) {
    let o = out.data_mut();
    if let Some(s) = x.as_slice() {
        for (dst, v) in o.iter_mut().zip(s) {
            *dst = f(v.clone());
        }
        return;
    }
    for (dst, off) in o.iter_mut().zip(x.shape.offsets()) {
        *dst = f(x.data[off].clone());
    }
}

/// Element-wise binary `out[i] = f(a[i], b[i])`, contiguous-fast / strided-slow.
/// `a` and `b` must share extents (asserted by the caller via output sizing).
/// As with [`apply_unary`], the output scalar `U` may differ from the inputs'.
pub fn apply_binary<T: Scalar, U: Scalar, const N: usize>(
    out: &mut Array<U, N>,
    a: ArrayView<T, N>,
    b: ArrayView<T, N>,
    f: impl Fn(T, T) -> U,
) {
    let o = out.data_mut();
    if let (Some(sa), Some(sb)) = (a.as_slice(), b.as_slice()) {
        for (dst, (va, vb)) in o.iter_mut().zip(sa.iter().zip(sb)) {
            *dst = f(va.clone(), vb.clone());
        }
        return;
    }
    // Strided: `a` and `b` share extents, so their offsets pair up row-major.
    for (dst, (oa, ob)) in o.iter_mut().zip(a.shape.offsets().zip(b.shape.offsets())) {
        *dst = f(a.data[oa].clone(), b.data[ob].clone());
    }
}

/// Copy a (possibly strided) view into a contiguous destination in row-major
/// order — the materialization primitive behind [`Array::assign`] and the
/// rank-changers.
pub(crate) fn write_row_major<T: Scalar, const N: usize>(dst: &mut [T], v: ArrayView<T, N>) {
    if let Some(s) = v.as_slice() {
        dst.clone_from_slice(s);
        return;
    }
    for (d, off) in dst.iter_mut().zip(v.shape.offsets()) {
        *d = v.data[off].clone();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_zeros() {
        let a = Array::full([2, 3], 1.0_f64);
        assert_eq!(a.extents(), [2, 3]);
        assert_eq!(a.len(), 6);
        assert_eq!(a.data(), &[1.0; 6]);

        let b = Array::<f64, 1>::zeros([4]);
        assert_eq!(b.data(), &[0.0; 4]);
    }

    #[test]
    fn scalar() {
        let a = Array::scalar(42.0_f64);
        assert_eq!(a.extents(), [] as [usize; 0]);
        assert_eq!(a.len(), 1);
        // A rank-0 array holds its one scalar at the empty index.
        assert_eq!(a[[]], 42.0);
    }

    #[test]
    fn from_slice_matches_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = ArrayView::from_slice([2, 3], &data);
        assert_eq!(v.extents(), [2, 3]);
        assert!(v.shape().is_contiguous());
        assert_eq!(v.to_array(), Array::from_vec([2, 3], data.clone()));
    }

    #[test]
    #[should_panic(expected = "from_slice: extents [2, 3] expect 6 scalars, got 5")]
    fn from_slice_wrong_len() {
        let _ = ArrayView::<f64, 2>::from_slice([2, 3], &[0.0; 5]);
    }

    #[test]
    #[should_panic(expected = "from_parts: shape spans 5 scalars, got 4")]
    fn from_parts_data_too_short() {
        // A [3]-extent, stride-2 column addresses offsets {0, 2, 4}.
        let _ = ArrayView::<f64, 1>::from_parts(Shape::strided([3], [2]), &[0.0; 4]);
    }

    #[test]
    fn assign_and_index_mut() {
        let mut a = Array::<f64, 1>::zeros([3]);
        let b = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
        a.assign(b.view());
        assert_eq!(a.data(), &[1.0, 2.0, 3.0]);
        a[[1]] = 20.0;
        assert_eq!(a.data(), &[1.0, 20.0, 3.0]);
    }

    #[test]
    fn assign_materializes_a_strided_view() {
        let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Column 1: extent 3, stride 2, from index 1.
        let col1 = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
        let mut a = Array::<f64, 1>::zeros([3]);
        a.assign(col1);
        assert_eq!(a.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "assign: extents mismatch")]
    fn assign_wrong_extents() {
        let mut a = Array::<f64, 1>::zeros([3]);
        let b = Array::<f64, 1>::zeros([2]);
        a.assign(b.view());
    }

    #[test]
    fn index_is_per_axis() {
        let mut a = Array::from_vec([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(a[[0, 0]], 0.0);
        assert_eq!(a[[0, 2]], 2.0);
        assert_eq!(a[[1, 0]], 3.0);
        assert_eq!(a[[1, 2]], 5.0);
        a[[1, 1]] = 40.0;
        assert_eq!(a.data(), &[0.0, 1.0, 2.0, 3.0, 40.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "index [0, 3] out of bounds for extents [2, 3]")]
    fn index_out_of_bounds_per_axis() {
        // Flat offset 3 is inside the buffer, but [0, 3] is off the end of a row.
        let a = Array::<f64, 2>::zeros([2, 3]);
        let _ = a[[0, 3]];
    }

    #[test]
    fn view_index_resolves_strides() {
        let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(panel.view()[[2, 0]], 5.0);
        // Column 1: extent 3, stride 2, from index 1.
        let col1 = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
        assert_eq!(col1[[0]], 2.0);
        assert_eq!(col1[[1]], 4.0);
        assert_eq!(col1[[2]], 6.0);
    }

    #[test]
    fn reshape() {
        // Same-rank reshape (rank is compile-time fixed at `N`).
        let mut a = Array::from_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.reshape([3, 2]);
        assert_eq!(a.extents(), [3, 2]);
        assert_eq!(a.len(), 6);
    }

    #[test]
    #[should_panic(expected = "reshape")]
    fn reshape_wrong_size() {
        let mut a = Array::<f64, 2>::zeros([2, 3]);
        a.reshape([2, 2]);
    }

    #[test]
    fn view_is_copy_and_inline() {
        use std::mem::size_of;
        let word = size_of::<usize>();
        let fatptr = size_of::<&[f64]>(); // 2 words
        // data(&[T]) + extents[N] + strides[N] — all inline (no offset).
        assert_eq!(size_of::<ArrayView<f64, 1>>(), fatptr + word * 2);
        assert_eq!(size_of::<ArrayView<f64, 2>>(), fatptr + word * 4);
        fn assert_copy<T: Copy>() {}
        assert_copy::<ArrayView<f64, 3>>();
    }

    #[test]
    fn as_slice_and_strided_column() {
        let panel = Array::from_vec(
            [3, 4],
            vec![
                0.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0,
            ],
        );
        assert!(panel.view().as_slice().is_some());

        // Column 1: extent 3, stride 4, from index 1 — strided.
        let col1 = ArrayView::from_parts(Shape::strided([3], [4]), &panel.data()[1..]);
        assert!(col1.as_slice().is_none());
        assert_eq!(col1.to_vec(), vec![1.0, 5.0, 9.0]);
        assert_eq!(&*col1.to_contiguous(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn unary_contiguous_and_strided_agree() {
        let x = Array::from_vec([3], vec![1.0, 4.0, 9.0]);
        let mut out = Array::<f64, 1>::zeros([3]);
        apply_unary(&mut out, x.view(), |v: f64| v.sqrt());
        assert_eq!(out.data(), &[1.0, 2.0, 3.0]);

        let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let col1 = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
        let mut out = Array::<f64, 1>::zeros([3]);
        apply_unary(&mut out, col1, |v: f64| v * 10.0);
        assert_eq!(out.data(), &[20.0, 40.0, 60.0]);
    }

    #[test]
    fn binary_mixed_contiguous_and_strided() {
        let a = Array::from_vec([3], vec![100.0, 200.0, 300.0]);
        let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bcol = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
        let mut out = Array::<f64, 1>::zeros([3]);
        apply_binary(&mut out, a.view(), bcol, |x, y| x + y);
        assert_eq!(out.data(), &[102.0, 204.0, 306.0]);
    }

    #[test]
    fn write_row_major_squeezes_strided() {
        let panel = Array::from_vec([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // Column 2: extent 2, stride 3, from index 2 -> [2.0, 5.0].
        let col2 = ArrayView::from_parts(Shape::strided([2], [3]), &panel.data()[2..]);
        let mut dst = [0.0; 2];
        write_row_major(&mut dst, col2);
        assert_eq!(dst, [2.0, 5.0]);
    }

    #[test]
    fn partial_eq_includes_shape() {
        let a = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
        let c = Array::from_vec([3], vec![1.0, 2.0, 4.0]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
