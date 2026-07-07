//! Dense array with a **compile-time rank** `N` and a strided
//! (`std::mdspan`-like) borrowed view — the unified currency of the operator
//! interface.
//!
//! * [`Array<T, N>`] — owned, row-major **contiguous** backing store; lives in
//!   operator `State`.
//! * [`ArrayView<'a, T, N>`] ≈ `mdspan<T, dextents<usize, N>, layout_stride>`:
//!   `&[T]` + `offset` + per-axis `extents` + per-axis `strides`, all inline.
//!   It is `Copy`, `Send`, `Sync`, and self-contained — there is **no**
//!   by-reference shape, so view-emitting operators need neither in-state shape
//!   storage nor a per-generation arena.
//!
//! The rank `N` is static (known at compile time); the extents and strides are
//! dynamic (runtime). A zero-copy slice of an array (a column, a squeezed axis)
//! is just another `ArrayView` with adjusted `offset`/`strides` over the same
//! buffer; the elementwise core ([`apply_unary`]/[`apply_binary`]) has a
//! contiguous fast path and a strided fallback, so a strided view feeds directly
//! into the next operator with no materialization.

use std::borrow::Cow;
use std::ops;

use super::Scalar;

// ===========================================================================
// Shape — inline extents + strides (the mdspan `layout_stride` mapping)
// ===========================================================================

/// Per-axis extents and strides for a rank-`N` array, all inline (`Copy`).
///
/// `strides[d]` is the flat-buffer step for one increment along axis `d`. A
/// [`row_major`](Self::row_major) shape is C-contiguous; a sliced view carries
/// non-canonical strides over the parent's buffer.
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
}

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
    /// Create a rank-0 array holding one element.
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

    /// Create an array from a flat row-major buffer and extents.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != extents.iter().product()`.
    pub fn from_vec(extents: [usize; N], data: Vec<T>) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(
            data.len(),
            shape.len(),
            "from_vec: extents {:?} expect {} elements, got {}",
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
    /// The array shape (extents + canonical strides).
    #[inline(always)]
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis extents.
    #[inline(always)]
    pub fn extents(&self) -> [usize; N] {
        self.shape.extents
    }

    /// Number of dimensions (the compile-time rank `N`).
    #[inline(always)]
    pub const fn ndim(&self) -> usize {
        N
    }

    /// Number of scalars (product of extents).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array holds no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Bulk access
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Flat immutable slice of all elements (row-major).
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Flat mutable slice of all elements (row-major).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow the whole array as a contiguous [`ArrayView`].
    #[inline(always)]
    pub fn view(&self) -> ArrayView<'_, T, N> {
        ArrayView {
            data: &self.data,
            offset: 0,
            shape: self.shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Indexing (flat)
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> ops::Index<usize> for Array<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: Scalar, const N: usize> ops::IndexMut<usize> for Array<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

// ---------------------------------------------------------------------------
// Assignment / reshape
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Copy data from a flat slice.
    ///
    /// # Panics
    ///
    /// Panics if `value.len() != self.len()`.
    #[inline(always)]
    pub fn assign(&mut self, value: &[T]) {
        assert_eq!(
            value.len(),
            self.len(),
            "assign: expected {} scalars, got {}",
            self.len(),
            value.len(),
        );
        self.data.clone_from_slice(value);
    }

    /// Change the extents in place (same rank), without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different element count.
    pub fn reshape(&mut self, extents: [usize; N]) {
        let shape = Shape::row_major(extents);
        assert_eq!(
            self.len(),
            shape.len(),
            "reshape: current len {} != new extents {:?} ({} elements)",
            self.len(),
            extents,
            shape.len(),
        );
        self.shape = shape;
    }
}

// ===========================================================================
// ArrayView<'a, T, N> — strided, Copy, self-contained
// ===========================================================================

/// A borrowed, strided (`mdspan`-like), rank-`N` view: the zero-copy edge
/// currency of the operator interface.
///
/// It holds the whole backing buffer (`&[T]`) plus an `offset` and a
/// [`Shape`]; `offset` + `strides` address into the buffer, so a column or a
/// squeezed axis is a view over the **same** buffer with no copy. `Copy` (the
/// payload is references + plain `usize`s) and fully lifetime-checked — a view
/// cannot outlive the array it borrows from. The engine's per-generation
/// contract keeps wire-borne views valid between recomputes.
#[derive(Debug)]
pub struct ArrayView<'a, T: Scalar, const N: usize> {
    /// The whole backing buffer; `offset` + `strides` address into it.
    data: &'a [T],
    offset: usize,
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

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// View a flat buffer as a row-major contiguous array of `extents`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != extents.iter().product()`.
    pub fn contiguous(data: &'a [T], extents: [usize; N]) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(
            data.len(),
            shape.len(),
            "ArrayView::contiguous: extents {:?} expect {} elements, got {}",
            extents,
            shape.len(),
            data.len(),
        );
        Self {
            data,
            offset: 0,
            shape,
        }
    }

    /// Build a strided view directly from parts: the backing buffer, a flat
    /// `offset`, and a [`Shape`]. The caller guarantees every addressed element
    /// `offset + Σ idx[d]·strides[d]` lies in `data`.
    #[inline(always)]
    pub fn from_parts(data: &'a [T], offset: usize, shape: Shape<N>) -> Self {
        Self {
            data,
            offset,
            shape,
        }
    }

    /// The view shape (extents + strides).
    #[inline(always)]
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis extents.
    #[inline(always)]
    pub fn extents(&self) -> [usize; N] {
        self.shape.extents
    }

    /// Number of scalars (product of extents).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    /// Whether the view holds no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// The flat backing buffer plus the addressing `offset` — for callers that
    /// walk the view with explicit stride arithmetic.
    #[inline(always)]
    pub fn buffer(&self) -> (&'a [T], usize) {
        (self.data, self.offset)
    }

    /// The contiguous fast path: `Some(flat slice)` iff the view is exhaustive
    /// (canonical row-major strides). `None` for a strided view (e.g. a column).
    #[inline(always)]
    pub fn contiguous_slice(&self) -> Option<&'a [T]> {
        if self.shape.is_contiguous() {
            Some(&self.data[self.offset..self.offset + self.shape.len()])
        } else {
            None
        }
    }

    /// Borrow the view's elements as a contiguous flat slice, materializing into
    /// an owned buffer (row-major) only when the view is strided. Zero-copy for
    /// the common contiguous case.
    pub fn to_contiguous(&self) -> Cow<'a, [T]> {
        match self.contiguous_slice() {
            Some(s) => Cow::Borrowed(s),
            None => Cow::Owned(self.to_vec()),
        }
    }

    /// Materialize the view into a fresh row-major `Vec<T>`.
    pub fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.shape.len());
        for_each_offset(&self.shape, self.offset, |off| {
            out.push(self.data[off].clone())
        });
        out
    }

    /// Copy the view into an owned, contiguous [`Array`].
    pub fn to_array(&self) -> Array<T, N> {
        Array::from_vec(self.shape.extents, self.to_vec())
    }
}

impl<'a, T: Scalar, const N: usize> From<&'a Array<T, N>> for ArrayView<'a, T, N> {
    #[inline(always)]
    fn from(a: &'a Array<T, N>) -> Self {
        a.view()
    }
}

// ===========================================================================
// Strided traversal + elementwise core
//
// [array-view-refactor] `apply_unary`/`apply_binary`/`write_row_major` are the
// shared elementwise/materialization primitives the operator layer is being
// migrated onto (stages 3–4); `allow(dead_code)` until the operators module is
// un-gated.
// ===========================================================================

/// Advance a mixed-radix logical index one step in row-major order, tracking the
/// running flat offset incrementally (no per-element `O(N)` recompute).
#[inline]
fn advance<const N: usize>(
    idx: &mut [usize; N],
    ext: &[usize; N],
    strd: &[usize; N],
    off: &mut usize,
) {
    let mut d = N;
    while d > 0 {
        d -= 1;
        idx[d] += 1;
        *off += strd[d];
        if idx[d] < ext[d] {
            return;
        }
        idx[d] = 0;
        *off -= ext[d] * strd[d];
    }
}

/// Visit every element's flat offset in row-major order, calling `f(offset)`.
#[inline]
fn for_each_offset<const N: usize>(shape: &Shape<N>, base: usize, mut f: impl FnMut(usize)) {
    let total = shape.len();
    if total == 0 {
        return;
    }
    let (ext, strd) = (shape.extents, shape.strides);
    let mut idx = [0usize; N];
    let mut off = base;
    for _ in 0..total {
        f(off);
        advance(&mut idx, &ext, &strd, &mut off);
    }
}

/// Element-wise unary `out[i] = f(x[i])`, contiguous-fast / strided-slow.
pub(crate) fn apply_unary<T: Scalar, const N: usize>(
    out: &mut Array<T, N>,
    x: &ArrayView<T, N>,
    f: &impl Fn(T) -> T,
) {
    let o = out.as_mut_slice();
    if let Some(s) = x.contiguous_slice() {
        for (dst, v) in o.iter_mut().zip(s) {
            *dst = f(v.clone());
        }
        return;
    }
    let data = x.data;
    let mut i = 0usize;
    for_each_offset(&x.shape, x.offset, |off| {
        o[i] = f(data[off].clone());
        i += 1;
    });
}

/// Element-wise binary `out[i] = f(a[i], b[i])`, contiguous-fast / strided-slow.
/// `a` and `b` must share extents (asserted by the caller via output sizing).
pub(crate) fn apply_binary<T: Scalar, const N: usize>(
    out: &mut Array<T, N>,
    a: &ArrayView<T, N>,
    b: &ArrayView<T, N>,
    f: &impl Fn(T, T) -> T,
) {
    let o = out.as_mut_slice();
    if let (Some(sa), Some(sb)) = (a.contiguous_slice(), b.contiguous_slice()) {
        for (dst, (va, vb)) in o.iter_mut().zip(sa.iter().zip(sb)) {
            *dst = f(va.clone(), vb.clone());
        }
        return;
    }
    // Strided: `a` and `b` share extents; each tracks its own offset.
    let ext = a.shape.extents;
    let (stra, strb) = (a.shape.strides, b.shape.strides);
    let mut idx = [0usize; N];
    let (mut oa, mut ob) = (a.offset, b.offset);
    for dst in o.iter_mut() {
        *dst = f(a.data[oa].clone(), b.data[ob].clone());
        let mut d = N;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            oa += stra[d];
            ob += strb[d];
            if idx[d] < ext[d] {
                break;
            }
            idx[d] = 0;
            oa -= ext[d] * stra[d];
            ob -= ext[d] * strb[d];
        }
    }
}

/// Copy a (possibly strided) view into a contiguous destination in row-major
/// order — the materialization primitive the rank-changers use.
#[allow(dead_code)]
pub(crate) fn write_row_major<T: Scalar, const N: usize>(dst: &mut [T], v: &ArrayView<T, N>) {
    if let Some(s) = v.contiguous_slice() {
        dst.clone_from_slice(s);
        return;
    }
    let data = v.data;
    let mut i = 0usize;
    for_each_offset(&v.shape, v.offset, |off| {
        dst[i] = data[off].clone();
        i += 1;
    });
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
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.as_slice(), &[1.0; 6]);

        let b = Array::<f64, 1>::zeros([4]);
        assert_eq!(b.as_slice(), &[0.0; 4]);
    }

    #[test]
    fn scalar() {
        let a = Array::scalar(42.0_f64);
        assert_eq!(a.extents(), [] as [usize; 0]);
        assert_eq!(a.len(), 1);
        assert_eq!(a.ndim(), 0);
        assert_eq!(a[0], 42.0);
    }

    #[test]
    fn assign_and_index_mut() {
        let mut a = Array::<f64, 1>::zeros([3]);
        let b = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
        a.assign(b.as_slice());
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0]);
        a[1] = 20.0;
        assert_eq!(a.as_slice(), &[1.0, 20.0, 3.0]);
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
        // data(&[T]) + offset(usize) + extents[N] + strides[N] — all inline.
        assert_eq!(size_of::<ArrayView<f64, 1>>(), fatptr + word * (1 + 2));
        assert_eq!(size_of::<ArrayView<f64, 2>>(), fatptr + word * (1 + 4));
        fn assert_copy<T: Copy>() {}
        assert_copy::<ArrayView<f64, 3>>();
    }

    #[test]
    fn contiguous_slice_and_strided_column() {
        let panel = Array::from_vec(
            [3, 4],
            vec![
                0.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0,
            ],
        );
        assert!(panel.view().contiguous_slice().is_some());

        // Column 1: extent 3, stride 4, offset 1 — strided.
        let col1 = ArrayView::from_parts(panel.as_slice(), 1, Shape::strided([3], [4]));
        assert!(col1.contiguous_slice().is_none());
        assert_eq!(col1.to_vec(), vec![1.0, 5.0, 9.0]);
        assert_eq!(&*col1.to_contiguous(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn unary_contiguous_and_strided_agree() {
        let x = Array::from_vec([3], vec![1.0, 4.0, 9.0]);
        let mut out = Array::<f64, 1>::zeros([3]);
        apply_unary(&mut out, &x.view(), &|v: f64| v.sqrt());
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0]);

        let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let col1 = ArrayView::from_parts(panel.as_slice(), 1, Shape::strided([3], [2]));
        let mut out = Array::<f64, 1>::zeros([3]);
        apply_unary(&mut out, &col1, &|v: f64| v * 10.0);
        assert_eq!(out.as_slice(), &[20.0, 40.0, 60.0]);
    }

    #[test]
    fn binary_mixed_contiguous_and_strided() {
        let a = Array::from_vec([3], vec![100.0, 200.0, 300.0]);
        let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bcol = ArrayView::from_parts(panel.as_slice(), 1, Shape::strided([3], [2]));
        let mut out = Array::<f64, 1>::zeros([3]);
        apply_binary(&mut out, &a.view(), &bcol, &|x, y| x + y);
        assert_eq!(out.as_slice(), &[102.0, 204.0, 306.0]);
    }

    #[test]
    fn write_row_major_squeezes_strided() {
        let panel = Array::from_vec([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // Column 2: extent 2, stride 3, offset 2 -> [2.0, 5.0].
        let col2 = ArrayView::from_parts(panel.as_slice(), 2, Shape::strided([2], [3]));
        let mut dst = [0.0; 2];
        write_row_major(&mut dst, &col2);
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
