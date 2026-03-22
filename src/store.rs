//! Unified time-series storage with cheap view types.
//!
//! [`Store<T>`] holds a contiguous scalar buffer and a timestamp buffer.
//! A configurable window size controls whether the store retains full
//! history (`window = 0`) or only the most recent N elements.
//!
//! # Window sizes
//!
//! * `window = 0` — unbounded: the store grows without limit (full history).
//! * `window = 1` — single element: [`push_default`] overwrites in place,
//!   then [`element_view_mut`] gives the operator a mutable view to write.
//! * `window = N` — fixed sliding window: keeps the most recent N elements.
//!
//! `start` tracks dead space at the front of the vectors.  [`trim`] advances
//! `start` in O(1); physical compaction (`drain`) happens only when dead
//! space ≥ live space, giving **amortized O(stride)** per trim regardless of
//! window size.
//!
//! # Invariants
//!
//! * `stride == default.len() == shape.iter().product::<usize>()`
//! * `values.len() == stride × timestamps.len()`
//! * `start <= timestamps.len()`
//! * Logical length `timestamps.len() - start >= 0`
//!
//! # Views
//!
//! Views are zero-allocation borrows into a Store:
//!
//! * [`ElementView`] — immutable view of the current (last) element.
//! * [`ElementViewMut`] — mutable view of a (possibly uncommitted) element.
//! * [`SeriesView`] — immutable view of the full history.

use crate::types::Scalar;

const INITIAL_CAPACITY: usize = 16;
const INITIAL_TIMESTAMP: i64 = i64::MIN;

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// Unified time series storage.
///
/// See [module-level docs](self).
#[derive(Debug)]
pub struct Store<T: Scalar> {
    values: Vec<T>,
    timestamps: Vec<i64>,
    shape: Box<[usize]>,
    default: Box<[T]>,
    start: usize,
    window: usize,
}

impl<T: Scalar> Store<T> {
    /// Create a store with an explicit window size.
    ///
    /// * `window = 0`: unbounded (full history).
    /// * `window >= 1`: fixed window of at most `window` elements.
    ///
    /// `default.len()` must equal the stride (product of shape dimensions).
    pub fn new(shape: &[usize], default: &[T], window: usize) -> Self {
        let stride = shape.iter().product::<usize>();
        assert!(default.len() == stride);
        let capacity = Self::capacity_for(window);
        let mut values = Vec::with_capacity(stride * capacity);
        values.extend_from_slice(default);
        let mut timestamps = Vec::with_capacity(capacity);
        timestamps.push(INITIAL_TIMESTAMP);
        Self {
            values,
            timestamps,
            shape: shape.into(),
            default: default.into(),
            start: 0,
            window,
        }
    }

    /// Create a single-element store (`window = 1`).
    pub fn element(shape: &[usize], default: &[T]) -> Self {
        Self::new(shape, default, 1)
    }

    /// Create an unbounded store (`window = 0`, full history).
    pub fn series(shape: &[usize], default: &[T]) -> Self {
        Self::new(shape, default, 0)
    }

    // -- Accessors -----------------------------------------------------------

    /// Element shape (e.g. `&[2, 3]` for a 2×3 matrix element).
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of scalars per element.
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.default.len()
    }

    /// Window size: `0` = unbounded, `N` = at most N elements.
    #[inline(always)]
    pub fn window(&self) -> usize {
        self.window
    }

    /// Whether the store is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.timestamps.len() == self.start
    }

    /// Number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.timestamps.len() - self.start
    }

    /// All logical timestamps.
    #[inline(always)]
    pub fn timestamps(&self) -> &[i64] {
        &self.timestamps[self.start..]
    }

    /// All logical scalar values (flat, length = `stride × len`).
    #[inline(always)]
    pub fn values(&self) -> &[T] {
        &self.values[self.start * self.stride()..]
    }

    /// Current element (last `stride` scalars).
    #[inline(always)]
    pub fn current(&self) -> &[T] {
        let start = self.values.len() - self.stride();
        &self.values[start..]
    }

    /// Current element (mutable, last `stride` scalars).
    #[inline(always)]
    pub fn current_mut(&mut self) -> &mut [T] {
        let start = self.values.len() - self.stride();
        &mut self.values[start..]
    }

    // -- Views ---------------------------------------------------------------

    /// Immutable view of the current element.
    #[inline(always)]
    pub fn current_view(&self) -> ElementView<'_, T> {
        ElementView {
            values: self.current(),
            shape: &self.shape,
        }
    }

    /// Mutable view of the current element.
    #[inline(always)]
    pub fn current_view_mut(&mut self) -> ElementViewMut<'_, T> {
        let start = self.values.len() - self.stride();
        ElementViewMut {
            values: &mut self.values[start..],
            shape: &self.shape,
        }
    }

    /// Immutable view of the full logical history.
    #[inline(always)]
    pub fn series_view(&self) -> SeriesView<'_, T> {
        SeriesView {
            values: self.values(),
            timestamps: self.timestamps(),
            shape: &self.shape,
            stride: self.stride(),
        }
    }

    // -- Mutation -------------------------------------------------------------

    /// Push a new element with known values.  Always succeeds.
    #[inline(always)]
    pub fn push(&mut self, timestamp: i64, value: &[T]) {
        assert!(value.len() == self.stride());
        self.values.extend_from_slice(value);
        self.timestamps.push(timestamp);
        self.trim();
    }

    /// Tentatively append a new element filled with the store's default values.
    ///
    /// The new element is appended but **not** trimmed.  The caller must
    /// call [`commit`] on success or [`rollback`] on failure.
    ///
    /// After this call, `element_view_mut()` points at the new element,
    /// pre-filled with the default values from construction.
    #[inline(always)]
    pub fn push_default(&mut self, timestamp: i64) {
        self.values.extend_from_slice(&self.default);
        self.timestamps.push(timestamp);
    }

    /// Commit a tentative push: trim the window.
    ///
    /// Called after [`push_default`] when the operator/source produced output.
    #[inline(always)]
    pub fn commit(&mut self) {
        self.trim();
    }

    /// Roll back a tentative push: remove the last element.
    ///
    /// Called after [`push_default`] when the operator/source did NOT
    /// produce output.  Restores the store to its pre-push state.
    #[inline(always)]
    pub fn rollback(&mut self) {
        assert!(self.len() > 1);
        self.values.truncate(self.values.len() - self.stride());
        self.timestamps.pop();
    }

    /// Ensure the window is at least `min` elements.
    ///
    /// * `min == 0` → set to unbounded.
    /// * Already unbounded → no-op.
    /// * Otherwise → set to `max(current, min)`.
    pub fn ensure_min_window(&mut self, min: usize) {
        if min == 0 {
            self.window = 0;
        } else if self.window != 0 {
            self.window = self.window.max(min);
        }
        let capacity = Self::capacity_for(self.window);
        if capacity > self.timestamps.len() {
            self.values
                .reserve(self.stride() * capacity - self.values.len());
            self.timestamps.reserve(capacity - self.timestamps.len());
        }
    }

    /// Advance `start` so that logical length ≤ window.  O(1).
    /// Then compact if dead space ≥ compact threshold.
    fn trim(&mut self) {
        if self.window > 0 {
            let len = self.len();
            if len > self.window {
                self.start += len - self.window;
                if self.start + self.window >= Self::capacity_for(self.window) {
                    self.compact();
                }
            }
        }
    }

    /// Physically remove dead elements at the front.
    fn compact(&mut self) {
        if self.start > 0 {
            self.values.drain(..self.start * self.stride());
            self.timestamps.drain(..self.start);
            self.start = 0;
        }
    }

    /// Storage capacity for a given window size.
    /// For unbounded series, returns minimum capacity.
    fn capacity_for(window: usize) -> usize {
        INITIAL_CAPACITY.max(window * 2)
    }
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// Immutable view of the current element in a [`Store`].
///
/// Two fat pointers: `values` (length = stride) and `shape`.
/// Construction is a bounds computation — no allocation.
#[derive(Debug, Clone, Copy)]
pub struct ElementView<'a, T: Scalar> {
    /// Scalar values of the current element (length = stride).
    pub values: &'a [T],
    /// Element shape.
    pub shape: &'a [usize],
}

/// Mutable view of a (possibly uncommitted) element in a [`Store`].
///
/// One mutable fat pointer (`values`, length = stride) and one shared fat
/// pointer (`shape`).  Construction is a bounds computation — no allocation.
#[derive(Debug)]
pub struct ElementViewMut<'a, T: Scalar> {
    /// Scalar values of the element (length = stride, mutable).
    pub values: &'a mut [T],
    /// Element shape.
    pub shape: &'a [usize],
}

/// Immutable view of the full history in a [`Store`].
///
/// Two fat pointers (`values`, `timestamps`), one shared pointer (`shape`),
/// and one `usize` (`stride`).  Construction is trivial — no allocation.
#[derive(Debug, Clone, Copy)]
pub struct SeriesView<'a, T: Scalar> {
    /// All scalar values (length = stride × len).
    pub values: &'a [T],
    /// All timestamps (length = len).
    pub timestamps: &'a [i64],
    /// Element shape.
    pub shape: &'a [usize],
    /// Scalars per element.
    pub stride: usize,
}

// -- View convenience methods ------------------------------------------------

impl<T: Scalar> ElementView<'_, T> {
    /// Number of scalars in this element.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl<T: Scalar> ElementViewMut<'_, T> {
    /// Number of scalars in this element.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Overwrite all values from a slice.
    #[inline(always)]
    pub fn clone_from_slice(&mut self, src: &[T]) {
        self.values.clone_from_slice(src);
    }
}

impl<T: Scalar> SeriesView<'_, T> {
    /// Number of elements in the series.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Current (last) element values.
    #[inline(always)]
    pub fn current(&self) -> &[T] {
        let start = self.values.len() - self.stride;
        &self.values[start..]
    }

    /// Element at index `i` (0-based within the logical window).
    #[inline(always)]
    pub fn element(&self, i: usize) -> &[T] {
        let start = i * self.stride;
        &self.values[start..start + self.stride]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_store_basic() {
        let mut s = Store::element(&[], &[0.0_f64]);
        assert_eq!(s.window(), 1);
        assert_eq!(s.len(), 1);
        assert_eq!(s.stride(), 1);
        assert_eq!(s.current(), &[0.0]);
        assert_eq!(s.timestamps(), &[INITIAL_TIMESTAMP]);

        s.push(100, &[42.0]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.current(), &[42.0]);
        assert_eq!(s.timestamps(), &[100]);
    }

    #[test]
    fn series_store_basic() {
        let mut s = Store::series(&[2], &[1.0_f64, 2.0]);
        assert_eq!(s.window(), 0);
        assert_eq!(s.len(), 1);
        assert_eq!(s.stride(), 2);
        assert_eq!(s.current(), &[1.0, 2.0]);

        s.push(10, &[3.0, 4.0]);
        assert_eq!(s.len(), 2);
        assert_eq!(s.current(), &[3.0, 4.0]);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.timestamps(), &[INITIAL_TIMESTAMP, 10]);

        s.push(20, &[5.0, 6.0]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn fixed_window_store() {
        let mut s = Store::new(&[], &[0.0_f64], 3);
        assert_eq!(s.window(), 3);
        s.push(1, &[1.0]);
        s.push(2, &[2.0]);
        assert_eq!(s.len(), 3); // initial + 2
        assert_eq!(s.values(), &[0.0, 1.0, 2.0]);

        // 4th element trims the oldest.
        s.push(3, &[3.0]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0]);
        assert_eq!(s.timestamps(), &[1, 2, 3]);

        s.push(4, &[4.0]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.values(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn element_view_from_store() {
        let s = Store::element(&[3], &[10.0_f64, 20.0, 30.0]);
        let v = s.current_view();
        assert_eq!(v.values, &[10.0, 20.0, 30.0]);
        assert_eq!(v.shape, &[3]);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn element_view_mut_from_store() {
        let mut s = Store::element(&[], &[0.0_f64]);
        {
            let v = s.current_view_mut();
            v.values[0] = 99.0;
        }
        assert_eq!(s.current(), &[99.0]);
    }

    #[test]
    fn series_view_from_store() {
        let mut s = Store::series(&[], &[1.0_f64]);
        s.push(10, &[2.0]);
        s.push(20, &[3.0]);

        let v = s.series_view();
        assert_eq!(v.len(), 3);
        assert_eq!(v.values, &[1.0, 2.0, 3.0]);
        assert_eq!(v.timestamps, &[INITIAL_TIMESTAMP, 10, 20]);
        assert_eq!(v.current(), &[3.0]);
        assert_eq!(v.element(0), &[1.0]);
        assert_eq!(v.element(1), &[2.0]);
    }

    #[test]
    fn push_default_unbounded_success() {
        let mut s = Store::series(&[], &[0.0_f64]);
        s.push_default(10);
        s.current_view_mut().values[0] = 42.0;
        s.commit();
        assert_eq!(s.len(), 2);
        assert_eq!(s.current(), &[42.0]);
        assert_eq!(s.timestamps(), &[INITIAL_TIMESTAMP, 10]);
    }

    #[test]
    fn push_default_unbounded_rollback() {
        let mut s = Store::series(&[], &[0.0_f64]);
        s.push_default(10);
        // Operator decides not to produce — caller rolls back.
        s.rollback();
        assert_eq!(s.len(), 1);
        assert_eq!(s.current(), &[0.0]);
    }

    #[test]
    fn push_default_single_success() {
        let mut s = Store::element(&[], &[0.0_f64]);
        s.push_default(10);
        s.current_view_mut().values[0] = 7.0;
        s.commit();
        assert_eq!(s.len(), 1);
        assert_eq!(s.current(), &[7.0]);
        assert_eq!(s.timestamps(), &[10]);
    }

    #[test]
    fn push_default_fixed_window() {
        let mut s = Store::new(&[], &[0.0_f64], 2);
        s.push_default(1);
        s.current_view_mut().values[0] = 1.0;
        s.commit();
        assert_eq!(s.len(), 2);
        assert_eq!(s.values(), &[0.0, 1.0]);

        // Third element trims oldest.
        s.push_default(2);
        s.current_view_mut().values[0] = 2.0;
        s.commit();
        assert_eq!(s.len(), 2);
        assert_eq!(s.values(), &[1.0, 2.0]);
        assert_eq!(s.timestamps(), &[1, 2]);
    }

    #[test]
    fn push_default_vector() {
        let mut s = Store::series(&[3], &[1.0_f64, 2.0, 3.0]);
        s.push_default(100);
        {
            let view = s.current_view_mut();
            assert_eq!(view.shape, &[3]);
            view.values[0] = 10.0;
            view.values[1] = 20.0;
            view.values[2] = 30.0;
        }
        s.commit();
        assert_eq!(s.len(), 2);
        assert_eq!(s.current(), &[10.0, 20.0, 30.0]);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn scalar_shape() {
        let s = Store::element(&[], &[42.0_f64]);
        assert_eq!(s.stride(), 1);
        assert_eq!(s.shape(), &[] as &[usize]);
        assert_eq!(s.current(), &[42.0]);
    }

    #[test]
    fn ensure_min_window() {
        let mut s = Store::element(&[], &[0.0_f64]);
        assert_eq!(s.window(), 1);

        s.ensure_min_window(5);
        assert_eq!(s.window(), 5);

        s.ensure_min_window(3);
        assert_eq!(s.window(), 5); // can only grow

        s.ensure_min_window(0);
        assert_eq!(s.window(), 0); // unbounded

        s.ensure_min_window(10);
        assert_eq!(s.window(), 0); // stays unbounded
    }

    #[test]
    fn compaction_amortised() {
        // Window=2: after many pushes, start advances and periodic
        // compaction keeps memory bounded.
        let mut s = Store::new(&[], &[0.0_f64], 2);
        for i in 1..=100 {
            s.push(i, &[i as f64]);
        }
        assert_eq!(s.len(), 2);
        assert_eq!(s.current(), &[100.0]);
        assert_eq!(s.values(), &[99.0, 100.0]);
        // Dead space is bounded by the compact threshold.
        assert!(s.start < s.window());
    }
}
