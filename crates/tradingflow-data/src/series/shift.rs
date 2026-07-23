//! Helper functions for shifting timestamps in a [`SeriesView`].

use super::SeriesView;
use crate::Scalar;

/// Shifts the index of `a` by `n` periods, returning a series view.
///
/// Unpaired elements are dropped rather than filled with `NaN`.
/// The output series can be shorter than the input series.
pub fn shift<T: Scalar, const N: usize>(a: SeriesView<T, N>, n: isize) -> SeriesView<T, N> {
    let layout = a.layout();
    let stride = a.stride();
    let instants = if n > 0 {
        &a.instants()[a.len().min(n.unsigned_abs())..]
    } else {
        &a.instants()[..a.len().saturating_sub(n.unsigned_abs())]
    };
    let data = if n > 0 {
        a.data()
    } else {
        &a.data()[a.len().min(n.unsigned_abs()) * stride..]
    };
    let base = a.range().start;
    // SAFETY: by invariants of `a`.
    unsafe { SeriesView::from_parts_unchecked(layout, stride, instants, data, base) }
}
