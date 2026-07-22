//! The cross-sectional ranking step shared by [`Gaussianize`](super::Gaussianize)
//! and [`Percentile`](super::Percentile). Private — not re-exported.

use std::cmp::Ordering;

use num_traits::Float;

/// Rank the finite entries of `src`: write their indices into `scratch`, in
/// ascending value order, blank the whole of `dst` to NaN, and return how many
/// entries were finite (so `scratch[..n]` holds the ranking).
pub(super) fn rank_finite<T: Float>(src: &[T], scratch: &mut [usize], dst: &mut [T]) -> usize {
    let mut n_valid = 0usize;
    for (i, v) in src.iter().enumerate() {
        if !v.is_nan() {
            scratch[n_valid] = i;
            n_valid += 1;
        }
    }
    scratch[..n_valid].sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

    let nan = T::nan();
    for slot in dst.iter_mut() {
        *slot = nan;
    }
    n_valid
}
