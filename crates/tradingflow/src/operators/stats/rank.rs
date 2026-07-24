use num_traits::Float;
use std::cmp::Ordering;

/// Assign each **finite** entry of `src` its average rank position in the range
/// `0..n_valid`, writing it into `pos[i]`; every non-finite entry (NaN or ±∞)
/// gets `NaN`. Tied values share the mean of the positions they span, so equal
/// inputs map to equal ranks (matching the SciPy `average` convention) rather
/// than to an arbitrary order-dependent tie-break.
///
/// `idx` is caller-owned scratch of the same length as `src`. Returns the
/// number of finite entries `n_valid`, so a `pos` of `NaN` and a valid
/// `0..n_valid` position are the two possible outcomes per element.
pub(super) fn rank_positions<T: Float>(src: &[T], idx: &mut [usize], pos: &mut [f64]) -> usize {
    let mut n_valid = 0;
    for (i, &v) in src.iter().enumerate() {
        pos[i] = f64::NAN;
        if v.is_finite() {
            idx[n_valid] = i;
            n_valid += 1;
        }
    }
    idx[..n_valid].sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

    // Sorted equal values are adjacent; give each tie group the mean of the
    // positions it occupies.
    let mut start = 0;
    while start < n_valid {
        let mut end = start + 1;
        while end < n_valid && src[idx[end]] == src[idx[start]] {
            end += 1;
        }
        let avg = (start + end - 1) as f64 / 2.0;
        for &k in &idx[start..end] {
            pos[k] = avg;
        }
        start = end;
    }
    n_valid
}
