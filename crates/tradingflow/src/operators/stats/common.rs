use num_traits::Float;
use std::cmp::Ordering;

/// Whether a computed variance is indistinguishable from zero, given the
/// theoretical error bounds of the two-pass variance algorithm:
///
/// ```text
/// var <= n·ε·var + (n·mean·ε)²
/// ```
///
/// The first term bounds the accumulation error of summing `n` squared
/// deviations; the second bounds the cancellation error of `x - mean` when
/// the mean dominates the spread. A variance at or under the bound is
/// rounding noise rather than dispersion, and normalizing by it would
/// amplify representation error into full-scale output — the guard exists
/// for the operators that *divide* by the spread they computed. The bound
/// follows Chan, Golub & LeVeque, *Algorithms for computing the sample
/// variance: analysis and recommendations* (1983), and matches
/// scikit-learn's `_is_constant_feature`.
///
/// With `mean = 0` the bound reduces to `1 <= n·ε`, which never holds at any
/// realistic `n` — deliberately so: centering is exact there, no digits were
/// cancelled, and a tiny variance is genuinely computed. The test detects
/// *numerical* degeneracy only; a semantically degenerate cross-section
/// (say, heavy ties with one real straggler) has an exactly-computed
/// variance and passes.
///
/// `variance` may be the sample or the population estimate — the two differ
/// by `n / (n - 1)`, well inside the slack of an order-of-magnitude error
/// bound. An exact zero is always caught (`0 <= (n·mean·ε)²` holds even at
/// `mean = 0`), so callers need no separate `<= 0` branch.
pub fn is_constant<T: Float>(variance: T, mean: T, n: T) -> bool {
    let eps = T::epsilon();
    variance <= n * eps * variance + (n * mean * eps).powi(2)
}

/// Assign each **finite** entry of `src` its average rank position in the range
/// `0..n_valid`, writing it into `pos[i]`; every non-finite entry (NaN or ±∞)
/// gets `NaN`. Tied values share the mean of the positions they span, so equal
/// inputs map to equal ranks (matching the SciPy `average` convention) rather
/// than to an arbitrary order-dependent tie-break.
///
/// `idx` is caller-owned scratch of the same length as `src`. Returns the
/// number of finite entries `n_valid`, so a `pos` of `NaN` and a valid
/// `0..n_valid` position are the two possible outcomes per element.
pub fn rank_positions<T: Float>(src: &[T], idx: &mut [usize], pos: &mut [T]) -> usize {
    let mut n_valid = 0;
    for (i, &v) in src.iter().enumerate() {
        pos[i] = T::nan();
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
        let avg = T::from(start + end - 1).unwrap() / T::from(2).unwrap();
        for &k in &idx[start..end] {
            pos[k] = avg;
        }
        start = end;
    }
    n_valid
}
