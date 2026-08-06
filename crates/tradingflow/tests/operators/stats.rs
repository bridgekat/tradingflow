//! `operators::stats` — the cross-sectional normalizers.
//!
//! All four operators (`percentile`, `gaussianize`, `standardize`,
//! `winsorize`) are stateless with respect to time: each generation they take
//! one array and rank or rescale it across its own elements. The invariants
//! worth pinning are therefore the exact formula, the treatment of missing
//! (non-finite) entries, and the degenerate cross-sections.

use tradingflow::data::{Array, Instant};
use tradingflow::graph::typed::Builder;
use tradingflow::graph::{Operator, Pool};
use tradingflow::operators::{array, stats};
use tradingflow::ports::ArrayPort;

use crate::harness::*;

// ---------------------------------------------------------------------------
// Local fixtures
// ---------------------------------------------------------------------------

/// Pushes one rank-1 cross-section through `op` and returns the output.
///
/// The build-time source carries the same data as the poke so the operator's
/// scratch buffers are sized correctly; the single `stabilize` is the one
/// generation under test.
fn run1<S>(op: S, data: Vec<f64>) -> Vec<f64>
where
    S: Operator<Inputs = ArrayPort<f64, 1>, Outputs = ArrayPort<f64, 1>, Context = Instant>,
{
    let input = data;
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(input.clone()));
    let out = b.op(op, srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = input.into();
    g.stabilize(&mut pool, &Instant::MIN);
    vals(g.view(out))
}

/// The reference percentile: average rank over the *finite* entries only,
/// mapped by `(rank + 0.5) / n_finite`, with `NaN` where the input was not
/// finite. Written out longhand so the tests state the formula rather than
/// restating a decimal table.
fn expected_percentile(x: &[f64]) -> Vec<f64> {
    let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = finite.len() as f64;
    x.iter()
        .map(|&v| {
            if !v.is_finite() {
                return f64::NAN;
            }
            let less = finite.iter().filter(|&&u| u < v).count() as f64;
            let equal = finite.iter().filter(|&&u| u == v).count() as f64;
            // Ties share the mean of the positions `less .. less + equal`.
            let rank = less + (equal - 1.0) / 2.0;
            (rank + 0.5) / n
        })
        .collect()
}

// ---------------------------------------------------------------------------
// percentile
// ---------------------------------------------------------------------------

/// `percentile` maps each element to `(average rank + 0.5) / n`, so an
/// unsorted cross-section comes back as its rank order scaled into `(0, 1)`
/// with no element ever hitting 0 or 1.
#[test]
fn percentile_is_rank_plus_half_over_n() {
    let x = vec![30.0, 10.0, 50.0, 20.0, 40.0];
    let got = run1(stats::percentile(), x.clone());

    assert_close(&got, &expected_percentile(&x), "percentile vs reference");
    // Spelled out: sorted order is 10 < 20 < 30 < 40 < 50, so the ranks are
    // 2, 0, 4, 1, 3 and the outputs are (rank + 0.5) / 5.
    assert_close(&got, &[0.5, 0.1, 0.9, 0.3, 0.7], "percentile literal");
    assert!(
        got.iter().all(|&p| p > 0.0 && p < 1.0),
        "percentile is open on both ends: {got:?}"
    );
}

/// Tied inputs share the mean of the positions they span (SciPy's `average`
/// convention), so equal values get equal percentiles regardless of the order
/// the sort happened to put them in.
#[test]
fn percentile_ties_share_their_average_rank() {
    let x = vec![10.0, 20.0, 20.0, 30.0];
    let got = run1(stats::percentile(), x.clone());

    assert_close(
        &got,
        &expected_percentile(&x),
        "percentile ties vs reference",
    );
    // Positions 1 and 2 average to 1.5 → (1.5 + 0.5) / 4 = 0.5.
    assert_close(&got, &[0.125, 0.5, 0.5, 0.875], "percentile ties literal");
    assert_eq!(got[1], got[2], "tied inputs must rank identically");
}

/// Non-finite entries are *missing*, not extreme: they are dropped from the
/// ranking, they shrink the denominator to the finite count, and their own
/// slots come back as NaN. `+∞` in particular does not rank as the maximum.
#[test]
fn percentile_drops_non_finite_and_shrinks_the_denominator() {
    let x = vec![30.0, f64::NAN, 10.0, f64::INFINITY, 20.0, f64::NEG_INFINITY];
    let got = run1(stats::percentile(), x.clone());

    // Three finite values (10 < 20 < 30) → the denominator is 3, not 6.
    assert_close(
        &got,
        &[
            2.5 / 3.0,
            f64::NAN,
            0.5 / 3.0,
            f64::NAN,
            1.5 / 3.0,
            f64::NAN,
        ],
        "percentile with missing entries",
    );
    assert_close(
        &got,
        &expected_percentile(&x),
        "percentile nan vs reference",
    );
}

// ---------------------------------------------------------------------------
// gaussianize
// ---------------------------------------------------------------------------

/// `gaussianize` is `percentile` pushed through Φ⁻¹: the median element maps
/// to 0, ranks symmetric about the median map to opposite values (so the whole
/// cross-section sums to 0), and the map is strictly increasing in the input.
#[test]
fn gaussianize_is_the_inverse_normal_cdf_of_the_rank() {
    let x = vec![30.0, 10.0, 50.0, 20.0, 40.0];
    let z = run1(stats::gaussianize(), x.clone());

    // 30 is the median of five distinct values → rank 0.5 → Φ⁻¹(0.5) = 0.
    assert!(z[0].abs() < 1e-12, "median must map to 0, got {}", z[0]);
    // Ranks 0.1/0.9 and 0.3/0.7 are mirror images, so everything cancels.
    assert!(
        z.iter().sum::<f64>().abs() < 1e-9,
        "symmetric ranks must sum to 0: {z:?}"
    );

    // Strictly monotone: sorting by the input must sort the output.
    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(z.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert!(
        pairs.windows(2).all(|w| w[0].1 < w[1].1),
        "gaussianize must be increasing in the input: {pairs:?}"
    );

    // Two spot values: the ranks are 0.9 (for 50) and 0.7 (for 40), and Φ⁻¹ of
    // those is the standard normal's 90th and 70th percentile.
    assert_close_tol(
        &[z[2], z[4]],
        &[1.281_551_565_544_600_4, 0.524_400_512_708_040_7],
        1e-9,
        "Φ⁻¹ spot values",
    );
}

/// `gaussianize` and `percentile` share `rank.rs`, so they agree on exactly
/// which slots are missing and on the rank order — the only difference is the
/// final Φ⁻¹. The denominator shrinks to the finite count for both.
#[test]
fn gaussianize_shares_the_percentile_ranking_and_nan_contract() {
    let x = vec![30.0, f64::NAN, 10.0, f64::INFINITY, 20.0];
    let z = run1(stats::gaussianize(), x.clone());
    let p = run1(stats::percentile(), x.clone());

    for (i, (&zi, &pi)) in z.iter().zip(&p).enumerate() {
        assert_eq!(
            zi.is_nan(),
            pi.is_nan(),
            "slot {i}: gaussianize {zi} and percentile {pi} disagree on missingness"
        );
    }
    // Three finite values → ranks 5/6 (30), 1/6 (10) and 0.5 (20).
    assert!(
        z[4].abs() < 1e-12,
        "the middle rank maps to 0, got {}",
        z[4]
    );
    assert!(
        (z[0] + z[2]).abs() < 1e-12,
        "ranks 1/6 and 5/6 must cancel: {} + {}",
        z[0],
        z[2]
    );
    assert!(z[0] > 0.0 && z[2] < 0.0, "rank order must survive: {z:?}");
}

// ---------------------------------------------------------------------------
// standardize
// ---------------------------------------------------------------------------

/// `standardize` divides by the **population** standard deviation (denominator
/// `n`, not `n − 1`). The two conventions differ by ~11% at `n = 5`, so this
/// test fails loudly if the operator ever switches.
#[test]
fn standardize_uses_population_not_sample_variance() {
    let x = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let z = run1(stats::standardize(), x.clone());

    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let ssd: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    let population = (ssd / n).sqrt();
    let sample = (ssd / (n - 1.0)).sqrt();

    let expected: Vec<f64> = x.iter().map(|v| (v - mean) / population).collect();
    assert_close(&z, &expected, "standardize");
    assert!(
        (z[0] - (x[0] - mean) / sample).abs() > 0.1,
        "the sample-variance convention would be visibly different: {z:?}"
    );
}

/// By construction the output has mean 0 and population variance 1.
#[test]
fn standardize_yields_zero_mean_and_unit_variance() {
    let z = run1(stats::standardize(), quarter_path(7, 32));

    let n = z.len() as f64;
    let mean = z.iter().sum::<f64>() / n;
    let var = z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    assert!(mean.abs() < 1e-12, "mean is {mean}");
    assert!((var - 1.0).abs() < 1e-12, "variance is {var}");
}

/// Non-finite entries never enter the mean or the sum of squared deviations,
/// and their own slots come back as NaN. The moments are taken over the finite
/// count alone, so a single `+∞` does not poison the cross-section.
#[test]
fn standardize_excludes_non_finite_from_the_moments() {
    let z = run1(
        stats::standardize(),
        vec![1.0, 2.0, 3.0, f64::INFINITY, f64::NAN],
    );

    // Finite part is [1, 2, 3]: mean 2, population σ = sqrt(2/3).
    let s = (2.0_f64 / 3.0).sqrt();
    assert_close(
        &z,
        &[-1.0 / s, 0.0, 1.0 / s, f64::NAN, f64::NAN],
        "standardize with missing entries",
    );
}

/// A z-score needs at least two finite values and a non-zero spread. Fewer
/// than two finite entries, or a constant cross-section, produces all NaN
/// rather than a division by zero.
#[test]
fn standardize_is_all_nan_without_spread() {
    for (ctx, input) in [
        ("one finite value", vec![5.0, f64::NAN, f64::NAN]),
        ("zero variance", vec![7.0, 7.0, 7.0]),
    ] {
        let z = run1(stats::standardize(), input);
        assert!(z.iter().all(|v| v.is_nan()), "{ctx}: {z:?}");
    }
}

// ---------------------------------------------------------------------------
// winsorize
// ---------------------------------------------------------------------------

/// `winsorize(p)` clamps to the `k`-th smallest and `k`-th largest finite
/// values, where `k = floor(p · n)`. Over `[0, 10)` with `p = 0.1` that is
/// exactly one element off each tail, so the bounds are unambiguously 1 and 8.
#[test]
fn winsorize_clips_at_both_tails() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let w = run1(stats::winsorize(0.1), x);

    assert_eq!(w, vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]);
}

/// The clip depth is `floor(p · n)`, i.e. truncated rather than rounded: at
/// `n = 10`, `p = 0.25` still clips two off each tail and any `p < 0.1` clips
/// nothing at all.
#[test]
fn winsorize_clip_depth_is_floor_of_p_times_n() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();

    assert_eq!(
        run1(stats::winsorize(0.25), x.clone()),
        vec![2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0],
        "floor(0.25 * 10) = 2"
    );
    assert_eq!(
        run1(stats::winsorize(0.09), x.clone()),
        x,
        "floor(0.09 * 10) = 0 → identity"
    );
    assert_eq!(
        run1(stats::winsorize(0.0), x.clone()),
        x,
        "p = 0 → identity"
    );
}

/// Winsorizing an already-winsorized cross-section is a no-op: the clip bounds
/// are order statistics of the input, and clamping does not move them.
#[test]
fn winsorize_is_idempotent() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let once = run1(stats::winsorize(0.1), x);
    let twice = run1(stats::winsorize(0.1), once.clone());

    assert_eq!(twice, once);
}

/// `winsorize` splits the non-finite cases apart from the rank-based
/// operators: NaN is missing and passes straight through, but `±∞` is the
/// outlier winsorization exists to tame, so it is *clamped* to the finite
/// bounds instead of being turned into NaN. Neither ever becomes a bound.
#[test]
fn winsorize_passes_nan_through_but_clamps_infinities() {
    let w = run1(
        stats::winsorize(0.0),
        vec![
            f64::NEG_INFINITY,
            1.0,
            2.0,
            3.0,
            4.0,
            f64::INFINITY,
            f64::NAN,
        ],
    );

    // Bounds come from the four finite values only: lo = 1, hi = 4.
    assert_close(
        &w,
        &[1.0, 1.0, 2.0, 3.0, 4.0, 4.0, f64::NAN],
        "winsorize with non-finite entries",
    );
}

// ---------------------------------------------------------------------------
// corr
// ---------------------------------------------------------------------------

/// Pushes one pair of rank-1 cross-sections through `op` and returns its
/// rank-0 output.
fn run2<S>(op: S, a: Vec<f64>, b: Vec<f64>) -> f64
where
    S: Operator<
            Inputs = (ArrayPort<f64, 1>, ArrayPort<f64, 1>),
            Outputs = ArrayPort<f64, 0>,
            Context = Instant,
        >,
{
    let mut bld = Builder::new();
    let (sa, va) = bld.source(array::constant(a.clone()));
    let (sb, vb) = bld.source(array::constant(b.clone()));
    let out = bld.op(op, (va, vb));
    let mut g = bld.build();
    let mut pool = Pool::new(0);

    *g.state_mut(sa) = a.into();
    *g.state_mut(sb) = b.into();
    g.stabilize(&mut pool, &Instant::MIN);
    vals(g.view(out))[0]
}

/// The reference Pearson correlation over the pairwise-finite entries,
/// written out longhand: centered cross-moment over the geometric mean of the
/// centered second moments.
fn expected_corr(x: &[f64], y: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a, b))
        .collect();
    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let sxy: f64 = pairs.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let sxx: f64 = pairs.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    let syy: f64 = pairs.iter().map(|p| (p.1 - mean_y).powi(2)).sum();
    sxy / (sxx * syy).sqrt()
}

/// `corr` is the Pearson coefficient of the two cross-sections, matching the
/// longhand centered-moment formula on an arbitrary pair.
#[test]
fn corr_matches_the_closed_form() {
    let x = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6];
    let y = vec![2.0, 7.0, 1.0, 8.0, 2.8, 1.8];
    let got = run2(stats::corr(false, 2), x.clone(), y.clone());
    assert_close(&[got], &[expected_corr(&x, &y)], "corr vs reference");
}

/// An affine relation with positive slope correlates to exactly +1, and a
/// negative slope to −1: the coefficient is scale- and shift-invariant.
#[test]
fn corr_is_plus_minus_one_on_affine_pairs() {
    let x = vec![3.0, 1.0, 4.0, 1.5, 9.0];
    let up: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    let down: Vec<f64> = x.iter().map(|v| -0.5 * v + 3.0).collect();
    assert_close(
        &[run2(stats::corr(false, 2), x.clone(), up)],
        &[1.0],
        "slope up",
    );
    assert_close(
        &[run2(stats::corr(false, 2), x, down)],
        &[-1.0],
        "slope down",
    );
}

/// A non-finite entry on **either** side drops the pair (pairwise-complete):
/// the result equals `corr` of the surviving pairs, unaffected by what the
/// dropped positions hold on the other side.
#[test]
fn corr_skips_non_finite_pairwise() {
    // Positions 2 (x is NaN) and 4 (y is +∞) drop; the survivors are affine.
    let x = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
    let y = vec![2.0, 4.0, 999.0, 8.0, f64::INFINITY];
    let got = run2(stats::corr(false, 2), x, y);
    assert_close(&[got], &[1.0], "corr over the surviving affine pairs");
}

/// Degenerate cross-sections give NaN: fewer than two valid pairs, fewer than
/// `min_count` valid pairs, or a zero-variance side.
#[test]
fn corr_is_nan_without_pairs_or_spread() {
    let x = vec![1.0, 2.0, 3.0];
    // One valid pair.
    let sparse = vec![5.0, f64::NAN, f64::NAN];
    assert!(run2(stats::corr(false, 2), x.clone(), sparse).is_nan());
    // Three valid pairs, but the floor is raised above them.
    let y = vec![2.0, 5.0, 4.0];
    assert!(run2(stats::corr(false, 4), x.clone(), y.clone()).is_nan());
    assert!(!run2(stats::corr(false, 3), x.clone(), y.clone()).is_nan());
    // A constant side has zero variance.
    assert!(run2(stats::corr(false, 2), x, vec![7.0, 7.0, 7.0]).is_nan());
}

/// The coefficient is symmetric in its arguments, bit for bit.
#[test]
fn corr_is_symmetric() {
    let x = vec![3.0, 1.0, f64::NAN, 1.5, 9.0];
    let y = vec![2.0, 7.0, 1.0, f64::NAN, 2.8];
    let xy = run2(stats::corr(false, 2), x.clone(), y.clone());
    let yx = run2(stats::corr(false, 2), y, x);
    assert_eq!(xy.to_bits(), yx.to_bits(), "corr(x, y) vs corr(y, x)");
}

// ---------------------------------------------------------------------------
// Shared edge cases
// ---------------------------------------------------------------------------

/// A cross-section with nothing finite in it has no ranking and no moments:
/// every operator returns all NaN rather than dividing by a zero count.
#[test]
fn all_nan_input_is_all_nan_for_every_operator() {
    let x = vec![f64::NAN; 4];
    for (name, got) in [
        ("percentile", run1(stats::percentile(), x.clone())),
        ("gaussianize", run1(stats::gaussianize(), x.clone())),
        ("standardize", run1(stats::standardize(), x.clone())),
        ("winsorize", run1(stats::winsorize(0.1), x.clone())),
    ] {
        assert_eq!(got.len(), x.len(), "{name} must preserve the shape");
        assert!(got.iter().all(|v| v.is_nan()), "{name}: {got:?}");
    }
}

/// A one-element cross-section: the single value is its own median, so
/// `percentile` gives 0.5, `gaussianize` gives Φ⁻¹(0.5) = 0 and `winsorize`
/// clamps to `[x, x]` (the identity). `standardize` needs two finite values
/// and so is the odd one out with NaN.
#[test]
fn single_element_cross_section() {
    assert_eq!(run1(stats::percentile(), vec![42.0]), vec![0.5]);
    assert_close_tol(
        &run1(stats::gaussianize(), vec![42.0]),
        &[0.0],
        1e-12,
        "gaussianize singleton",
    );
    assert_eq!(run1(stats::winsorize(0.1), vec![42.0]), vec![42.0]);
    assert!(run1(stats::standardize(), vec![42.0])[0].is_nan());
}

/// The operators are generic over rank but have no axis parameter: a rank-2
/// array is ranked as **one flat cross-section over all of its elements**, not
/// row by row. The output keeps the input's extents.
#[test]
fn rank_two_input_is_one_cross_section_over_the_whole_array() {
    // Rows [3, 1, 5] and [2, 6, 4] — a per-row ranking would give each row the
    // same three percentiles {1/6, 3/6, 5/6}; a whole-array ranking spreads all
    // six values over (0, 1).
    let data = vec![3.0, 1.0, 5.0, 2.0, 6.0, 4.0];
    let input = Array::from_parts([2, 3], data.clone().into());

    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(input.clone()));
    let pct = b.op(stats::percentile(), srcv);
    let zsc = b.op(stats::standardize(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = input;
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(pct).extents(), [2, 3], "extents are preserved");
    let got = vals(g.view(pct));
    assert_close(&got, &expected_percentile(&data), "rank-2 percentile");
    // Values 1..=6 have ranks 0..=5 over the whole array → (v - 0.5) / 6.
    let sixths: Vec<f64> = data.iter().map(|&v| (v - 0.5) / 6.0).collect();
    assert_close(&got, &sixths, "rank-2 percentile literal");
    assert!(
        (got[0] - 0.5).abs() > 1e-9,
        "a per-row ranking would put the row-0 median 3 at 0.5: {got:?}"
    );

    // Same story for the moments: mean and σ are taken over all six elements.
    let mean = 3.5;
    let sigma = (data.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 6.0).sqrt();
    let expected: Vec<f64> = data.iter().map(|v| (v - mean) / sigma).collect();
    assert_close(&vals(g.view(zsc)), &expected, "rank-2 standardize");
}
