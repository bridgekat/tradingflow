//! Integration tests for `tradingflow::operators::predictor`.
//!
//! Every predictor is a Python segment, so the whole file needs the `python`
//! feature and an interpreter with NumPy (plus SciPy, and CVXPY for
//! [`lasso`](tradingflow::operators::predictor::mean::lasso)).
//!
//! All of them share one wiring —
//! `(sample_signal, features, target, rebalance_signal, universe)` — and one
//! cadence: sample ticks record, rebalance ticks emit. So they share a driver,
//! generic over the rank of what they emit, and the tests differ only in the
//! panel they feed and the property they pin.
//!
//! The linear predictors are checked against an *exactly* linear panel, where
//! the pooled fit has a unique zero-residual solution and the prediction is
//! `features · beta` to machine precision. That pins the model rather than a
//! transcribed constant, and it makes the window, offset and masking tests
//! sharp: a fit that reached back one period too far, or paired the wrong
//! target, misses exactly and visibly.

#![cfg(feature = "python")]

use tradingflow::data::{Array, Instant};
use tradingflow::graph::typed::Builder;
use tradingflow::graph::{Pool, Segment};
use tradingflow::operators::array::constant;
use tradingflow::operators::predictor::{Config, mean, variance};
use tradingflow::operators::series::record_all;
use tradingflow::ports::{ArrayPort, SignalPort};

use crate::harness::*;

/// Slightly looser than [`EPS`]: the linear predictors reach their answer
/// through a QR factorization, not a running sum.
const TOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// One driver step: what is sampled this generation, and whether it closes a
/// rebalance.
struct Tick {
    /// The `(N, F)` features and `(N,)` target recorded this generation, or
    /// `None` for a generation that records nothing.
    sample: Option<(Vec<f64>, Vec<f64>)>,
    rebalance: bool,
}

/// Records a `(features, target)` pair without rebalancing.
fn sample(features: &[f64], target: &[f64]) -> Tick {
    Tick {
        sample: Some((features.to_vec(), target.to_vec())),
        rebalance: false,
    }
}

/// Records a pair and then rebalances on it — the common case, where the
/// prediction uses the cross-section just sampled.
fn sample_and_rebalance(features: &[f64], target: &[f64]) -> Tick {
    Tick {
        sample: Some((features.to_vec(), target.to_vec())),
        rebalance: true,
    }
}

/// Rebalances without recording anything.
fn rebalance() -> Tick {
    Tick {
        sample: None,
        rebalance: true,
    }
}

/// A generation in which nothing at all happens — the quiet tick that proves
/// the prediction is retained rather than recomputed.
fn idle() -> Tick {
    Tick {
        sample: None,
        rebalance: false,
    }
}

/// What a driven predictor produced.
struct Run {
    /// One row per emission, each the whole flattened prediction.
    emitted: Vec<Vec<f64>>,
    /// The prediction standing on the output port after the last generation.
    held: Vec<f64>,
}

/// Drives `predictor` over `ticks` with a fixed `universe`, wiring
/// `record_all` onto its output so the pulses are observable downstream —
/// reading the signal port directly always yields `false`, since `reset`
/// clears it at the end of every generation.
fn drive<S, const R: usize>(
    predictor: S,
    universe: &[f64],
    num_features: usize,
    ticks: &[Tick],
) -> Run
where
    S: Segment<
            Inputs = (
                SignalPort<0>,
                ArrayPort<f64, 2>,
                ArrayPort<f64, 1>,
                SignalPort<0>,
                ArrayPort<f64, 1>,
            ),
            Outputs = (SignalPort<0>, ArrayPort<f64, R>),
            Context = Instant,
        >,
{
    let n = universe.len();
    let mut b = Builder::new();
    let (sample_sig, sample_sigv) = b.source(signal());
    let (features, featuresv) = b.source(constant(Array::<f64, 2>::zeros([n, num_features])));
    let (target, targetv) = b.source(constant(Array::<f64, 1>::zeros([n])));
    let (rebalance_sig, rebalance_sigv) = b.source(signal());
    let (_, universev) = b.source(constant(universe.to_vec()));

    let out = b.segment(
        predictor,
        (sample_sigv, featuresv, targetv, rebalance_sigv, universev),
    );
    let recorded = b.segment(record_all(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, tick) in ticks.iter().enumerate() {
        if let Some((x, y)) = &tick.sample {
            *g.state_mut(features) = Array::from_parts([n, num_features], x.clone().into());
            *g.state_mut(target) = Array::from_parts([n], y.clone().into());
            let _ = g.state_mut(sample_sig);
        }
        if tick.rebalance {
            let _ = g.state_mut(rebalance_sig);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }

    let held = vals(g.view(out.1));
    let width = held.len();
    let flat = series_vals(g.view(recorded));
    Run {
        emitted: flat.chunks(width).map(<[f64]>::to_vec).collect(),
        held,
    }
}

// ---------------------------------------------------------------------------
// An exactly linear panel
// ---------------------------------------------------------------------------

/// The coefficients the linear panel below is built from.
const BETA: [f64; 2] = [1.5, -0.5];

/// Three stocks by two features, varied enough per period that the pooled
/// design is full rank after a single tick.
fn features_at(period: usize) -> Vec<f64> {
    let k = period as f64;
    vec![1.0 + k, 0.5, -0.5, 2.0 + k, 2.0, -1.0 - k]
}

/// The target that makes the panel exactly linear: `target = features · beta`,
/// with no residual, so the pooled fit is unique and zero-error.
fn linear_target(features: &[f64]) -> Vec<f64> {
    features
        .chunks(2)
        .map(|row| row[0] * BETA[0] + row[1] * BETA[1])
        .collect()
}

/// The prediction an exact fit must produce for a given cross-section.
fn expected(features: &[f64]) -> Vec<f64> {
    linear_target(features)
}

/// `count` periods of the exactly linear panel, rebalancing on the last.
fn linear_run(count: usize) -> Vec<Tick> {
    (0..count)
        .map(|p| {
            let x = features_at(p);
            let y = linear_target(&x);
            if p == count - 1 {
                sample_and_rebalance(&x, &y)
            } else {
                sample(&x, &y)
            }
        })
        .collect()
}

const ALL: [f64; 3] = [1.0, 1.0, 1.0];

// ---------------------------------------------------------------------------
// Mean predictors
// ---------------------------------------------------------------------------

/// On a panel with a zero-residual linear fit, pooled OLS must find it: the
/// prediction is `features · beta` to machine precision, for every stock.
#[test]
fn pooled_ols_recovers_an_exactly_linear_panel() {
    let run = drive(
        mean::linear_regression(Config::default(), None, 0),
        &ALL,
        2,
        &linear_run(4),
    );

    assert_eq!(run.emitted.len(), 1, "one emission per rebalance");
    assert_close_tol(
        &run.emitted[0],
        &expected(&features_at(3)),
        TOL,
        "prediction on the newest cross-section",
    );
}

/// The incremental fit maintains a Gram instead of re-reading the window, but
/// it is the same estimator — on the same panel it must land on the same
/// numbers, not merely close ones.
#[test]
fn the_incremental_fit_agrees_with_the_windowed_one() {
    let ticks = linear_run(6);
    let windowed = drive(
        mean::linear_regression(Config::default(), None, 0),
        &ALL,
        2,
        &ticks,
    );
    let incremental = drive(
        mean::linear_regression_incr(Config::default()),
        &ALL,
        2,
        &ticks,
    );

    assert_close_tol(
        &incremental.emitted[0],
        &windowed.emitted[0],
        TOL,
        "incremental OLS vs windowed OLS",
    );
}

/// Ridge shrinks coefficients toward zero, so its predictions sit between the
/// unbiased OLS fit and the flat target mean — and a heavier penalty sits
/// closer to the mean. Pinning the ordering rather than a value is what makes
/// this a test of the penalty rather than of the solver.
#[test]
fn a_heavier_ridge_penalty_shrinks_further_toward_the_mean() {
    let ticks = linear_run(6);
    let target = linear_target(&features_at(5));
    let mean_of = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let flat = mean_of(&target);

    let distance = |alpha: f64| {
        let run = drive(
            mean::ridge(Config::default(), alpha, None, 0),
            &ALL,
            2,
            &ticks,
        );
        let d: f64 = run.emitted[0].iter().map(|p| (p - flat).abs()).sum();
        d
    };

    let (light, heavy) = (distance(0.01), distance(10.0));
    assert!(
        heavy < light,
        "alpha=10 should shrink further than alpha=0.01: {heavy} vs {light}"
    );
    assert!(
        light > 0.0,
        "a light penalty should still spread predictions"
    );
}

/// The sample predictor ignores features entirely and answers with each
/// stock's own mean target over the window.
#[test]
fn the_sample_predictor_is_the_per_stock_mean_target() {
    let run = drive(mean::sample(Config::default()), &ALL, 2, &linear_run(4));

    let means: Vec<f64> = (0..3)
        .map(|stock| {
            (0..4)
                .map(|p| linear_target(&features_at(p))[stock])
                .sum::<f64>()
                / 4.0
        })
        .collect();
    assert_close_tol(&run.emitted[0], &means, TOL, "per-stock sample mean");
}

/// A prediction fires exactly once per rebalance and is *retained* in
/// between — quiet generations neither re-emit nor clear it.
#[test]
fn predictions_pulse_once_per_rebalance_and_are_retained_between() {
    let x0 = features_at(0);
    let x1 = features_at(1);
    let run = drive(
        mean::linear_regression(Config::default(), None, 0),
        &ALL,
        2,
        &[
            sample(&x0, &linear_target(&x0)),
            sample_and_rebalance(&x1, &linear_target(&x1)),
            idle(),
            idle(),
        ],
    );

    assert_eq!(run.emitted.len(), 1, "one pulse, from the single rebalance");
    assert_close_tol(
        &run.held,
        &expected(&x1),
        TOL,
        "the prediction survives two quiet generations",
    );
}

/// A stock outside the universe is not predicted for. `NaN` — not zero — is
/// the marker, so downstream consumers cannot mistake "no opinion" for "no
/// expected return".
#[test]
fn stocks_outside_the_universe_are_not_predicted_for() {
    let run = drive(
        mean::linear_regression(Config::default(), None, 0),
        &[1.0, 0.0, 1.0],
        2,
        &linear_run(4),
    );

    let full = expected(&features_at(3));
    assert_close_tol(
        &run.emitted[0],
        &[full[0], f64::NAN, full[2]],
        TOL,
        "the excluded stock is NaN",
    );
}

/// `min_periods` withholds a stock until the window covers it well enough.
/// The rebalance still fires — the cadence is a property of the wiring, not of
/// whether the model had anything to say.
#[test]
fn min_periods_withholds_stocks_short_of_coverage() {
    let config = Config {
        min_periods: Some(4),
        ..Config::default()
    };

    let short = drive(
        mean::linear_regression(config, None, 0),
        &ALL,
        2,
        &linear_run(3),
    );
    assert_eq!(short.emitted.len(), 1, "the rebalance still fires");
    assert_close_tol(
        &short.emitted[0],
        &[f64::NAN; 3],
        TOL,
        "three periods is short of min_periods=4",
    );

    let long = drive(
        mean::linear_regression(config, None, 0),
        &ALL,
        2,
        &linear_run(4),
    );
    assert_close_tol(
        &long.emitted[0],
        &expected(&features_at(3)),
        TOL,
        "the fourth period unlocks the prediction",
    );
}

/// `max_periods` bounds the window, so a regime the panel has moved past
/// stops influencing the fit. Feeding two exactly-linear regimes with
/// different coefficients makes that sharp: a window that reached one period
/// too far would blend them and miss both.
#[test]
fn the_window_forgets_periods_beyond_max_periods() {
    // Two periods under BETA, then two under a different vector.
    let other = [0.25, 2.0];
    let shifted = |x: &[f64]| -> Vec<f64> {
        x.chunks(2)
            .map(|row| row[0] * other[0] + row[1] * other[1])
            .collect()
    };

    let mut ticks = Vec::new();
    for p in 0..2 {
        let x = features_at(p);
        ticks.push(sample(&x, &linear_target(&x)));
    }
    for p in 2..4 {
        let x = features_at(p);
        let y = shifted(&x);
        ticks.push(if p == 3 {
            sample_and_rebalance(&x, &y)
        } else {
            sample(&x, &y)
        });
    }

    let run = drive(
        mean::linear_regression(
            Config {
                max_periods: Some(2),
                ..Config::default()
            },
            None,
            0,
        ),
        &ALL,
        2,
        &ticks,
    );

    assert_close_tol(
        &run.emitted[0],
        &shifted(&features_at(3)),
        TOL,
        "only the last two periods are in the window",
    );
}

/// `target_offset` pairs each feature cross-section with a *later* target,
/// which is what makes the fit predictive rather than contemporaneous. Here
/// the target realized at period `p + 1` is generated from the features of
/// period `p`, so only the offset pairing has a zero-residual solution.
#[test]
fn target_offset_pairs_features_with_a_later_target() {
    let mut ticks = Vec::new();
    for p in 0..5 {
        let x = features_at(p);
        // The target landing now was generated by the previous period's
        // features; the very first has no predecessor, so leave it unfittable
        // and let min-coverage carry it.
        let y = linear_target(&features_at(p.saturating_sub(1)));
        ticks.push(if p == 4 {
            sample_and_rebalance(&x, &y)
        } else {
            sample(&x, &y)
        });
    }

    let run = drive(
        mean::linear_regression(
            Config {
                target_offset: 1,
                ..Config::default()
            },
            None,
            0,
        ),
        &ALL,
        2,
        &ticks,
    );

    assert_close_tol(
        &run.emitted[0],
        &expected(&features_at(4)),
        TOL,
        "the offset fit predicts the next period's target",
    );
}

/// Lasso reaches the same kind of fit through a solver rather than a QR, so a
/// negligible penalty must land back on the OLS answer.
#[test]
fn lasso_with_a_negligible_penalty_agrees_with_ols() {
    let ticks = linear_run(6);
    let ols = drive(
        mean::linear_regression(Config::default(), None, 0),
        &ALL,
        2,
        &ticks,
    );
    let l1 = drive(
        mean::lasso(Config::default(), 1e-9, None, 0),
        &ALL,
        2,
        &ticks,
    );

    assert_close_tol(&l1.emitted[0], &ols.emitted[0], 1e-4, "lasso vs OLS");
}

// ---------------------------------------------------------------------------
// Variance predictors
// ---------------------------------------------------------------------------

/// Five stocks — enough that a covariance matrix has a 2x2 minor drawn
/// entirely from off-diagonal cells, which is what makes the single-index rank
/// test expressible at all.
const WIDE: [f64; 5] = [1.0; 5];

/// A deterministic return panel with real cross-sectional structure: a shared
/// cycle each stock sees at a different phase, plus a per-stock wobble. Smooth
/// rather than random so the tests are exactly reproducible, and wide enough
/// that the estimators genuinely diverge from one another.
fn variance_ticks(periods: usize, stocks: usize) -> Vec<Tick> {
    (0..periods)
        .map(|p| {
            let k = p as f64;
            let x: Vec<f64> = (0..stocks)
                .flat_map(|i| [1.0 + k + i as f64, 0.5 - i as f64])
                .collect();
            let y: Vec<f64> = (0..stocks)
                .map(|i| {
                    let i = i as f64;
                    0.01 * (0.7 * k + 0.4 * i).sin() + 0.004 * (1.3 * k * (i + 1.0)).cos()
                })
                .collect();
            if p == periods - 1 {
                sample_and_rebalance(&x, &y)
            } else {
                sample(&x, &y)
            }
        })
        .collect()
}

/// Reference pairwise-complete sample covariance of a panel, straight from the
/// definition.
fn reference_covariance(ticks: &[Tick], stocks: usize) -> Vec<f64> {
    let rows: Vec<&Vec<f64>> = ticks
        .iter()
        .filter_map(|t| t.sample.as_ref())
        .map(|(_, y)| y)
        .collect();
    let t = rows.len() as f64;
    let means: Vec<f64> = (0..stocks)
        .map(|i| rows.iter().map(|r| r[i]).sum::<f64>() / t)
        .collect();

    let mut out = vec![0.0; stocks * stocks];
    for i in 0..stocks {
        for j in 0..stocks {
            let s: f64 = rows
                .iter()
                .map(|r| (r[i] - means[i]) * (r[j] - means[j]))
                .sum();
            out[i * stocks + j] = s / (t - 1.0);
        }
    }
    out
}

/// The common-covariance shrinkage target: every variance the average sample
/// variance, every covariance the average sample covariance.
fn common_covariance_target(cov: &[f64], stocks: usize) -> Vec<f64> {
    let diag: f64 = (0..stocks).map(|i| cov[i * stocks + i]).sum::<f64>() / stocks as f64;
    let off: f64 = (0..stocks)
        .flat_map(|i| (0..stocks).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .map(|(i, j)| cov[i * stocks + j])
        .sum::<f64>()
        / (stocks * stocks - stocks) as f64;

    let mut out = vec![off; stocks * stocks];
    for i in 0..stocks {
        out[i * stocks + i] = diag;
    }
    out
}

/// The sample estimator is exactly the textbook covariance — the baseline the
/// structured estimators are defined as departures from.
#[test]
fn the_sample_estimator_is_the_textbook_covariance() {
    let ticks = variance_ticks(20, 5);
    let run = drive(variance::sample(Config::default()), &WIDE, 2, &ticks);

    assert_eq!(run.emitted.len(), 1);
    assert_close_tol(
        &run.emitted[0],
        &reference_covariance(&ticks, 5),
        TOL,
        "pairwise sample covariance",
    );
}

/// Every covariance estimator must return a finite, symmetric matrix with a
/// non-negative diagonal. An optimizer downstream assumes all three, and a
/// violation yields silently wrong risk numbers rather than an error.
#[test]
fn every_covariance_estimator_is_symmetric() {
    let ticks = variance_ticks(20, 5);
    let c = Config::default();

    let runs: Vec<(&str, Run)> = vec![
        ("sample", drive(variance::sample(c), &WIDE, 2, &ticks)),
        (
            "single_index",
            drive(variance::single_index(c), &WIDE, 2, &ticks),
        ),
        (
            "shrinkage[common]",
            drive(
                variance::shrinkage(c, variance::Target::CommonCovariance),
                &WIDE,
                2,
                &ticks,
            ),
        ),
        (
            "shrinkage[correlation]",
            drive(
                variance::shrinkage(c, variance::Target::ConstantCorrelation),
                &WIDE,
                2,
                &ticks,
            ),
        ),
        (
            "shrinkage[single_index]",
            drive(
                variance::shrinkage(c, variance::Target::SingleIndex),
                &WIDE,
                2,
                &ticks,
            ),
        ),
        (
            "rmt[zero]",
            drive(
                variance::rmt(c, variance::Replacement::Zero),
                &WIDE,
                2,
                &ticks,
            ),
        ),
        (
            "rmt[mean]",
            drive(
                variance::rmt(c, variance::Replacement::Mean),
                &WIDE,
                2,
                &ticks,
            ),
        ),
        (
            "hierarchical[upgma]",
            drive(
                variance::hierarchical(c, variance::Linkage::Upgma),
                &WIDE,
                2,
                &ticks,
            ),
        ),
        (
            "hierarchical[wpgma]",
            drive(
                variance::hierarchical(c, variance::Linkage::Wpgma),
                &WIDE,
                2,
                &ticks,
            ),
        ),
        (
            "hierarchical[hausdorff]",
            drive(
                variance::hierarchical(c, variance::Linkage::Hausdorff),
                &WIDE,
                2,
                &ticks,
            ),
        ),
    ];

    for (name, run) in runs {
        let m = &run.emitted[0];
        assert_eq!(m.len(), 25, "{name}: a 5x5 matrix");
        for i in 0..5 {
            for j in 0..5 {
                let (a, b) = (m[i * 5 + j], m[j * 5 + i]);
                assert!(a.is_finite(), "{name}: non-finite at ({i}, {j}): {m:?}");
                assert!(
                    (a - b).abs() <= TOL,
                    "{name}: asymmetric at ({i}, {j}): {m:?}"
                );
            }
            assert!(
                m[i * 5 + i] >= -TOL,
                "{name}: negative variance on the diagonal: {m:?}"
            );
        }
    }
}

/// Shrinkage is a convex combination, so every entry must land between the
/// sample estimate and the target.
///
/// The intensity is estimated rather than given, so the test also has to rule
/// out the degenerate case: at `alpha = 0` or `1` the bracket holds trivially
/// and proves nothing. A panel wide and long enough for an interior intensity
/// is what makes the assertion bite. Saturation on a short window is itself
/// correct behaviour — a sample covariance estimated from barely more periods
/// than there are stocks really is worth discarding entirely.
#[test]
fn shrinkage_lies_between_the_sample_estimate_and_its_target() {
    let ticks = variance_ticks(20, 5);
    let c = Config::default();

    let sample = drive(variance::sample(c), &WIDE, 2, &ticks);
    let target = common_covariance_target(&sample.emitted[0], 5);
    let shrunk = drive(
        variance::shrinkage(c, variance::Target::CommonCovariance),
        &WIDE,
        2,
        &ticks,
    );

    let mut interior = 0;
    for (i, (&f, (&s, &x))) in target
        .iter()
        .zip(sample.emitted[0].iter().zip(&shrunk.emitted[0]))
        .enumerate()
    {
        let (lo, hi) = if s <= f { (s, f) } else { (f, s) };
        assert!(
            x >= lo - TOL && x <= hi + TOL,
            "entry {i} is {x}, outside [{lo}, {hi}]",
        );
        if x > lo + TOL && x < hi - TOL {
            interior += 1;
        }
    }
    assert_eq!(
        interior, 25,
        "the intensity saturated, so the bracket proves nothing",
    );
}

/// The single-index model is one factor plus idiosyncratic variance, so its
/// off-diagonal entries are `sigma_f^2 * beta_i * beta_j` — rank one. Any 2x2
/// minor drawn entirely from off-diagonal cells therefore has a vanishing
/// determinant. The minor has to avoid the diagonal, since that is exactly
/// where the idiosyncratic term lives.
#[test]
fn the_single_index_covariance_has_a_rank_one_factor_block() {
    let run = drive(
        variance::single_index(Config::default()),
        &WIDE,
        2,
        &variance_ticks(20, 5),
    );
    let m = &run.emitted[0];

    // Rows {0, 1} against columns {2, 3}: all four cells are off-diagonal.
    let det = m[2] * m[5 + 3] - m[3] * m[5 + 2];
    let scale = (m[2] * m[5 + 3]).abs() + (m[3] * m[5 + 2]).abs();
    assert!(
        det.abs() <= 1e-12 * scale.max(1e-12),
        "off-diagonal block is not rank one: determinant {det} against scale {scale}",
    );
}

/// A covariance predictor obeys the same cadence rules as a mean one: a
/// rebalance with no history still fires, rather than stalling a downstream
/// metric that counts on one emission per period.
#[test]
fn a_rebalance_without_history_still_emits() {
    let run = drive(
        variance::sample(Config::default()),
        &WIDE,
        2,
        &[rebalance(), idle()],
    );

    assert_eq!(run.emitted.len(), 1, "the rebalance fires regardless");
    assert_close_tol(&run.emitted[0], &[f64::NAN; 25], TOL, "nothing to estimate");
}

/// A covariance estimator never reads the feature panel — it fits from the
/// target cross-sections alone — so a stock whose features are unusable is
/// still priced. What the estimator needs from that stock is its returns, and
/// it has them.
///
/// This is what lets the variance predictors skip recording features
/// altogether, which on a wide panel is the difference between megabytes and
/// gigabytes of retained history. A mean predictor, which genuinely reads
/// features, drops the same stock.
#[test]
fn a_covariance_predictor_prices_stocks_with_unusable_features() {
    let mut ticks = variance_ticks(20, 5);
    // Stock 0's leading feature is missing throughout.
    for tick in &mut ticks {
        if let Some((x, _)) = tick.sample.as_mut() {
            x[0] = f64::NAN;
        }
    }

    let covariance = drive(variance::sample(Config::default()), &WIDE, 2, &ticks);
    let row = &covariance.emitted[0][..5];
    assert!(
        row.iter().all(|v| v.is_finite()),
        "the covariance still prices stock 0: {row:?}",
    );

    let mean = drive(
        mean::linear_regression(Config::default(), None, 0),
        &WIDE,
        2,
        &ticks,
    );
    assert!(
        mean.emitted[0][0].is_nan(),
        "a feature-reading predictor drops it: {:?}",
        mean.emitted[0],
    );
}

/// A bounded window makes the incremental fit *subtract* periods as they age,
/// which is the one path the expanding window never exercises. Two
/// exactly-linear regimes make the result unambiguous: if the down-date were
/// wrong the fit would still carry the first regime and match neither.
///
/// The pool retains each period's moment contribution rather than its raw
/// cross-section, so this also pins that the two are equivalent.
#[test]
fn the_incremental_window_subtracts_periods_as_they_age() {
    let other = [0.25, 2.0];
    let shifted = |x: &[f64]| -> Vec<f64> {
        x.chunks(2)
            .map(|row| row[0] * other[0] + row[1] * other[1])
            .collect()
    };

    let mut ticks = Vec::new();
    for p in 0..2 {
        let x = features_at(p);
        ticks.push(sample(&x, &linear_target(&x)));
    }
    for p in 2..4 {
        let x = features_at(p);
        let y = shifted(&x);
        ticks.push(if p == 3 {
            sample_and_rebalance(&x, &y)
        } else {
            sample(&x, &y)
        });
    }

    let run = drive(
        mean::linear_regression_incr(Config {
            max_periods: Some(2),
            ..Config::default()
        }),
        &ALL,
        2,
        &ticks,
    );

    assert_close_tol(
        &run.emitted[0],
        &shifted(&features_at(3)),
        TOL,
        "only the two periods still inside the window remain folded in",
    );
}
