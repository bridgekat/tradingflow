//! Integration tests for `tradingflow::operators::portfolio`.
//!
//! Every portfolio is a Python segment, so the whole file needs the `python`
//! feature and an interpreter with NumPy and SciPy; the solving portfolios
//! additionally need CVXPY.
//!
//! The closed-form mean portfolios are checked against their formulas exactly,
//! since each is a short expression with no solver in the way. The solving
//! portfolios cannot be — an iterative solver has no closed form to compare
//! against — so they are pinned by the properties that define them instead:
//! minimum variance really does carry less risk than equal weight, a tighter
//! tracking-error budget really does pull toward the benchmark, and the two
//! solvers for the same problem really do agree.
//!
//! Most tests set `logarithmic: false`, so the moments reach the optimizer as
//! written and the assertions are about the portfolio rather than about the
//! moment map. [`the_lognormal_map_converts_moments_before_optimizing`] is the
//! one that pins the map itself.

#![cfg(feature = "python")]

use tradingflow::data::Instant;
use tradingflow::graph::typed::Builder;
use tradingflow::graph::{Pool, Segment};
use tradingflow::operators::array::constant;
use tradingflow::operators::portfolio::{Config, mean, mean_variance, variance};
use tradingflow::operators::series::record_all;

use crate::harness::*;

/// The solving portfolios run iterative solvers to a finite tolerance, so
/// their assertions get room the closed-form ones do not need.
const SOLVER_TOL: f64 = 5e-3;

/// Moments as written, so a test asserts on the optimizer rather than on the
/// lognormal conversion in front of it.
fn linear() -> Config {
    Config {
        logarithmic: false,
        ..Config::default()
    }
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// What a driven portfolio produced.
struct Run {
    /// One row per emission, each the whole `[N]` book.
    emitted: Vec<Vec<f64>>,
    /// The book standing on the output port after the last generation.
    held: Vec<f64>,
}

/// Drives a portfolio whose second moment input is an `[N]` vector — the mean
/// portfolios. `rebalances` gives one generation per entry.
fn drive_mean<S>(portfolio: S, universe: &[f64], mu: &[f64], rebalances: &[bool]) -> Run
where
    S: Segment<Inputs = mean::Inputs, Outputs = mean::Outputs, Context = Instant>,
{
    let mut b = Builder::new();
    let (signal, signalv) = b.source(signal());
    let (_, universev) = b.source(constant(arr1(universe.to_vec())));
    let (_, muv) = b.source(constant(arr1(mu.to_vec())));

    let out = b.segment(portfolio, (signalv, universev, muv));
    let recorded = b.segment(record_all(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, &rebalance) in rebalances.iter().enumerate() {
        if rebalance {
            let _ = g.state_mut(signal);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }
    collect(&g.view(out.1), &g.view(recorded))
}

/// Drives a portfolio whose second moment input is an `[N, N]` matrix — the
/// variance portfolios.
fn drive_variance<S>(portfolio: S, universe: &[f64], sigma: &[f64], rebalances: &[bool]) -> Run
where
    S: Segment<Inputs = variance::Inputs, Outputs = variance::Outputs, Context = Instant>,
{
    let n = universe.len();
    let mut b = Builder::new();
    let (signal, signalv) = b.source(signal());
    let (_, universev) = b.source(constant(arr1(universe.to_vec())));
    let (_, sigmav) = b.source(constant(arr([n, n], sigma.to_vec())));

    let out = b.segment(portfolio, (signalv, universev, sigmav));
    let recorded = b.segment(record_all(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, &rebalance) in rebalances.iter().enumerate() {
        if rebalance {
            let _ = g.state_mut(signal);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }
    collect(&g.view(out.1), &g.view(recorded))
}

/// Drives a portfolio taking both moments.
fn drive_mean_variance<S>(
    portfolio: S,
    universe: &[f64],
    mu: &[f64],
    sigma: &[f64],
    rebalances: &[bool],
) -> Run
where
    S: Segment<Inputs = mean_variance::Inputs, Outputs = mean_variance::Outputs, Context = Instant>,
{
    let n = universe.len();
    let mut b = Builder::new();
    let (signal, signalv) = b.source(signal());
    let (_, universev) = b.source(constant(arr1(universe.to_vec())));
    let (_, muv) = b.source(constant(arr1(mu.to_vec())));
    let (_, sigmav) = b.source(constant(arr([n, n], sigma.to_vec())));

    let out = b.segment(portfolio, (signalv, universev, muv, sigmav));
    let recorded = b.segment(record_all(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, &rebalance) in rebalances.iter().enumerate() {
        if rebalance {
            let _ = g.state_mut(signal);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }
    collect(&g.view(out.1), &g.view(recorded))
}

/// Splits a recorded series back into one book per emission. Pulses are only
/// observable downstream, since `reset` clears the output signal at the end of
/// every generation.
fn collect(
    held: &tradingflow::data::ArrayView<'_, f64, 1>,
    recorded: &tradingflow::data::SeriesView<'_, f64, 1>,
) -> Run {
    let held = vals(*held);
    let width = held.len();
    let flat = series_vals(*recorded);
    Run {
        emitted: flat.chunks(width).map(<[f64]>::to_vec).collect(),
        held,
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const N: usize = 6;
const ALL: [f64; N] = [1.0; N];

/// Predicted returns with a clear ordering, one negative, so the long-only
/// portfolios all have something to exclude.
const MU: [f64; N] = [0.03, 0.02, 0.01, -0.01, 0.005, 0.015];

/// A factor-plus-diagonal covariance with deliberately unequal variances, so
/// a risk-aware portfolio has a reason to tilt.
fn covariance() -> Vec<f64> {
    // Loadings on one factor, then an idiosyncratic diagonal.
    let loadings = [0.20, 0.15, 0.10, 0.05, 0.12, 0.08];
    let idiosyncratic = [0.02, 0.01, 0.04, 0.09, 0.01, 0.03];

    let mut out = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            out[i * N + j] = loadings[i] * loadings[j];
        }
        out[i * N + i] += idiosyncratic[i];
    }
    out
}

/// `wᵀΣw`, the predicted variance of a book.
fn risk(weights: &[f64], sigma: &[f64]) -> f64 {
    let n = weights.len();
    (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| weights[i] * sigma[i * n + j] * weights[j])
        .sum()
}

fn total(weights: &[f64]) -> f64 {
    weights.iter().sum()
}

// ---------------------------------------------------------------------------
// Mean portfolios
// ---------------------------------------------------------------------------

/// Whatever rule it applies, a long-only portfolio hands back a fully invested
/// book with no negative weights. Downstream accounting assumes both.
#[test]
fn every_mean_portfolio_is_a_normalized_long_only_book() {
    let c = linear();
    let books: Vec<(&str, Run)> = vec![
        (
            "proportional",
            drive_mean(mean::proportional(c), &ALL, &MU, &[true]),
        ),
        (
            "softmax",
            drive_mean(mean::softmax(c, 0.01), &ALL, &MU, &[true]),
        ),
        (
            "rank_equal",
            drive_mean(mean::rank_equal(c, 0.5), &ALL, &MU, &[true]),
        ),
        (
            "rank_linear",
            drive_mean(mean::rank_linear(c, 0.5), &ALL, &MU, &[true]),
        ),
        (
            "rank_bucket",
            drive_mean(mean::rank_bucket(c, 0.5, 1.0), &ALL, &MU, &[true]),
        ),
    ];

    for (name, run) in books {
        let book = &run.emitted[0];
        assert_eq!(book.len(), N, "{name}: one weight per stock");
        assert_close_tol(
            &[total(book)],
            &[1.0],
            1e-12,
            &format!("{name}: fully invested"),
        );
        for (i, &w) in book.iter().enumerate() {
            assert!(w >= 0.0, "{name}: negative weight {w} at {i}: {book:?}");
        }
    }
}

/// Proportional weighting is the normalized positive part of the predictions,
/// and nothing more.
#[test]
fn proportional_weights_are_the_normalized_positive_predictions() {
    let run = drive_mean(mean::proportional(linear()), &ALL, &MU, &[true]);

    let positive: Vec<f64> = MU.iter().map(|m| m.max(0.0)).collect();
    let sum: f64 = positive.iter().sum();
    let expected: Vec<f64> = positive.iter().map(|p| p / sum).collect();

    assert_close(
        &run.emitted[0],
        &expected,
        "normalized positive predictions",
    );
}

/// The rank portfolios select the same stocks; they differ only in how weight
/// is spread across them. Pinning both against the selection makes that
/// explicit — and catches an off-by-one in the count of stocks taken.
#[test]
fn the_rank_portfolios_agree_on_selection_and_differ_on_weighting() {
    // Five of the six predictions are positive; half of five rounds to two.
    let equal = drive_mean(mean::rank_equal(linear(), 0.5), &ALL, &MU, &[true]);
    let linear_tilt = drive_mean(mean::rank_linear(linear(), 0.5), &ALL, &MU, &[true]);

    // Stocks 0 and 1 have the two highest predictions.
    assert_close(
        &equal.emitted[0],
        &[0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        "equal split across the top two",
    );
    // Linear weights of 2 and 1, normalized.
    assert_close(
        &linear_tilt.emitted[0],
        &[2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
        "linear tilt across the top two",
    );

    let selected = |b: &[f64]| -> Vec<bool> { b.iter().map(|w| *w > 0.0).collect() };
    assert_eq!(
        selected(&equal.emitted[0]),
        selected(&linear_tilt.emitted[0]),
        "the two rank portfolios select the same stocks",
    );
}

/// Temperature is the concentration dial: a cold softmax piles into the
/// best-predicted stock, a hot one flattens toward equal weight.
#[test]
fn softmax_temperature_controls_concentration() {
    let peak = |t: f64| {
        drive_mean(mean::softmax(linear(), t), &ALL, &MU, &[true]).emitted[0]
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max)
    };

    let (cold, warm, hot) = (peak(0.002), peak(0.02), peak(10.0));
    assert!(
        cold > warm && warm > hot,
        "colder should concentrate further: {cold} vs {warm} vs {hot}",
    );
    assert!(
        cold > 0.9,
        "a very cold softmax is nearly winner-take-all: {cold}"
    );
    assert_close_tol(
        &[hot],
        &[1.0 / N as f64],
        1e-3,
        "a very hot softmax is nearly equal weight",
    );
}

/// Buckets partition the cross-section: each stock lands in exactly one
/// decile-style slice, so disjoint buckets never overlap and covering ones
/// account for everything.
#[test]
fn rank_buckets_partition_the_cross_section() {
    let c = linear();
    let bottom = drive_mean(mean::rank_bucket(c, 0.0, 0.5), &ALL, &MU, &[true]);
    let top = drive_mean(mean::rank_bucket(c, 0.5, 1.0), &ALL, &MU, &[true]);

    for i in 0..N {
        let (lo, hi) = (bottom.emitted[0][i], top.emitted[0][i]);
        assert!(
            (lo > 0.0) != (hi > 0.0),
            "stock {i} is in neither bucket or both: {lo} and {hi}",
        );
    }

    // The bucket is defined on rank, so the negative prediction is held rather
    // than dropped — unlike every other mean portfolio.
    assert!(
        bottom.emitted[0][3] > 0.0,
        "the worst-predicted stock is in the bottom bucket: {:?}",
        bottom.emitted[0],
    );
}

/// A stock outside the universe, or one the predictor had no opinion on, holds
/// nothing — and the rest of the book still adds to one.
#[test]
fn stocks_without_a_universe_or_a_prediction_hold_nothing() {
    let partial = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let excluded = drive_mean(mean::proportional(linear()), &partial, &MU, &[true]);
    assert_eq!(excluded.emitted[0][2], 0.0, "out of universe");
    assert_eq!(excluded.emitted[0][4], 0.0, "out of universe");
    assert_close_tol(
        &[total(&excluded.emitted[0])],
        &[1.0],
        1e-12,
        "still invested",
    );

    let mut unpredicted = MU;
    unpredicted[1] = f64::NAN;
    let run = drive_mean(mean::proportional(linear()), &ALL, &unpredicted, &[true]);
    assert_eq!(run.emitted[0][1], 0.0, "no prediction, no position");
    assert_close_tol(&[total(&run.emitted[0])], &[1.0], 1e-12, "still invested");
}

/// A book is produced once per rebalance and held in between, so a portfolio
/// rebalancing more slowly than the graph ticks does not blank its positions
/// on the quiet generations.
#[test]
fn the_book_pulses_once_per_rebalance_and_is_retained_between() {
    let run = drive_mean(
        mean::proportional(linear()),
        &ALL,
        &MU,
        &[false, true, false, false],
    );

    assert_eq!(run.emitted.len(), 1, "one pulse from the single rebalance");
    assert_close(
        &run.held,
        &run.emitted[0],
        "the book survives two quiet generations",
    );
}

/// Before the first rebalance a portfolio holds nothing rather than something
/// arbitrary — the quiescent book the engine reads at build time.
#[test]
fn nothing_is_held_before_the_first_rebalance() {
    let run = drive_mean(mean::proportional(linear()), &ALL, &MU, &[false, false]);

    assert!(run.emitted.is_empty(), "no rebalance, no emission");
    assert_close(&run.held, &[0.0; N], "flat until the first rebalance");
}

/// The lognormal map is what stands between a predictor's log returns and an
/// optimizer's linear ones. With no covariance to contribute a variance term
/// it reduces to `expm1`, which is sharp enough to pin: the same predictions
/// produce demonstrably different books depending on the flag.
#[test]
fn the_lognormal_map_converts_moments_before_optimizing() {
    let as_written = drive_mean(mean::proportional(linear()), &ALL, &MU, &[true]);
    let as_logs = drive_mean(mean::proportional(Config::default()), &ALL, &MU, &[true]);

    let normalized = |xs: Vec<f64>| -> Vec<f64> {
        let sum: f64 = xs.iter().sum();
        xs.iter().map(|x| x / sum).collect()
    };
    assert_close(
        &as_written.emitted[0],
        &normalized(MU.iter().map(|m| m.max(0.0)).collect()),
        "linear moments pass through",
    );
    assert_close(
        &as_logs.emitted[0],
        &normalized(MU.iter().map(|m| m.exp_m1().max(0.0)).collect()),
        "log moments are converted",
    );
}

// ---------------------------------------------------------------------------
// Solving portfolios
// ---------------------------------------------------------------------------

/// The defining property of minimum variance: no other book in the feasible
/// set carries less predicted risk. Equal weight is a feasible book, so it is
/// a valid — and non-trivial — thing to beat.
#[test]
fn minimum_variance_carries_less_risk_than_equal_weight() {
    let sigma = covariance();
    let run = drive_variance(
        variance::minimum_variance(linear(), true, 4),
        &ALL,
        &sigma,
        &[true],
    );
    let book = &run.emitted[0];

    let equal = [1.0 / N as f64; N];
    let (optimized, baseline) = (risk(book, &sigma), risk(&equal, &sigma));
    assert!(
        optimized < baseline,
        "minimum variance should beat equal weight: {optimized} vs {baseline}",
    );
    assert_close_tol(&[total(book)], &[1.0], SOLVER_TOL, "fully invested");

    // The tilt is about factor exposure, not about each stock's own variance.
    // Every stock here loads positively on one factor, and long-only cannot
    // hedge that, so the heaviest loader is the one worth avoiding — stock 0,
    // even though stock 3 has the larger total variance. Stock 4 combines a
    // middling loading with the smallest idiosyncratic term and takes the most
    // weight.
    let equal = 1.0 / N as f64;
    assert!(
        book[0] < equal,
        "the heaviest factor loading should be underweighted: {book:?}",
    );
    assert!(
        book[4] > equal,
        "the lowest-variance stock should be overweighted: {book:?}",
    );
}

/// `markowitz` in mean-variance mode and `admm_mnr` solve the same
/// optimization by entirely different means — CVXPY canonicalizing to a cone
/// program, against a matrix-free ADMM outer loop over MINRES. Given the same
/// covariance approximation they must reach the same optimum, which is a far
/// stronger check on both than any single-solver assertion could be.
#[test]
fn the_two_mean_variance_solvers_agree_on_the_same_problem() {
    let sigma = covariance();
    let (delta, rank) = (5.0, 4);

    let cvxpy = drive_mean_variance(
        mean_variance::markowitz(
            linear(),
            mean_variance::Mode::MinMeanVariance,
            delta,
            true,
            true,
            rank,
        ),
        &ALL,
        &MU,
        &sigma,
        &[true],
    );
    let admm = drive_mean_variance(
        mean_variance::admm_mnr(linear(), delta, true, None, Some(rank)),
        &ALL,
        &MU,
        &sigma,
        &[true],
    );

    assert_close_tol(
        &admm.emitted[0],
        &cvxpy.emitted[0],
        SOLVER_TOL,
        "matrix-free ADMM vs CVXPY on the same problem",
    );
}

/// Risk aversion is a dial, and turning it up must actually buy less risk.
/// Comparing two solves of the same problem sidesteps needing a closed form
/// for either.
#[test]
fn a_higher_risk_aversion_buys_a_lower_variance_book() {
    let sigma = covariance();
    let book = |delta: f64| {
        drive_mean_variance(
            mean_variance::markowitz(
                linear(),
                mean_variance::Mode::MinMeanVariance,
                delta,
                true,
                true,
                4,
            ),
            &ALL,
            &MU,
            &sigma,
            &[true],
        )
        .emitted[0]
            .clone()
    };

    let (aggressive, cautious) = (book(0.5), book(50.0));
    assert!(
        risk(&cautious, &sigma) < risk(&aggressive, &sigma),
        "delta=50 should hold less risk than delta=0.5: {} vs {}",
        risk(&cautious, &sigma),
        risk(&aggressive, &sigma),
    );
    // And give up return for it — otherwise it found a free lunch, which on a
    // fixed set of moments would mean the aggressive solve was not optimal.
    let expected = |b: &[f64]| -> f64 { b.iter().zip(MU).map(|(w, m)| w * m).sum() };
    assert!(
        expected(&cautious) < expected(&aggressive),
        "less risk must cost return: {} vs {}",
        expected(&cautious),
        expected(&aggressive),
    );
}

/// A tracking-error budget bounds how far the book may stray from the
/// benchmark, so shrinking it must pull the book in. At zero there is no room
/// to deviate at all and the answer is the benchmark itself.
#[test]
fn a_tighter_tracking_error_budget_pulls_toward_the_benchmark() {
    let sigma = covariance();
    let benchmark = [1.0 / N as f64; N];

    let distance = |bound: f64| {
        let run = drive_mean_variance(
            mean_variance::benchmark_relative(linear(), bound, true, true, 4),
            &ALL,
            &MU,
            &sigma,
            &[true],
        );
        let book = run.emitted[0].clone();
        let deviation: Vec<f64> = book.iter().zip(benchmark).map(|(w, b)| w - b).collect();
        (risk(&deviation, &sigma).max(0.0).sqrt(), book)
    };

    let (loose, _) = distance(0.05);
    let (tight, _) = distance(0.005);
    assert!(
        tight < loose,
        "a tighter budget should track more closely: {tight} vs {loose}",
    );
    assert!(
        tight <= 0.005 + SOLVER_TOL,
        "the budget should bind: realized tracking error {tight}",
    );

    let (_, pinned) = distance(0.0);
    assert_close_tol(
        &pinned,
        &benchmark,
        1e-12,
        "no budget leaves nothing to do but hold the benchmark",
    );
}

/// A per-name cap is a hard constraint, not a preference: no weight may exceed
/// it, and a cap tight enough to bind must actually spread the book out.
#[test]
fn the_position_cap_bounds_every_weight() {
    let sigma = covariance();
    let uncapped = drive_mean_variance(
        mean_variance::admm_mnr(linear(), 5.0, true, None, Some(4)),
        &ALL,
        &MU,
        &sigma,
        &[true],
    );
    let peak = uncapped.emitted[0].iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        peak > 0.2,
        "the uncapped book concentrates somewhere: {peak}"
    );

    let cap = 0.2;
    let capped = drive_mean_variance(
        mean_variance::admm_mnr(linear(), 5.0, true, Some(cap), Some(4)),
        &ALL,
        &MU,
        &sigma,
        &[true],
    );
    for (i, &w) in capped.emitted[0].iter().enumerate() {
        assert!(
            w <= cap + SOLVER_TOL,
            "weight {w} at stock {i} exceeds the cap {cap}: {:?}",
            capped.emitted[0],
        );
    }
    assert_close_tol(
        &[total(&capped.emitted[0])],
        &[1.0],
        SOLVER_TOL,
        "still fully invested under the cap",
    );
}

/// The solving portfolios mask exactly as the closed-form ones do: a stock the
/// universe excludes, or one the predictors could not price, is left out of
/// the optimization rather than optimized around.
#[test]
fn the_solving_portfolios_exclude_unpriced_stocks() {
    let mut sigma = covariance();
    // Stock 4 has no predicted variance.
    sigma[4 * N + 4] = f64::NAN;
    let mut mu = MU;
    // Stock 2 has no predicted return.
    mu[2] = f64::NAN;
    // Stock 5 is outside the universe.
    let universe = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];

    let run = drive_mean_variance(
        mean_variance::markowitz(
            linear(),
            mean_variance::Mode::MinMeanVariance,
            5.0,
            true,
            true,
            3,
        ),
        &universe,
        &mu,
        &sigma,
        &[true],
    );
    let book = &run.emitted[0];

    assert_eq!(book[2], 0.0, "no predicted return, no position");
    assert_eq!(book[4], 0.0, "no predicted variance, no position");
    assert_eq!(book[5], 0.0, "out of universe, no position");
    assert_close_tol(
        &[total(book)],
        &[1.0],
        SOLVER_TOL,
        "the rest is fully invested",
    );
}
