//! Integration tests for `tradingflow::operators::metrics`: the
//! since-inception performance statistics.
//!
//! Four of the six (`compound_return`, `average_return`, `volatility`,
//! `sharpe_ratio`) are clock-gated — their inputs are `(clock, data)` and they
//! fold an observation only on a generation where the clock notified *and* the
//! leading data element is not NaN; otherwise they hold their previous output
//! and report `notify = false`. `drawdown` and `turnover` are single-input and
//! fold on every data update. Every output is a rank-0 scalar.
//!
//! Expected values are computed inline from the price path wherever the
//! statistic has a closed form, so the tests pin the *formula* (population vs
//! sample dispersion, geometric vs arithmetic compounding) rather than a
//! transcribed constant.

use tradingflow::data::Instant;
use tradingflow::graph::typed::Builder;
use tradingflow::graph::{Pool, Segment};
use tradingflow::operators::{array, metrics};
use tradingflow::ports::{ArrayPort, UnitPort};

use crate::harness::*;

// ---------------------------------------------------------------------------
// Wiring helpers
// ---------------------------------------------------------------------------

/// Drives a clock-gated scalar metric over `path`, pulsing the clock exactly
/// once per sample, and returns the metric's output after every generation.
fn gated<S>(metric: S, path: &[f64]) -> Vec<f64>
where
    S: Segment<
            Inputs = (UnitPort, ArrayPort<f64, 0>),
            Outputs = ArrayPort<f64, 0>,
            Context = Instant,
        >,
{
    let mut b = Builder::new();
    let (data, datav) = b.source(array::scalar(0.0_f64));
    let (tick, tickv) = b.source(const_val(()));
    let out = b.segment(metric, (tickv, datav));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut seen = Vec::with_capacity(path.len());
    for (i, &p) in path.iter().enumerate() {
        *g.state_mut(data) = scalar(p);
        let _ = g.state_mut(tick);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        seen.push(vals(g.view(out))[0]);
    }
    seen
}

/// Drives an ungated rank-0 metric (`drawdown`) over `path`, one sample per
/// generation, and returns its output after every generation.
fn ungated<S>(metric: S, path: &[f64]) -> Vec<f64>
where
    S: Segment<Inputs = ArrayPort<f64, 0>, Outputs = ArrayPort<f64, 0>, Context = Instant>,
{
    let mut b = Builder::new();
    let (data, datav) = b.source(array::scalar(0.0_f64));
    let out = b.segment(metric, datav);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut seen = Vec::with_capacity(path.len());
    for (i, &p) in path.iter().enumerate() {
        *g.state_mut(data) = scalar(p);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        seen.push(vals(g.view(out))[0]);
    }
    seen
}

/// Drives `turnover` over a sequence of equal-width weight vectors, one
/// rebalance per generation, and returns its output after every generation.
fn turnover_path(books: &[Vec<f64>]) -> Vec<f64> {
    let width = books[0].len();
    let mut b = Builder::new();
    let (cell, w) = b.source(array::zeros::<f64, 1>([width]));
    let out = b.segment(metrics::turnover(), w);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut seen = Vec::with_capacity(books.len());
    for (i, book) in books.iter().enumerate() {
        assert_eq!(book.len(), width, "the universe width must stay fixed");
        *g.state_mut(cell) = arr1(book.clone());
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        seen.push(vals(g.view(out))[0]);
    }
    seen
}

// ---------------------------------------------------------------------------
// Reference statistics
// ---------------------------------------------------------------------------

/// A deterministic strictly-positive price path: `quarter_path` lands in
/// `[0, 250)` and a non-positive price sends the metrics down a degenerate
/// branch, so shift it clear of zero.
fn positive_path(len: usize) -> Vec<f64> {
    quarter_path(20250725, len)
        .iter()
        .map(|v| v + 100.0)
        .collect()
}

/// The simple per-period returns of a price path — the series every gated
/// metric folds.
fn returns(path: &[f64]) -> Vec<f64> {
    path.windows(2).map(|w| w[1] / w[0] - 1.0).collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Population standard deviation (divides by `n`) — what the operators compute.
fn pop_std(xs: &[f64]) -> f64 {
    let m = mean(xs);
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Sample standard deviation (divides by `n - 1`) — what they do *not* compute.
fn sample_std(xs: &[f64]) -> f64 {
    let m = mean(xs);
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

/// Mean over population deviation, NaN where there is no dispersion.
fn sharpe(xs: &[f64]) -> f64 {
    let s = pop_std(xs);
    if s > 0.0 { mean(xs) / s } else { f64::NAN }
}

/// The expected emission sequence of a return-folding metric over `path`: the
/// first tick is an observation but not yet a return (so it emits `warm`), and
/// tick `k` reports `stat` over the first `k` returns.
fn expected_seq(path: &[f64], warm: f64, stat: fn(&[f64]) -> f64) -> Vec<f64> {
    let rets = returns(path);
    std::iter::once(warm)
        .chain((1..path.len()).map(|k| stat(&rets[..k])))
        .collect()
}

/// A non-finite weight is zero exposure, matching `turnover`'s cleaning.
fn clean(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

/// L1 distance between two weight vectors, NaN treated as zero.
fn l1(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (clean(y) - clean(x)).abs())
        .sum()
}

// ---------------------------------------------------------------------------
// compound_return
// ---------------------------------------------------------------------------

/// `compound_return` reports the *per-tick* geometric growth rate since
/// inception, `(current / first)^(1 / periods) - 1`, where `periods` is one
/// less than the number of observations folded so far.
#[test]
fn compound_return_is_the_per_tick_geometric_growth_rate() {
    let path = positive_path(10);
    let expected: Vec<f64> = (0..path.len())
        .map(|k| {
            if k == 0 {
                0.0
            } else {
                (path[k] / path[0]).powf(1.0 / k as f64) - 1.0
            }
        })
        .collect();
    assert_close(
        &gated(metrics::compound_return(), &path),
        &expected,
        "compound return",
    );
}

/// The compounding is multiplicative, not additive: +10% then -10% ends 1%
/// below the start, so the growth rate is `sqrt(0.99) - 1 < 0` — an arithmetic
/// average of the two moves would call the path flat.
#[test]
fn compound_return_compounds_multiplicatively() {
    let out = gated(metrics::compound_return(), &[100.0, 110.0, 99.0]);
    assert_close(
        &out,
        &[0.0, 0.1, 0.99_f64.sqrt() - 1.0],
        "compound return over a round trip",
    );

    let rate = out[2];
    assert!(rate < 0.0, "a -1% round trip reported a non-negative rate");
    // Compounding the reported rate over the two periods reproduces the path.
    assert_close(
        &[100.0 * (1.0 + rate) * (1.0 + rate)],
        &[99.0],
        "recompounded path end",
    );
}

/// A non-positive observation makes the rate NaN for that tick only: the tick
/// still counts as a period, so once the path is positive again the exponent
/// reflects every observation folded, including the bad one.
#[test]
fn compound_return_is_nan_on_a_non_positive_tick_then_recovers() {
    let out = gated(metrics::compound_return(), &[100.0, 0.0, 110.0]);
    assert_close(
        &out,
        &[0.0, f64::NAN, 1.1_f64.powf(1.0 / 2.0) - 1.0],
        "compound return across a zero price",
    );
}

/// A non-positive *first* observation poisons the metric permanently: the first
/// price is latched as the denominator on the first tick and never revisited,
/// so every later tick reports NaN however the path recovers.
#[test]
fn a_non_positive_first_price_poisons_compound_return_forever() {
    let out = gated(metrics::compound_return(), &[0.0, 100.0, 110.0]);
    assert_close(
        &out,
        &[0.0, f64::NAN, f64::NAN],
        "compound return anchored at zero",
    );
}

// ---------------------------------------------------------------------------
// average_return
// ---------------------------------------------------------------------------

/// `average_return` is the arithmetic mean of the simple per-period returns
/// folded so far — checked against an inline reference at every tick.
#[test]
fn average_return_is_the_running_mean_of_period_returns() {
    let path = positive_path(10);
    let expected = expected_seq(&path, f64::NAN, mean);
    assert_close(
        &gated(metrics::average_return(), &path),
        &expected,
        "average return",
    );
}

/// The mean is arithmetic, so +10% followed by -10% averages to exactly zero
/// even though the path lost money — the complement of
/// `compound_return_compounds_multiplicatively`.
#[test]
fn average_return_of_a_symmetric_round_trip_is_zero() {
    let out = gated(metrics::average_return(), &[100.0, 110.0, 99.0]);
    assert_close(
        &out,
        &[f64::NAN, 0.1, 0.0],
        "average return over a round trip",
    );
}

/// Non-positive prices are handled asymmetrically: the period that *falls* to
/// zero folds a -100% return, but the period that *leaves* a non-positive price
/// is dropped entirely (the fold is gated on `prev > 0`), so the recovery never
/// enters the mean.
#[test]
fn average_return_drops_the_period_leaving_a_non_positive_price() {
    let out = gated(metrics::average_return(), &[100.0, 110.0, 0.0, 50.0]);
    let after_crash = mean(&[0.1, -1.0]);
    assert_close(
        &out,
        &[f64::NAN, 0.1, after_crash, after_crash],
        "average return across a zero price",
    );
}

// ---------------------------------------------------------------------------
// volatility
// ---------------------------------------------------------------------------

/// One return has no dispersion around its own mean, so the first computable
/// volatility is exactly zero — not NaN, and not a cancellation epsilon.
#[test]
fn volatility_of_a_single_return_is_zero() {
    let out = gated(metrics::volatility(), &[100.0, 110.0]);
    assert!(out[0].is_nan(), "warm-up should read NaN, got {}", out[0]);
    assert_eq!(out[1], 0.0, "a single return must have exactly zero spread");
}

/// `volatility` is the *population* standard deviation of the returns (divides
/// by `n`), and it is raw per-period — no annualization factor is applied.
#[test]
fn volatility_is_the_population_standard_deviation() {
    let path = positive_path(10);
    let expected = expected_seq(&path, f64::NAN, pop_std);
    let out = gated(metrics::volatility(), &path);
    assert_close(&out, &expected, "volatility");

    let rets = returns(&path);
    let last = out[out.len() - 1];
    assert!(
        (last - sample_std(&rets)).abs() > 1e-3,
        "volatility {last} matches the sample (n-1) deviation {}",
        sample_std(&rets)
    );
    assert!(
        (last - pop_std(&rets) * 252.0_f64.sqrt()).abs() > 1e-3,
        "volatility {last} looks annualized"
    );
}

/// A path growing at a constant rate has zero return dispersion: the variance
/// is recovered from running sums, and that cancellation must still land on
/// exactly zero rather than a small positive residue.
#[test]
fn volatility_of_a_constant_growth_path_is_zero() {
    // Exact binary fractions: every return is exactly 0.25.
    let out = gated(metrics::volatility(), &[100.0, 125.0, 156.25, 195.3125]);
    assert!(out[0].is_nan(), "warm-up should read NaN, got {}", out[0]);
    assert_eq!(
        &out[1..],
        &[0.0, 0.0, 0.0],
        "a constant growth rate must have zero volatility"
    );
}

// ---------------------------------------------------------------------------
// sharpe_ratio
// ---------------------------------------------------------------------------

/// `sharpe_ratio` is the mean return over the population standard deviation of
/// the returns — no risk-free rate, no annualization.
#[test]
fn sharpe_ratio_is_the_mean_over_the_population_deviation() {
    let path = positive_path(10);
    let expected = expected_seq(&path, f64::NAN, sharpe);
    let out = gated(metrics::sharpe_ratio(), &path);
    assert_close(&out, &expected, "sharpe ratio");

    let rets = returns(&path);
    let last = out[out.len() - 1];
    assert!(
        (last - mean(&rets) / sample_std(&rets)).abs() > 1e-3,
        "sharpe {last} used the sample (n-1) deviation"
    );
}

/// Zero volatility yields NaN, not zero and not infinity: a steadily
/// compounding path with a strictly positive mean return reports no Sharpe
/// ratio at all, and neither does the single-return warm-up case.
#[test]
fn sharpe_ratio_is_nan_when_volatility_is_zero() {
    let path = [100.0, 125.0, 156.25, 195.3125];
    let out = gated(metrics::sharpe_ratio(), &path);
    assert!(
        mean(&returns(&path)) > 0.0,
        "the fixture path must have a positive mean return"
    );
    assert!(
        out.iter().all(|v| v.is_nan()),
        "zero volatility must give NaN, got {out:?}"
    );
}

// ---------------------------------------------------------------------------
// drawdown
// ---------------------------------------------------------------------------

/// Before any sample `drawdown` reads exactly zero — unlike the gated metrics,
/// whose warm-up value is NaN.
#[test]
fn drawdown_before_any_sample_is_zero() {
    let mut b = Builder::new();
    let (_data, datav) = b.source(array::scalar(0.0_f64));
    let dd = b.segment(metrics::drawdown(), datav);
    let g = b.build();
    assert_eq!(vals(g.view(dd)), &[0.0], "warm-up drawdown");
}

/// Every new high is exactly zero drawdown, whether it is the first sample or a
/// later peak.
#[test]
fn drawdown_at_a_new_high_is_zero() {
    let out = ungated(metrics::drawdown(), &[100.0, 120.0, 90.0, 130.0]);
    assert_eq!(out[0], 0.0, "the first sample is its own peak");
    assert_eq!(out[1], 0.0, "a new high is a zero drawdown");
    assert_eq!(out[3], 0.0, "a new high after a trough is a zero drawdown");
}

/// The peak is a running *maximum*: it never resets on a recovery, so a partial
/// rebound is still measured against the old high.
#[test]
fn drawdown_running_max_does_not_reset_on_recovery() {
    let path = [100.0, 120.0, 90.0, 100.0, 110.0];
    let out = ungated(metrics::drawdown(), &path);
    let expected: Vec<f64> = {
        let mut peak = f64::MIN;
        path.iter()
            .map(|&p| {
                peak = peak.max(p);
                (p - peak) / peak
            })
            .collect()
    };
    assert_close(&out, &expected, "drawdown");
    assert!(
        out.iter().all(|&v| v <= 0.0),
        "drawdown must be non-positive, got {out:?}"
    );
}

/// A NaN sample is not an observation: the operator holds its previous output
/// and leaves the running peak untouched.
#[test]
fn drawdown_holds_its_value_through_nan() {
    let out = ungated(metrics::drawdown(), &[100.0, 80.0, f64::NAN, 90.0]);
    assert_close(
        &out,
        &[0.0, -0.2, -0.2, -0.1],
        "drawdown across a missing sample",
    );
}

// ---------------------------------------------------------------------------
// turnover
// ---------------------------------------------------------------------------

/// `turnover` emits the L1 change of the weight vector between consecutive
/// rebalances; the first rebalance is warm-up and leaves the output NaN.
#[test]
fn turnover_is_the_l1_change_between_rebalances() {
    let books = vec![
        vec![0.5, 0.3, 0.2],
        vec![0.2, 0.3, 0.5],
        vec![0.4, 0.4, 0.2],
        vec![0.1, 0.1, 0.8],
    ];
    let expected: Vec<f64> = std::iter::once(f64::NAN)
        .chain(books.windows(2).map(|w| l1(&w[0], &w[1])))
        .collect();
    assert_close(&turnover_path(&books), &expected, "turnover");
}

/// Re-publishing the same book trades nothing.
#[test]
fn turnover_of_an_unchanged_book_is_zero() {
    let book = vec![0.4, 0.35, 0.25];
    let out = turnover_path(&[book.clone(), book.clone(), book]);
    assert_close(&out, &[f64::NAN, 0.0, 0.0], "turnover of a static book");
}

/// Rotating the whole book out of one name and into another moves 100% of the
/// portfolio, which the L1 convention scores as 2.0.
#[test]
fn turnover_of_a_full_rotation_is_two() {
    let out = turnover_path(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_close(&out, &[f64::NAN, 2.0], "turnover of a full rotation");
}

/// Names entering and leaving the universe are marked by NaN weights, which
/// count as zero exposure — a name leaving contributes its whole old weight and
/// a name entering its whole new weight.
#[test]
fn turnover_treats_missing_weights_as_zero_exposure() {
    let books = vec![
        vec![0.5, 0.5, f64::NAN], // the third name is not in the universe yet
        vec![0.5, f64::NAN, 0.5], // the second leaves, the third enters
        vec![0.5, 0.5, 0.5],      // the second re-enters
    ];
    // |0.5-0.5| + |0-0.5| + |0.5-0| = 1.0, then |0| + |0.5-0| + |0.5-0.5| = 0.5.
    let out = turnover_path(&books);
    assert_close(
        &out,
        &[f64::NAN, 1.0, 0.5],
        "turnover across universe changes",
    );
    let expected: Vec<f64> = std::iter::once(f64::NAN)
        .chain(books.windows(2).map(|w| l1(&w[0], &w[1])))
        .collect();
    assert_close(&out, &expected, "turnover reference");
}

/// The warm-up rebalance does not notify, so nothing downstream of `turnover`
/// recomputes until there is an actual change to report.
#[test]
fn turnover_does_not_notify_on_the_warm_up_rebalance() {
    let mut b = Builder::new();
    let (cell, w) = b.source(array::zeros::<f64, 1>([3]));
    let out = b.segment(metrics::turnover(), w);
    let folds = b.segment(count::<0>(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(cell) = arr1(vec![0.4, 0.3, 0.3]);
    g.stabilize(&mut pool, &nano(1));
    assert_eq!(g.view(folds), 0, "the warm-up rebalance notified");
    assert!(vals(g.view(out))[0].is_nan(), "the warm-up emitted a value");

    *g.state_mut(cell) = arr1(vec![0.3, 0.3, 0.4]);
    g.stabilize(&mut pool, &nano(2));
    assert_eq!(g.view(folds), 1, "the second rebalance did not notify");
    assert_close(&vals(g.view(out)), &[0.2], "turnover");
}

// ---------------------------------------------------------------------------
// The shared clock gate
// ---------------------------------------------------------------------------

/// Wires all four gated metrics onto one `(clock, data)` pair, each behind a
/// recompute counter: `$ms` are the metric ports (compound return, average
/// return, volatility, Sharpe ratio) and `$folds` their counters.
macro_rules! gated_bench {
    ($b:ident, $data:ident, $tick:ident, $ms:ident, $folds:ident) => {
        let mut $b = Builder::new();
        let ($data, datav) = $b.source(array::scalar(0.0_f64));
        let ($tick, tickv) = $b.source(const_val(()));
        let $ms = [
            $b.segment(metrics::compound_return(), (tickv, datav)),
            $b.segment(metrics::average_return(), (tickv, datav)),
            $b.segment(metrics::volatility(), (tickv, datav)),
            $b.segment(metrics::sharpe_ratio(), (tickv, datav)),
        ];
        let $folds = $ms.map(|m| $b.segment(count::<0>(), m));
    };
}

/// The current value of each metric port in `$ms`.
macro_rules! metric_values {
    ($g:expr, $ms:expr) => {
        $ms.iter()
            .map(|&m| vals($g.view(m))[0])
            .collect::<Vec<f64>>()
    };
}

/// How many generations each metric has notified for.
macro_rules! fold_counts {
    ($g:expr, $folds:expr) => {
        $folds.iter().map(|&c| $g.view(c)).collect::<Vec<usize>>()
    };
}

/// The shared clock gate: a generation that changes the data without pulsing
/// the clock folds no observation — the metrics hold their values and do not
/// notify. The converse also holds: a clock pulse folds an observation even
/// when the data is unchanged, because the gate is the clock, not the data.
#[test]
fn an_off_clock_data_change_does_not_fold_an_observation() {
    gated_bench!(b, data, tick, ms, folds);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Two clocked observations: 100 then 110.
    for (i, p) in [100.0, 110.0].into_iter().enumerate() {
        *g.state_mut(data) = scalar(p);
        let _ = g.state_mut(tick);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }
    assert_eq!(fold_counts!(g, folds), [2, 2, 2, 2], "one fold per pulse");
    let held = metric_values!(g, ms);
    assert_close(
        &held,
        &[0.1, 0.1, 0.0, f64::NAN],
        "metrics after two clocked observations",
    );

    // A data change with no clock pulse: nothing folds, nothing notifies.
    *g.state_mut(data) = scalar(132.0);
    g.stabilize(&mut pool, &nano(3));
    assert_eq!(
        fold_counts!(g, folds),
        [2, 2, 2, 2],
        "an off-clock data change notified downstream"
    );
    assert_close(
        &metric_values!(g, ms),
        &held,
        "an off-clock data change folded",
    );

    // The next pulse folds the value the data has been holding all along.
    let _ = g.state_mut(tick);
    g.stabilize(&mut pool, &nano(4));
    assert_eq!(
        fold_counts!(g, folds),
        [3, 3, 3, 3],
        "the clock pulse did not fold"
    );
    let rets = [0.1, 0.2];
    assert_close(
        &metric_values!(g, ms),
        &[
            1.32_f64.powf(1.0 / 2.0) - 1.0,
            mean(&rets),
            pop_std(&rets),
            sharpe(&rets),
        ],
        "metrics after the third observation",
    );

    // A pulse without a data change is still an observation — a zero return.
    let _ = g.state_mut(tick);
    g.stabilize(&mut pool, &nano(5));
    assert_eq!(
        fold_counts!(g, folds),
        [4, 4, 4, 4],
        "a bare clock pulse did not fold"
    );
    let rets = [0.1, 0.2, 0.0];
    assert_close(
        &metric_values!(g, ms),
        &[
            1.32_f64.powf(1.0 / 3.0) - 1.0,
            mean(&rets),
            pop_std(&rets),
            sharpe(&rets),
        ],
        "metrics after a bare clock pulse",
    );
}

/// A clock pulse carrying NaN data is not an observation either: the metrics
/// hold and do not notify, and — because the NaN never becomes the previous
/// price — the next real sample's return is measured against the last good one.
#[test]
fn a_clock_tick_with_nan_data_is_not_an_observation() {
    gated_bench!(b, data, tick, ms, folds);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, p) in [100.0, 110.0, f64::NAN, 132.0].into_iter().enumerate() {
        *g.state_mut(data) = scalar(p);
        let _ = g.state_mut(tick);
        g.stabilize(&mut pool, &nano(i as i64 + 1));

        if p.is_nan() {
            assert_eq!(
                fold_counts!(g, folds),
                [2, 2, 2, 2],
                "a NaN observation notified"
            );
            assert_close(
                &metric_values!(g, ms),
                &[0.1, 0.1, 0.0, f64::NAN],
                "a NaN observation folded",
            );
        }
    }

    assert_eq!(
        fold_counts!(g, folds),
        [3, 3, 3, 3],
        "three real observations should have folded"
    );
    // 132 / 110 - 1 = 0.2: the return skipped over the NaN rather than being
    // measured against it (which would have dropped the period entirely).
    let rets = [0.1, 0.2];
    assert_close(
        &metric_values!(g, ms),
        &[
            1.32_f64.powf(1.0 / 2.0) - 1.0,
            mean(&rets),
            pop_std(&rets),
            sharpe(&rets),
        ],
        "metrics after a missing sample",
    );
}

/// Warm-up: every gated metric reads NaN before its first tick. The first tick
/// is an observation — it notifies — but it is not yet a *return*, so only
/// `compound_return` (which measures against the first price rather than the
/// previous one) has anything to report, and it reports exactly zero.
#[test]
fn gated_metrics_are_nan_until_their_first_return() {
    gated_bench!(b, data, tick, ms, folds);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    assert_close(
        &metric_values!(g, ms),
        &[f64::NAN; 4],
        "metrics before the first tick",
    );
    assert_eq!(
        fold_counts!(g, folds),
        [0, 0, 0, 0],
        "nothing folded at build"
    );

    *g.state_mut(data) = scalar(100.0);
    let _ = g.state_mut(tick);
    g.stabilize(&mut pool, &nano(1));
    assert_eq!(
        fold_counts!(g, folds),
        [1, 1, 1, 1],
        "the first observation must notify even where the value stays NaN"
    );
    assert_close(
        &metric_values!(g, ms),
        &[0.0, f64::NAN, f64::NAN, f64::NAN],
        "metrics after one observation",
    );
}
