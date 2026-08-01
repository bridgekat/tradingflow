//! Integration tests for `tradingflow::operators::metric`: the
//! since-inception performance statistics.
//!
//! The `return` family (`comp_return`, `return_mean`, `return_vol`,
//! `return_sharpe` and their `log_` counterparts) shares one driver: inputs are
//! `(signal, nav)`, a period is closed by a signal, and the net asset value
//! is required to be finite and strictly positive. A generation without a pulse
//! folds nothing, holds the previous output and reports `notify = false`. The
//! first pulse only latches the opening level — no period has closed yet — and
//! emits exactly zero. `drawdown` and `turnover` are single-input and fold on
//! every update. Every output is a rank-0 scalar.
//!
//! Expected values are computed inline from the price path wherever the
//! statistic has a closed form, so the tests pin the *formula* (population vs
//! sample dispersion, geometric vs arithmetic compounding) rather than a
//! transcribed constant.

use tradingflow::data::Instant;
use tradingflow::graph::typed::Builder;
use tradingflow::graph::{Pool, Segment};
use tradingflow::operators::{array, metric};
use tradingflow::ports::{ArrayPort, SignalPort};

use crate::harness::*;

// ---------------------------------------------------------------------------
// Wiring helpers
// ---------------------------------------------------------------------------

/// Drives a signal-gated scalar metric over `path`, pulsing the signal exactly
/// once per sample, and returns the metric's output after every generation.
fn gated<S>(metric: S, path: &[f64]) -> Vec<f64>
where
    S: Segment<
            Inputs = (SignalPort<0>, ArrayPort<f64, 0>),
            Outputs = ArrayPort<f64, 0>,
            Context = Instant,
        >,
{
    let mut b = Builder::new();
    let (data, datav) = b.source(array::scalar(0.0_f64));
    let (tick, tickv) = b.source(signal());
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

/// Drives `turnover` over a sequence of equal-width weight vectors, one
/// rebalance per generation, and returns its output after every generation.
fn turnover_path(books: &[Vec<f64>]) -> Vec<f64> {
    let width = books[0].len();
    let mut b = Builder::new();
    let (cell, w) = event_src(&mut b, arr1(vec![0.0_f64; width]));
    let out = b.segment(metric::portfolio::turnover(), w);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut seen = Vec::with_capacity(books.len());
    for (i, book) in books.iter().enumerate() {
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
/// `[0, 250)` and a non-positive price is a precondition violation, so shift it
/// clear of zero.
fn positive_path(len: usize) -> Vec<f64> {
    quarter_path(20250725, len)
        .iter()
        .map(|v| v + 100.0)
        .collect()
}

/// A path compounding at exactly `rate` per period.
fn geometric_path(start: f64, rate: f64, len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    let mut v = start;
    for _ in 0..len {
        out.push(v);
        v *= 1.0 + rate;
    }
    out
}

/// The simple per-period returns of a price path.
fn returns(path: &[f64]) -> Vec<f64> {
    path.windows(2).map(|w| w[1] / w[0] - 1.0).collect()
}

/// The log per-period returns, spelled as the operator spells them
/// (`ln(v) - ln(prev)`, not `ln(v / prev)`).
fn log_returns(path: &[f64]) -> Vec<f64> {
    path.windows(2).map(|w| w[1].ln() - w[0].ln()).collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Population standard deviation (divides by `n`) — what the operators compute.
fn pop_std(xs: &[f64]) -> f64 {
    let m = mean(xs);
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Sample standard deviation (divides by `n - 1`) — what they do *not*.
fn sample_std(xs: &[f64]) -> f64 {
    let m = mean(xs);
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

/// The expected emission sequence of a gated metric: zero on the opening
/// pulse, then `stat` over the returns accumulated so far.
fn expected_seq(
    path: &[f64],
    rets: impl Fn(&[f64]) -> Vec<f64>,
    stat: impl Fn(&[f64]) -> f64,
) -> Vec<f64> {
    let mut out = vec![0.0];
    for k in 2..=path.len() {
        out.push(stat(&rets(&path[..k])));
    }
    out
}

// ---------------------------------------------------------------------------
// comp_return
// ---------------------------------------------------------------------------

/// `comp_return` is the *per-period* geometric mean return, not the cumulative
/// one: it compounds `1 + r` across the window and takes the `count`-th root,
/// which telescopes to `(last / first)^(1/n) - 1`.
#[test]
fn comp_return_is_the_per_period_geometric_mean() {
    let path = positive_path(12);
    let out = gated(metric::performance::comp_return(), &path);

    let expected = expected_seq(&path, returns, |rs| {
        rs.iter()
            .map(|r| 1.0 + r)
            .product::<f64>()
            .powf(1.0 / rs.len() as f64)
            - 1.0
    });
    assert_close(&out, &expected, "comp_return");

    // The telescoped closed form, which is what makes it a growth *rate*.
    for k in 2..=path.len() {
        let n = (k - 1) as f64;
        let closed = (path[k - 1] / path[0]).powf(1.0 / n) - 1.0;
        assert_close(&[out[k - 1]], &[closed], &format!("closed form at {k}"));
    }
}

/// Compounding a constant rate reproduces that rate at every step — the
/// defining property of a geometric mean.
#[test]
fn comp_return_of_a_constant_rate_is_that_rate() {
    let path = geometric_path(100.0, 0.25, 6);
    let out = gated(metric::performance::comp_return(), &path);

    assert_eq!(out[0], 0.0, "the opening pulse closes no period");
    assert_close(&out[1..], &[0.25; 5], "constant growth rate");
}

/// A symmetric round trip is a *loss* geometrically (+10% then −10% ends below
/// par), which is exactly where the geometric and arithmetic means part ways.
#[test]
fn comp_return_of_a_round_trip_is_negative_where_the_mean_is_zero() {
    let path = [100.0, 110.0, 99.0];
    let geo = gated(metric::performance::comp_return(), &path);
    let ari = gated(metric::performance::return_mean(), &path);

    assert_close(&[geo[2]], &[0.99_f64.sqrt() - 1.0], "geometric");
    assert!(geo[2] < 0.0, "compounding a round trip loses");
    assert_close(&[ari[2]], &[0.0], "arithmetic");
}

// ---------------------------------------------------------------------------
// return_mean / log_return_mean
// ---------------------------------------------------------------------------

/// `return_mean` is the running arithmetic mean of the simple per-period
/// returns.
#[test]
fn return_mean_is_the_running_mean_of_simple_returns() {
    let path = positive_path(12);
    let out = gated(metric::performance::return_mean(), &path);
    assert_close(&out, &expected_seq(&path, returns, mean), "return_mean");
}

/// `log_return_mean` averages `ln(v) - ln(prev)`, which telescopes: the mean
/// log return depends only on the endpoints and the period count.
#[test]
fn log_return_mean_telescopes_to_the_endpoints() {
    let path = positive_path(12);
    let out = gated(metric::performance::log_return_mean(), &path);
    assert_close(
        &out,
        &expected_seq(&path, log_returns, mean),
        "log_return_mean",
    );

    for k in 2..=path.len() {
        let n = (k - 1) as f64;
        let closed = (path[k - 1].ln() - path[0].ln()) / n;
        assert_close(&[out[k - 1]], &[closed], &format!("telescoped at {k}"));
    }
}

/// The mean log return is the log of one plus the geometric mean return — the
/// bridge between the two compounding conventions.
#[test]
fn log_return_mean_is_the_log_of_the_geometric_mean() {
    let path = positive_path(10);
    let log = gated(metric::performance::log_return_mean(), &path);
    let geo = gated(metric::performance::comp_return(), &path);

    for k in 2..=path.len() {
        assert_close(
            &[log[k - 1]],
            &[(1.0 + geo[k - 1]).ln()],
            &format!("bridge at {k}"),
        );
    }
}

// ---------------------------------------------------------------------------
// return_vol / log_return_vol
// ---------------------------------------------------------------------------

/// Volatility is the *population* standard deviation of the returns, and is
/// not annualized.
#[test]
fn return_vol_is_the_population_standard_deviation() {
    let path = positive_path(12);
    let out = gated(metric::performance::return_vol(), &path);
    assert_close(&out, &expected_seq(&path, returns, pop_std), "return_vol");

    // Explicitly *not* the sample deviation, and not scaled by sqrt(252).
    let rs = returns(&path);
    let last = out[path.len() - 1];
    assert!(
        (last - sample_std(&rs)).abs() > 1e-4,
        "population and sample deviation must be distinguishable here"
    );
    assert!(
        (last - pop_std(&rs) * 252.0_f64.sqrt()).abs() > 1e-4,
        "must not be annualized"
    );
}

/// `log_return_vol` is the same dispersion over log returns.
#[test]
fn log_return_vol_is_the_population_deviation_of_log_returns() {
    let path = positive_path(12);
    let out = gated(metric::performance::log_return_vol(), &path);
    assert_close(
        &out,
        &expected_seq(&path, log_returns, pop_std),
        "log_return_vol",
    );
}

/// A riskless path reports zero risk, never undefined risk.
///
/// A rate of 1.25 is exactly representable in binary, so every return really is
/// `0.25` and the variance cancels to a true zero.
#[test]
fn return_vol_of_an_exactly_riskless_path_is_zero() {
    let path = geometric_path(100.0, 0.25, 6);
    for (k, &v) in gated(metric::performance::return_vol(), &path)
        .iter()
        .enumerate()
    {
        assert_eq!(v, 0.0, "vol at pulse {k} is {v}");
    }
}

/// The same for a rate that is *not* exactly representable — the regression
/// guard for the variance clamp.
///
/// The accumulator computes `E[r^2] - E[r]^2` from running sums. When the
/// returns agree only to within rounding, that subtraction cancels
/// catastrophically: at +10% over four periods it lands about `-1.7e-18`,
/// i.e. below zero, and an unclamped square root would make a perfectly steady
/// strategy report NaN risk. Clamping turns that into zero. The residue is
/// bounded by the cancellation itself (order `sqrt(eps) * |mean|`), so this
/// asserts a tight neighbourhood of zero rather than exact equality — the
/// one-pass formula cannot promise more, and `rolling`'s variance shares the
/// same characteristic.
#[test]
fn return_vol_of_a_nearly_riskless_path_is_never_nan() {
    for &(rate, len) in &[(0.1, 5), (0.1, 9), (0.01, 11), (0.07, 8)] {
        let path = geometric_path(100.0, rate, len);
        // Cancelling two quantities of size `mean^2` loses half the mantissa,
        // so the standard deviation inherits an absolute error of order
        // `sqrt(eps) * |mean|`; the factor leaves room for a few ULP.
        let bound = 16.0 * f64::EPSILON.sqrt() * rate.abs();
        for (op, out) in [
            ("vol", gated(metric::performance::return_vol(), &path)),
            (
                "log vol",
                gated(metric::performance::log_return_vol(), &path),
            ),
        ] {
            for (k, &v) in out.iter().enumerate() {
                assert!(
                    !v.is_nan(),
                    "rate {rate} len {len}: {op} at pulse {k} is NaN"
                );
                assert!(
                    (0.0..bound).contains(&v),
                    "rate {rate} len {len}: {op} at pulse {k} is {v}, bound {bound}"
                );
            }
        }
    }
}

/// A single closed period has no dispersion, so volatility is exactly zero
/// rather than NaN.
#[test]
fn return_vol_of_a_single_period_is_zero() {
    assert_eq!(
        gated(metric::performance::return_vol(), &[100.0, 110.0]),
        vec![0.0, 0.0]
    );
}

// ---------------------------------------------------------------------------
// return_sharpe / log_return_sharpe
// ---------------------------------------------------------------------------

/// The Sharpe ratio is the mean return over the population deviation — and,
/// because both share the same clamped variance, it is exactly the quotient of
/// the `return_mean` and `return_vol` operators on the same path.
#[test]
fn return_sharpe_is_mean_over_population_deviation() {
    let path = positive_path(12);
    let out = gated(metric::performance::return_sharpe(), &path);
    let expected = expected_seq(&path, returns, |rs| mean(rs) / pop_std(rs));
    assert_close(&out, &expected, "return_sharpe");

    let means = gated(metric::performance::return_mean(), &path);
    let vols = gated(metric::performance::return_vol(), &path);
    for k in 2..=path.len() {
        assert_close(
            &[out[k - 1]],
            &[means[k - 1] / vols[k - 1]],
            &format!("quotient identity at {k}"),
        );
    }
}

/// `log_return_sharpe` is the same ratio over log returns.
#[test]
fn log_return_sharpe_uses_log_returns() {
    let path = positive_path(12);
    let out = gated(metric::performance::log_return_sharpe(), &path);
    let expected = expected_seq(&path, log_returns, |rs| mean(rs) / pop_std(rs));
    assert_close(&out, &expected, "log_return_sharpe");
}

/// A riskless path divides a real mean by a true zero, so the ratio is the
/// signed infinity that is its limit rather than NaN. A single closed period is
/// riskless by construction, so this is reached immediately.
///
/// The rates are exactly representable in binary, so the variance is a true
/// zero rather than the rounding residue of the previous test.
#[test]
fn return_sharpe_of_a_riskless_path_is_infinite() {
    let up = gated(
        metric::performance::return_sharpe(),
        &geometric_path(100.0, 0.25, 5),
    );
    assert_eq!(up[0], 0.0, "the opening pulse closes no period");
    for (k, &v) in up.iter().enumerate().skip(1) {
        assert_eq!(v, f64::INFINITY, "riskless gain at pulse {k}");
    }

    let down = gated(
        metric::performance::return_sharpe(),
        &geometric_path(100.0, -0.25, 5),
    );
    for (k, &v) in down.iter().enumerate().skip(1) {
        assert_eq!(v, f64::NEG_INFINITY, "riskless loss at pulse {k}");
    }
}

/// A flat path has zero mean *and* zero deviation, so the ratio is genuinely
/// undefined and comes back NaN.
#[test]
fn return_sharpe_of_a_flat_path_is_nan() {
    let out = gated(metric::performance::return_sharpe(), &[100.0; 5]);
    assert_eq!(out[0], 0.0);
    for (k, &v) in out.iter().enumerate().skip(1) {
        assert!(v.is_nan(), "flat path at pulse {k} is {v}");
    }
}

// ---------------------------------------------------------------------------
// drawdown
// ---------------------------------------------------------------------------

/// Drawdown is the fraction below the running maximum, so it is zero at the
/// opening sample and at every new high, and never positive.
#[test]
fn drawdown_is_zero_at_every_new_high() {
    let out = gated(
        metric::performance::drawdown(),
        &[100.0, 120.0, 90.0, 150.0],
    );
    assert_eq!(out[0], 0.0, "the first sample is the first peak");
    assert_eq!(out[1], 0.0, "a new high");
    assert_close(&[out[2]], &[-0.25], "below the peak");
    assert_eq!(out[3], 0.0, "a new high clears the shortfall");
}

/// The running maximum never decreases, so a partial recovery still measures
/// against the old peak.
#[test]
fn drawdown_measures_against_a_peak_that_never_resets() {
    let path = positive_path(24);
    let out = gated(metric::performance::drawdown(), &path);

    let mut peak = f64::NEG_INFINITY;
    for (i, &v) in path.iter().enumerate() {
        peak = peak.max(v);
        assert_close(&[out[i]], &[(v - peak) / peak], &format!("sample {i}"));
        assert!(out[i] <= 0.0, "drawdown must not be positive at {i}");
    }
    assert!(
        out.iter().any(|&d| d < -1e-3),
        "the path must actually draw down"
    );
}

// ---------------------------------------------------------------------------
// turnover
// ---------------------------------------------------------------------------

/// Turnover is *cumulative*: each rebalance adds the L1 distance between the
/// new book and the previous one, and the previous book starts empty — so the
/// initial build charges the full notional it puts on.
#[test]
fn turnover_accumulates_the_l1_distance_between_books() {
    let books = vec![
        vec![0.5, 0.5, 0.0],
        vec![0.5, 0.5, 0.0],
        vec![0.25, 0.25, 0.5],
        vec![0.0, 0.0, 1.0],
    ];
    let out = turnover_path(&books);

    let mut prev = vec![0.0; 3];
    let mut running = 0.0;
    for (i, book) in books.iter().enumerate() {
        running += book
            .iter()
            .zip(&prev)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        assert_close(&[out[i]], &[running], &format!("rebalance {i}"));
        prev = book.clone();
    }
    assert_close(&[out[0]], &[1.0], "the initial build charges its notional");
    assert_close(&[out[1]], &[1.0], "an unchanged book adds nothing");
}

/// A full rotation out of one name and into another moves two units of weight.
#[test]
fn turnover_of_a_full_rotation_charges_both_legs() {
    let out = turnover_path(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_close(&[out[0]], &[1.0], "build");
    assert_close(&[out[1]], &[3.0], "build plus a two-unit rotation");
}

/// A short book is charged on the absolute change, so crossing from long to
/// short costs the whole distance travelled.
#[test]
fn turnover_charges_the_absolute_change_through_zero() {
    let out = turnover_path(&[vec![1.0], vec![-1.0]]);
    assert_close(&[out[0]], &[1.0], "build");
    assert_close(&[out[1]], &[3.0], "long to short travels two units");
}

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

/// The return family requires a strictly positive net asset value: a ratio
/// against zero or a negative level is not a return, so it is rejected rather
/// than silently producing infinities.
#[test]
#[should_panic(expected = "must be positive")]
fn return_rejects_a_non_positive_nav() {
    gated(metric::performance::return_mean(), &[100.0, 0.0]);
}

/// Missing data is likewise rejected rather than poisoning the accumulator.
#[test]
#[should_panic(expected = "must be finite")]
fn return_rejects_a_non_finite_nav() {
    gated(metric::performance::return_mean(), &[100.0, f64::NAN]);
}

/// The same precondition holds at the opening pulse, where the level is only
/// latched.
#[test]
#[should_panic(expected = "must be positive")]
fn return_rejects_a_non_positive_opening_nav() {
    gated(metric::performance::comp_return(), &[0.0, 100.0]);
}

/// `drawdown` carries the same requirement.
#[test]
#[should_panic(expected = "must be positive")]
fn drawdown_rejects_a_non_positive_nav() {
    gated(metric::performance::drawdown(), &[100.0, -1.0]);
}

/// `turnover` requires a fixed universe width. Growing or shrinking the weight
/// vector would silently drop names from the L1 sum, so it is rejected.
#[test]
#[should_panic(expected = "length mismatch")]
fn turnover_rejects_a_change_in_universe_width() {
    let mut b = Builder::new();
    let (cell, w) = event_src(&mut b, arr1(vec![0.0_f64; 2]));
    let _ = b.segment(metric::portfolio::turnover(), w);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(cell) = arr1(vec![0.5, 0.5]);
    g.stabilize(&mut pool, &nano(1));
    *g.state_mut(cell) = arr1(vec![0.4, 0.3, 0.3]);
    g.stabilize(&mut pool, &nano(2));
}

/// A missing weight is rejected rather than being read as zero exposure.
#[test]
#[should_panic(expected = "must be finite")]
fn turnover_rejects_a_non_finite_weight() {
    turnover_path(&[vec![0.5, 0.5], vec![0.5, f64::NAN]]);
}

// ---------------------------------------------------------------------------
// Signal gating
// ---------------------------------------------------------------------------

/// A generation where the net asset value moves but the signal does not pulse
/// closes no period: the metric neither folds the observation nor notifies,
/// and the next pulse measures against the level standing at that pulse.
#[test]
fn an_off_signal_move_closes_no_period() {
    let mut b = Builder::new();
    let (data, datav) = b.source(array::scalar(0.0_f64));
    let (tick, tickv) = b.source(signal());
    let mean = b.segment(metric::performance::return_mean(), (tickv, datav));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Opening pulse: latch 100, emit 0.
    *g.state_mut(data) = scalar(100.0);
    let _ = g.state_mut(tick);
    g.stabilize(&mut pool, &nano(1));
    assert_eq!(vals(g.view(mean))[0], 0.0);

    // The level moves twice with no pulse: nothing folds, nothing notifies.
    for (i, v) in [110.0, 130.0].into_iter().enumerate() {
        *g.state_mut(data) = scalar(v);
        g.stabilize(&mut pool, &nano(i as i64 + 2));
        assert_eq!(vals(g.view(mean))[0], 0.0, "held through the quiet tick");
    }

    // The next pulse closes one period against the level standing now (130),
    // not the intermediate 110 that was never sampled.
    let _ = g.state_mut(tick);
    g.stabilize(&mut pool, &nano(4));
    assert_close(&vals(g.view(mean)), &[0.3], "one period, 100 -> 130");
}

/// A pulse on an unchanged level still closes a period: it folds a zero
/// return, which pulls the mean down and the volatility up.
#[test]
fn a_pulse_without_a_move_closes_a_zero_return_period() {
    let path = [100.0, 110.0, 110.0];
    let means = gated(metric::performance::return_mean(), &path);
    let vols = gated(metric::performance::return_vol(), &path);

    assert_close(&[means[1]], &[0.1], "one period");
    assert_close(&[means[2]], &[0.05], "two periods, the second flat");
    assert!(vols[2] > 0.0, "a flat period adds dispersion");
}

// ---------------------------------------------------------------------------
// Predictor-quality metrics (Python segments)
// ---------------------------------------------------------------------------

/// The cross-sectional IC metric is a Python segment, so these need the
/// `python` feature and an interpreter with NumPy.
#[cfg(feature = "python")]
mod information_coefficient {
    use tradingflow::data::Array;
    use tradingflow::graph::Pool;
    use tradingflow::graph::typed::Builder;
    use tradingflow::operators::array::constant;
    use tradingflow::operators::metric::predictor::mean::information_coefficient;
    use tradingflow::operators::series::record_all;

    use crate::harness::*;

    /// One driver step: which of the two inputs fire this generation, and the
    /// cross-sections standing when they do.
    struct Tick {
        factor: Option<[f64; 3]>,
        target: Option<[f64; 3]>,
    }

    fn factor(xs: [f64; 3]) -> Tick {
        Tick {
            factor: Some(xs),
            target: None,
        }
    }

    fn target(xs: [f64; 3]) -> Tick {
        Tick {
            factor: None,
            target: Some(xs),
        }
    }

    /// What one run of the metric produced: the ICs a downstream `record_all`
    /// captured (one row per emission — the output signal is cleared by the
    /// end-of-generation reset, so a pulse is only observable downstream), and
    /// the value standing on the wire after each generation.
    struct Run {
        recorded: Vec<f64>,
        held: Vec<f64>,
    }

    /// Drives the metric over `ticks`.
    fn drive(ticks: &[Tick]) -> Run {
        let mut b = Builder::new();
        let (fsig, fsigv) = b.source(signal());
        let (fval, fvalv) = b.source(constant(Array::<f64, 1>::zeros([3])));
        let (tsig, tsigv) = b.source(signal());
        let (tval, tvalv) = b.source(constant(Array::<f64, 1>::zeros([3])));
        let ic = b.segment(information_coefficient(), (fsigv, fvalv, tsigv, tvalv));
        let rec = b.segment(record_all(), ic);
        let (_, icv) = ic;
        let mut g = b.build();
        let mut pool = Pool::new(0);

        let mut held = Vec::with_capacity(ticks.len());
        for (i, tick) in ticks.iter().enumerate() {
            if let Some(xs) = tick.factor {
                *g.state_mut(fval) = arr1(xs.to_vec());
                let _ = g.state_mut(fsig);
            }
            if let Some(xs) = tick.target {
                *g.state_mut(tval) = arr1(xs.to_vec());
                let _ = g.state_mut(tsig);
            }
            g.stabilize(&mut pool, &nano(i as i64 + 1));
            held.push(vals(g.view(icv))[0]);
        }
        Run {
            recorded: series_vals(g.view(rec)),
            held,
        }
    }

    /// The first factor emission only opens a period; each later one closes
    /// the open period and pulses the mean of the correlations folded inside
    /// it. Target ticks never pulse.
    #[test]
    fn each_factor_emission_after_the_first_closes_a_period() {
        let run = drive(&[
            // Warmup: stores [1,2,3], emits nothing.
            factor([1.0, 2.0, 3.0]),
            // Perfectly aligned, then perfectly inverted: +1 and -1.
            target([10.0, 20.0, 30.0]),
            target([30.0, 20.0, 10.0]),
            // Closes the period at the mean of those two.
            factor([3.0, 1.0, 2.0]),
            // A single aligned tick in the next period.
            target([30.0, 10.0, 20.0]),
            factor([1.0, 1.0, 1.0]),
        ]);

        // Two emissions from six generations: one per factor tick after the
        // warmup, and none from the target ticks.
        assert_close(
            &run.recorded,
            &[0.0, 1.0],
            "mean of (+1, -1), then one aligned tick",
        );
    }

    /// Between emissions the signal clears but the value is retained, so the
    /// wire always carries the last IC and the record gains no spurious rows.
    #[test]
    fn the_value_is_retained_between_emissions() {
        let run = drive(&[
            factor([1.0, 2.0, 3.0]),
            target([10.0, 20.0, 30.0]),
            factor([1.0, 2.0, 3.0]),
            // Two quiet generations after the emission.
            target([10.0, 20.0, 30.0]),
            target([30.0, 20.0, 10.0]),
        ]);

        assert_close(&run.recorded, &[1.0], "one emission, one row");
        assert_close(
            &run.held[2..],
            &[1.0, 1.0, 1.0],
            "the emitted value stays on the wire",
        );
    }

    /// Stocks missing from either side are dropped pairwise, and a tick with
    /// fewer than two usable stocks contributes no correlation while still
    /// counting toward the period's mean.
    #[test]
    fn coverage_is_pairwise_and_thin_ticks_dilute() {
        let nan = f64::NAN;
        let run = drive(&[
            factor([1.0, 2.0, nan]),
            // Only stocks 0 and 1 are usable on both sides: perfectly aligned.
            target([10.0, 20.0, 30.0]),
            factor([1.0, 2.0, 3.0]),
            // Only stock 0 survives: no correlation, but the tick still counts.
            target([10.0, nan, nan]),
            factor([1.0, 2.0, 3.0]),
        ]);

        assert_close(
            &run.recorded,
            &[1.0, 0.0],
            "pairwise-complete stocks correlate; a thin tick contributes zero",
        );
    }
}

/// The covariance-prediction metrics are Python segments sharing one input
/// shape — a predicted `(N, N)` covariance and a realized `(N,)` target — and
/// one cadence, so they share a driver. `minimum_variance` needs SciPy as well
/// as NumPy.
#[cfg(feature = "python")]
mod predictor_variance {
    use tradingflow::data::{Array, Instant};
    use tradingflow::graph::typed::Builder;
    use tradingflow::graph::{Pool, Segment};
    use tradingflow::operators::array::constant;
    use tradingflow::operators::metric::predictor::variance::{log_likelihood, minimum_variance};
    use tradingflow::operators::series::record_all;
    use tradingflow::ports::{ArrayPort, SignalPort};

    use crate::harness::*;

    /// One driver step: which of the two inputs fire this generation, and the
    /// values standing when they do. Predictions are 2x2 row-major.
    struct Tick {
        predict: Option<[f64; 4]>,
        target: Option<[f64; 2]>,
    }

    fn predict(m: [f64; 4]) -> Tick {
        Tick {
            predict: Some(m),
            target: None,
        }
    }

    fn target(xs: [f64; 2]) -> Tick {
        Tick {
            predict: None,
            target: Some(xs),
        }
    }

    /// Drives `metric` over `ticks`, returning the values a downstream
    /// `record_all` captured — one row per emission.
    fn drive<S>(metric: S, ticks: &[Tick]) -> Vec<f64>
    where
        S: Segment<
                Inputs = (
                    SignalPort<0>,
                    ArrayPort<f64, 2>,
                    SignalPort<0>,
                    ArrayPort<f64, 1>,
                ),
                Outputs = (SignalPort<0>, ArrayPort<f64, 0>),
                Context = Instant,
            >,
    {
        let mut b = Builder::new();
        let (psig, psigv) = b.source(signal());
        let (pval, pvalv) = b.source(constant(Array::<f64, 2>::zeros([2, 2])));
        let (tsig, tsigv) = b.source(signal());
        let (tval, tvalv) = b.source(constant(Array::<f64, 1>::zeros([2])));
        let out = b.segment(metric, (psigv, pvalv, tsigv, tvalv));
        let rec = b.segment(record_all(), out);
        let mut g = b.build();
        let mut pool = Pool::new(0);

        for (i, tick) in ticks.iter().enumerate() {
            if let Some(m) = tick.predict {
                *g.state_mut(pval) = arr([2, 2], m.to_vec());
                let _ = g.state_mut(psig);
            }
            if let Some(xs) = tick.target {
                *g.state_mut(tval) = arr1(xs.to_vec());
                let _ = g.state_mut(tsig);
            }
            g.stabilize(&mut pool, &nano(i as i64 + 1));
        }
        series_vals(g.view(rec))
    }

    /// A diagonal covariance makes both terms of the likelihood exact: the
    /// log-determinant is the sum of the log variances, and the Mahalanobis
    /// quadratic is the variance-weighted sum of squares.
    #[test]
    fn log_likelihood_is_log_det_plus_the_mean_quadratic() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let recorded = drive(
            log_likelihood(),
            &[
                // Warmup: caches the identity, emits nothing.
                predict(identity),
                // Quadratics of 1 and 4 under the identity.
                target([1.0, 0.0]),
                target([0.0, 2.0]),
                // Closes at log|I| + mean(1, 4) = 0 + 2.5, and caches
                // diag(4, 1).
                predict([4.0, 0.0, 0.0, 1.0]),
                // Quadratic of 2²/4 + 1²/1 = 2 under diag(4, 1).
                target([2.0, 1.0]),
                predict(identity),
            ],
        );

        assert_close(
            &recorded,
            &[2.5, 4.0_f64.ln() + 2.0],
            "log|Σ| + mean Mahalanobis quadratic",
        );
    }

    /// A stock the predictor could not price is dropped from both terms: its
    /// row and column of the precision matrix stay zero, so however wild its
    /// realized target, it moves neither the determinant nor the quadratic.
    #[test]
    fn log_likelihood_excludes_stocks_without_a_variance() {
        let nan = f64::NAN;
        let recorded = drive(
            log_likelihood(),
            &[
                predict([1.0, 0.0, 0.0, nan]),
                // Only stock 0 is priced: the quadratic is 3² = 9, and the
                // huge second entry contributes nothing.
                target([3.0, 1e6]),
                predict([1.0, 0.0, 0.0, nan]),
            ],
        );

        assert_close(&recorded, &[9.0], "log|1| + 9, stock 1 excluded");
    }

    /// Equal variances and zero covariance put half the portfolio in each
    /// stock, so the period's realized variance is that of the equally
    /// weighted portfolio return.
    #[test]
    fn minimum_variance_is_the_realized_variance_of_the_gmv_portfolio() {
        // Any equal-variance, zero-covariance matrix gives weights [½, ½] —
        // the lognormal conversion scales both diagonal entries alike.
        let equal = [0.01, 0.0, 0.0, 0.01];
        let recorded = drive(
            minimum_variance(),
            &[
                predict(equal),
                // Log returns of 0 and ln 2 map to linear 0 and 1, so the
                // portfolio returns are 0 and 1.
                target([0.0, 0.0]),
                target([2.0_f64.ln(), 2.0_f64.ln()]),
                predict(equal),
            ],
        );

        // Population variance of (0, 1) is ¼.
        assert_close(&recorded, &[0.25], "variance of the portfolio return");
    }

    /// A stock the predictor could not price holds a zero weight, and a
    /// non-finite realized target contributes a zero return rather than
    /// poisoning the period.
    #[test]
    fn minimum_variance_excludes_unpriced_stocks_and_missing_targets() {
        let nan = f64::NAN;
        let recorded = drive(
            minimum_variance(),
            &[
                // Only stock 0 is priced, so it carries the whole portfolio.
                predict([0.01, 0.0, 0.0, nan]),
                // Stock 1's realized return is large but ignored: the
                // portfolio returns are 0 and 1 as above.
                target([0.0, 5.0]),
                target([2.0_f64.ln(), 5.0]),
                // Closes at ¼, and reprices both stocks at weights [½, ½].
                predict([0.01, 0.0, 0.0, 0.01]),
                // A missing target contributes a zero return, so the finite
                // half alone moves the portfolio: ½ · 1 then ½ · 0.
                target([2.0_f64.ln(), nan]),
                target([0.0, nan]),
                predict([0.01, 0.0, 0.0, 0.01]),
            ],
        );

        assert_close(
            &recorded,
            &[0.25, 0.0625],
            "unpriced stock weightless; missing target contributes zero",
        );
    }
}
