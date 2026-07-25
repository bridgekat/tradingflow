//! `operators::feature::stocks` — the two domain-specific equity operators.
//!
//! Both are stateful across generations, so the tests are tick sequences
//! rather than single shots: `annualize` differences each element's YTD value
//! against that element's previous report, and `forward_adjust` accumulates a
//! per-element corporate-action multiplier that it applies to every later
//! close.
//!
//! Both follow the fine-grained event model (see `operators::event`): an input
//! carries an event only when its port notifies *and* the element is non-NaN,
//! so a NaN element inside a notified array is "no event for this element" and
//! an un-notified port is "no events at all". The outputs mirror this: a
//! notified output with a NaN element means that element had no event this
//! generation.

use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::{array, feature::stock};

use crate::harness::*;

// ===========================================================================
// annualize
// ===========================================================================

/// One `annualize` tick: the report period's `(year, day_of_year)` plus the
/// YTD row (NaN elements carry no event).
type ReportTick = (i32, i32, Vec<f64>);

/// Drives `annualize` over report ticks, poking all three sources each
/// generation, and returns the output row observed after each tick.
///
/// The `year` / `day` sources are `[1]` arrays against a `[width]` value row,
/// so every tick also exercises the broadcast of the calendar inputs.
fn annualize_ticks(width: usize, ticks: &[ReportTick]) -> Vec<Vec<f64>> {
    let mut b = Builder::new();
    let (values, valuesv) = b.source(array::constant(arr([width], vec![f64::NAN; width])));
    let (year, yearv) = b.source(array::constant(arr([1], vec![0_i32])));
    let (day, dayv) = b.source(array::constant(arr([1], vec![0_i32])));
    let out = b.segment(stock::annualize(), (valuesv, yearv, dayv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    ticks
        .iter()
        .enumerate()
        .map(|(i, (y, d, row))| {
            assert_eq!(row.len(), width, "every row must have the same width");
            *g.state_mut(values) = arr([width], row.clone());
            *g.state_mut(year) = arr([1], vec![*y]);
            *g.state_mut(day) = arr([1], vec![*d]);
            g.stabilize(&mut pool, &nano(i as i64));
            vals(g.view(out))
        })
        .collect()
}

/// The first report of a year annualizes the whole YTD figure: the scale is
/// `365 / day_of_year`, so a two-element YTD row on day 91 scales by 365/91.
#[test]
fn annualize_scales_the_first_tick_by_days_since_new_year() {
    let out = annualize_ticks(2, &[(2024, 91, vec![100.0, 20.0])]);

    let scale = 365.0 / 91.0;
    assert_close(&out[0], &[100.0 * scale, 20.0 * scale], "first tick");
}

/// Within a year the operator annualizes the *increment*: it differences the
/// YTD value against the previous report and scales by the days elapsed since
/// that report, not since 1 January.
#[test]
fn annualize_differences_successive_ticks_within_a_year() {
    let out = annualize_ticks(
        2,
        &[
            (2024, 91, vec![100.0, 20.0]),
            (2024, 182, vec![250.0, 50.0]),
            (2024, 273, vec![300.0, 51.0]),
        ],
    );

    // Each quarter is 91 days wide here, so the scale stays 365/91.
    let scale = 365.0 / 91.0;
    assert_close(&out[1], &[150.0 * scale, 30.0 * scale], "Q2 increment");
    assert_close(&out[2], &[50.0 * scale, 1.0 * scale], "Q3 increment");
}

/// A change of year restarts the YTD baseline: the new year's first report is
/// annualized from zero over its own `day_of_year`, never differenced against
/// the previous year's closing YTD (which would come out negative).
#[test]
fn annualize_restarts_on_a_new_year() {
    let out = annualize_ticks(1, &[(2024, 365, vec![400.0]), (2025, 31, vec![10.0])]);

    assert_eq!(out[0], vec![400.0], "a full year scales by 365/365 = 1");
    assert_close(&out[1], &[10.0 * 365.0 / 31.0], "restarted from zero");
    assert!(out[1][0] > 0.0, "must not difference against 400");
}

/// Day-of-year boundaries: day 1 is the shortest possible window and scales by
/// the full 365, day 365 is a whole year and is the identity, and day 0 gives
/// no window at all, so the output element is never written and stays NaN.
#[test]
fn annualize_day_of_year_boundaries() {
    let first_day = annualize_ticks(2, &[(2024, 1, vec![5.0, -2.0])]);
    assert_eq!(first_day[0], vec![5.0 * 365.0, -2.0 * 365.0]);

    let full_year = annualize_ticks(2, &[(2023, 365, vec![7.0, -3.0])]);
    assert_eq!(full_year[0], vec![7.0, -3.0]);

    let day_zero = annualize_ticks(1, &[(2024, 0, vec![5.0])]);
    assert!(day_zero[0][0].is_nan(), "day 0: {:?}", day_zero[0]);
}

/// The scaling constant is a flat 365 with no leap-year correction. 2024 *is*
/// a leap year, yet its day 366 scales by 365/366 (< 1) rather than being the
/// identity — pinned deliberately, since it is the kind of thing a future
/// "fix" would change silently.
#[test]
fn annualize_ignores_leap_years() {
    let out = annualize_ticks(1, &[(2024, 366, vec![366.0])]);

    assert_close(&out[0], &[366.0 * 365.0 / 366.0], "flat 365-day year");
    assert!(
        out[0][0] < 366.0,
        "a leap-aware scaling would return the YTD unchanged, got {}",
        out[0][0]
    );
}

/// A duplicate report on the same day has no window to annualize over, so the
/// output element is not rewritten — it *retains* (and re-emits) the previous
/// report's annualized value. The YTD baseline is still advanced though, so
/// the next good report differences against the duplicate's value rather than
/// against the report that produced the retained number.
#[test]
fn annualize_same_day_report_retains_output_but_advances_the_baseline() {
    let out = annualize_ticks(
        1,
        &[
            (2024, 91, vec![100.0]),
            (2024, 91, vec![120.0]),  // same day → zero elapsed
            (2024, 182, vec![250.0]), // 91 days after the duplicate
        ],
    );

    let scale = 365.0 / 91.0;
    assert_close(&out[0], &[100.0 * scale], "first report");
    assert_close(
        &out[1],
        &[100.0 * scale],
        "duplicate re-emits the old value",
    );
    assert_close(
        &out[2],
        &[(250.0 - 120.0) * scale],
        "baseline advanced through the duplicate",
    );
}

/// The calendar inputs are asserted monotonic per element: a report whose year
/// moves backwards (e.g. a restated old period arriving after a newer one)
/// panics rather than silently mangling the baseline.
#[test]
#[should_panic(expected = "year must be monotonic")]
fn annualize_rejects_a_backwards_year() {
    annualize_ticks(1, &[(2024, 91, vec![100.0]), (2023, 182, vec![50.0])]);
}

/// Within a year the day-of-year must not move backwards either.
#[test]
#[should_panic(expected = "day must be monotonic within a year")]
fn annualize_rejects_a_backwards_day_within_a_year() {
    annualize_ticks(1, &[(2024, 91, vec![100.0]), (2024, 60, vec![130.0])]);
}

/// Events are per element: a NaN element inside a notified row carries no
/// event, so its output element is NaN and its baseline is untouched, while
/// the other element proceeds independently. Each element differences against
/// *its own* previous report, not the row's.
#[test]
fn annualize_elements_carry_events_independently() {
    let out = annualize_ticks(
        2,
        &[
            (2024, 91, vec![100.0, f64::NAN]),
            (2024, 182, vec![f64::NAN, 50.0]),
            (2024, 273, vec![150.0, f64::NAN]),
        ],
    );

    assert_close(
        &out[0],
        &[100.0 * 365.0 / 91.0, f64::NAN],
        "element 1 has no event yet",
    );
    assert_close(
        &out[1],
        &[f64::NAN, 50.0 * 365.0 / 182.0],
        "element 1's first event annualizes from zero",
    );
    // Element 0 differences 150 - 100 over days 273 - 91: its own baseline,
    // untouched by the tick it sat out.
    assert_close(
        &out[2],
        &[(150.0 - 100.0) * 365.0 / (273.0 - 91.0), f64::NAN],
        "per-element baseline",
    );
}

/// A generation in which the calendar notifies but the values port does not is
/// "no events at all": the operator still recomputes (any-input gate) and
/// notifies, but every output element is NaN — the retained annualized values
/// are *not* re-emitted for a calendar-only tick.
#[test]
fn annualize_value_silence_yields_an_all_nan_notification() {
    let mut b = Builder::new();
    let (values, valuesv) = b.source(array::constant(arr([1], vec![f64::NAN])));
    let (year, yearv) = b.source(array::constant(arr([1], vec![0_i32])));
    let (day, dayv) = b.source(array::constant(arr([1], vec![0_i32])));
    let out = b.segment(stock::annualize(), (valuesv, yearv, dayv));
    let notified = b.segment(count::<1>(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(values) = arr([1], vec![100.0]);
    *g.state_mut(year) = arr([1], vec![2024]);
    *g.state_mut(day) = arr([1], vec![91]);
    g.stabilize(&mut pool, &nano(0));
    assert_close(&vals(g.view(out)), &[100.0 * 365.0 / 91.0], "good tick");
    assert_eq!(g.view(notified), 1);

    // Poke only the calendar: the values port does not notify.
    *g.state_mut(year) = arr([1], vec![2024]);
    g.stabilize(&mut pool, &nano(1));
    assert!(
        vals(g.view(out))[0].is_nan(),
        "calendar-only tick carries no events"
    );
    assert_eq!(g.view(notified), 2, "but the output still notifies");
}

// ===========================================================================
// forward_adjust
// ===========================================================================

/// One `forward_adjust` generation: the close, share-dividend and
/// cash-dividend legs to poke. `None` leaves that source un-poked, so its
/// input does not notify (equivalently: carries no event).
type AdjustTick = (Option<f64>, Option<f64>, Option<f64>);

/// Drives a rank-0 `forward_adjust` over `ticks`, returning per tick the
/// `(multiplier, adjusted close)` outputs plus the cumulative number of
/// generations in which the adjusted-close output notified.
fn run_forward_adjust(ticks: &[AdjustTick]) -> Vec<(f64, f64, usize)> {
    let mut b = Builder::new();
    let (close, closev) = b.source(array::scalar(f64::NAN));
    let (share, sharev) = b.source(array::scalar(f64::NAN));
    let (cash, cashv) = b.source(array::scalar(f64::NAN));
    let (mult, adj) = b.segment(stock::forward_adjust(), (closev, sharev, cashv));
    let notified = b.segment(count::<0>(), adj);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    ticks
        .iter()
        .enumerate()
        .map(|(i, &(c, s, d))| {
            if let Some(c) = c {
                *g.state_mut(close) = scalar(c);
            }
            if let Some(s) = s {
                *g.state_mut(share) = scalar(s);
            }
            if let Some(d) = d {
                *g.state_mut(cash) = scalar(d);
            }
            g.stabilize(&mut pool, &nano(i as i64));
            (
                vals(g.view(mult))[0],
                vals(g.view(adj))[0],
                g.view(notified),
            )
        })
        .collect()
}

/// A close with no corporate action leaves the multiplier at its identity 1,
/// so the adjusted close is the raw close, notifying every close tick.
#[test]
fn forward_adjust_close_only_tick_is_unadjusted() {
    let out = run_forward_adjust(&[(Some(10.0), None, None), (Some(11.0), None, None)]);

    assert_eq!(out[0], (1.0, 10.0, 1));
    assert_eq!(out[1], (1.0, 11.0, 2));
}

/// A cash dividend `c` paid against the previous close `p` multiplies the
/// running multiplier by `1 + c / (p − c)` = `p / (p − c)`, so the ex-dividend
/// quote re-adjusts to the cum-dividend level: 9.5 after a 0.5 dividend on a
/// 10.0 close is 10.0 again.
#[test]
fn forward_adjust_cash_dividend_restores_the_cum_dividend_level() {
    let out = run_forward_adjust(&[
        (Some(10.0), None, None),
        (Some(9.5), None, Some(0.5)), // ex-date: quote drops by the dividend
    ]);

    assert_eq!(out[0].1, 10.0, "no action yet");
    assert_close(
        &[out[1].0],
        &[10.0 / (10.0 - 0.5)],
        "multiplier p / (p - c)",
    );
    assert_close(&[out[1].1], &[10.0], "back to the pre-dividend level");
}

/// A share dividend of ratio `s` multiplies the multiplier by `1 + s` and does
/// not consult the previous close at all: a 1-for-1 bonus issue halves the
/// quote and doubles the multiplier.
#[test]
fn forward_adjust_share_dividend_scales_by_one_plus_the_ratio() {
    let out = run_forward_adjust(&[(Some(10.0), None, None), (Some(5.0), Some(1.0), None)]);

    assert_eq!(out[1].0, 2.0, "1-for-1 doubles the multiplier");
    assert_eq!(out[1].1, 10.0, "5.0 × (1 + 1) is exact");
}

/// Cash and share legs in the same generation compose multiplicatively, the
/// cash leg against the raw previous close (*not* the share-adjusted one):
/// `multiplier *= (1 + s) × p / (p − c)`.
#[test]
fn forward_adjust_applies_cash_and_share_together() {
    let (prev, cash, share) = (10.0_f64, 0.5_f64, 0.1_f64);
    let multiplier = (prev / (prev - cash)) * (1.0 + share);
    // What the quote does on the ex-date under both actions at once.
    let ex = (prev - cash) / (1.0 + share);

    let out = run_forward_adjust(&[
        (Some(prev), None, None),
        (Some(ex), Some(share), Some(cash)),
    ]);

    assert_close(&[out[1].0], &[multiplier], "combined multiplier");
    assert_close(
        &[out[1].1],
        &[prev],
        "the pair restores the pre-action level",
    );
}

/// Successive corporate actions compound into a single running product, each
/// cash leg divided by the close that immediately preceded *it* — so the third
/// action here divides by 50, not by the original 100.
#[test]
fn forward_adjust_compounds_successive_multipliers() {
    let ticks: Vec<AdjustTick> = vec![
        (Some(100.0), None, None),
        (Some(98.0), None, Some(2.0)), // cash 2 against a 100 close
        (Some(101.0), None, None),
        (Some(50.0), Some(1.0), None), // 1-for-1 bonus issue
        (Some(49.0), None, Some(1.0)), // cash 1 against a 50 close
        (Some(52.0), None, None),
    ];

    // The multiplier after each action, written as the running product.
    let f1 = 100.0 / 98.0;
    let f2 = f1 * 2.0;
    let f3 = f2 * (50.0 / 49.0);
    let expected_mult = [1.0, f1, f1, f2, f3, f3];
    let expected_adj = [
        100.0,
        98.0 * f1,
        101.0 * f1,
        50.0 * f2,
        49.0 * f3,
        52.0 * f3,
    ];

    let out = run_forward_adjust(&ticks);
    let mults: Vec<f64> = out.iter().map(|&(m, _, _)| m).collect();
    let adjs: Vec<f64> = out.iter().map(|&(_, a, _)| a).collect();
    assert_close(&mults, &expected_mult, "compounded multipliers");
    assert_close(&adjs, &expected_adj, "adjusted closes");
    assert_close(
        &[adjs[1]],
        &[100.0],
        "the cash action undoes the 100 → 98 drop",
    );
}

/// The three inputs carry events independently. A generation in which only a
/// dividend notifies banks the multiplier immediately; the adjusted close has
/// no close event to price, so it comes out NaN — but it *does* notify (an
/// all-NaN notification is "no events", not "no tick"). The banked multiplier
/// lands on the next close.
#[test]
fn forward_adjust_dividend_without_a_close_banks_the_multiplier() {
    let out = run_forward_adjust(&[
        (Some(10.0), None, None),
        (None, None, Some(0.5)), // dividend alone
        (Some(9.5), None, None), // the banked multiplier applies here
    ]);

    assert_eq!(out[0], (1.0, 10.0, 1));
    assert_close(&[out[1].0], &[10.0 / 9.5], "multiplier banks immediately");
    assert!(out[1].1.is_nan(), "no close event → NaN adjusted close");
    assert_eq!(out[1].2, 2, "the all-NaN generation still notifies");
    assert_close(
        &[out[2].1],
        &[10.0],
        "banked multiplier applied to the next close",
    );
}

/// Before the first close there is no previous quote to divide by, so a cash
/// dividend is dropped — but a share dividend needs no reference close and
/// applies. (The pre-rewrite operator dropped both legs; the share leg's
/// arithmetic is well-defined without a close, so it now counts.)
#[test]
fn forward_adjust_before_the_first_close_drops_cash_but_applies_share() {
    let out = run_forward_adjust(&[
        (None, Some(1.0), Some(0.5)), // both legs before any close
        (Some(10.0), None, None),
    ]);

    assert_eq!(out[0].0, 2.0, "share leg applies, cash leg is dropped");
    assert!(out[0].1.is_nan(), "no close yet");
    assert_close(&[out[1].1], &[20.0], "the share multiplier sticks");
}

/// Dividend feeds may pad event rows with explicit zeros; a zero leg is the
/// multiplicative identity and must leave the multiplier untouched.
#[test]
fn forward_adjust_zero_dividends_are_no_ops() {
    let out = run_forward_adjust(&[(Some(10.0), None, None), (Some(11.0), Some(0.0), Some(0.0))]);

    assert_eq!(out[1].0, 1.0, "zero legs leave the identity");
    assert_eq!(out[1].1, 11.0);
}

/// Events are per element: a NaN leg inside a notified dividend array skips
/// only that element, and each element's cash leg divides by *its own*
/// previous close.
#[test]
fn forward_adjust_elements_carry_events_independently() {
    let mut b = Builder::new();
    let (close, closev) = b.source(array::constant(arr1([f64::NAN, f64::NAN])));
    let (_, sharev) = b.source(array::constant(arr1([f64::NAN, f64::NAN])));
    let (cash, cashv) = b.source(array::constant(arr1([f64::NAN, f64::NAN])));
    let (mult, adj) = b.segment(stock::forward_adjust(), (closev, sharev, cashv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(close) = arr1([10.0, 20.0]);
    g.stabilize(&mut pool, &nano(0));
    assert_close(&vals(g.view(adj)), &[10.0, 20.0], "no actions yet");

    // A cash dividend on element 1 only; element 0's leg is NaN (no event).
    *g.state_mut(close) = arr1([10.0, 19.0]);
    *g.state_mut(cash) = arr1([f64::NAN, 1.0]);
    g.stabilize(&mut pool, &nano(1));
    assert_close(
        &vals(g.view(mult)),
        &[1.0, 20.0 / 19.0],
        "only element 1's multiplier moves, against its own 20.0 close",
    );
    assert_close(
        &vals(g.view(adj)),
        &[10.0, 20.0],
        "element 1 re-adjusts to its cum-dividend level",
    );
}

/// The dividend legs broadcast to the closes' extents: a `[1]` share dividend
/// against `[2]` closes applies to every element.
#[test]
fn forward_adjust_broadcasts_the_dividend_legs() {
    let mut b = Builder::new();
    let (close, closev) = b.source(array::constant(arr1([f64::NAN, f64::NAN])));
    let (share, sharev) = b.source(array::constant(arr([1], vec![f64::NAN])));
    let (_, cashv) = b.source(array::constant(arr([1], vec![f64::NAN])));
    let (mult, adj) = b.segment(stock::forward_adjust(), (closev, sharev, cashv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(close) = arr1([10.0, 20.0]);
    g.stabilize(&mut pool, &nano(0));

    *g.state_mut(close) = arr1([5.0, 10.0]);
    *g.state_mut(share) = arr([1], vec![1.0]);
    g.stabilize(&mut pool, &nano(1));
    assert_close(&vals(g.view(mult)), &[2.0, 2.0], "one leg, every element");
    assert_close(&vals(g.view(adj)), &[10.0, 20.0], "both quotes re-adjust");
}
