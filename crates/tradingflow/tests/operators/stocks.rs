//! `operators::stocks` — the two domain-specific equity operators.
//!
//! Both are stateful across generations, so the tests are tick sequences
//! rather than single shots: `annualize` differences a YTD vector against the
//! previous report, and `forward_adjust` accumulates a corporate-action factor
//! that it applies to every later price.

use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::{array, stocks};

use crate::harness::*;

// ===========================================================================
// annualize
// ===========================================================================

/// Drives `annualize` over a sequence of `[year, day_of_year, ytd…]` rows,
/// returning the annualized `[N]` output observed after each tick.
///
/// The build-time source only has to have the right *width* (the operator
/// reads nothing but the length from it), so it is a row of zeros.
fn annualize_ticks(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let width = rows[0].len();
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(arr([width], vec![0.0; width])));
    let out = b.segment(stocks::annualize(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            assert_eq!(row.len(), width, "every row must have the same width");
            *g.state_mut(src) = arr([width], row.clone());
            g.stabilize(&mut pool, &nano(i as i64));
            vals(g.view(out))
        })
        .collect()
}

/// The first tick of a year annualizes the whole YTD figure: the input layout
/// is `[year, day_of_year, ytd…]` and the scale is `365 / day_of_year`, so the
/// two-element YTD row of day 91, 2024 scales by 365/91.
#[test]
fn annualize_scales_the_first_tick_by_days_since_new_year() {
    let out = annualize_ticks(&[vec![2024.0, 91.0, 100.0, 20.0]]);

    let scale = 365.0 / 91.0;
    assert_eq!(out[0].len(), 2, "output drops the year/day header");
    assert_close(&out[0], &[100.0 * scale, 20.0 * scale], "first tick");
}

/// Within a year the operator annualizes the *increment*: it differences the
/// YTD vector against the previous tick and scales by the days elapsed since
/// that tick, not since 1 January.
#[test]
fn annualize_differences_successive_ticks_within_a_year() {
    let out = annualize_ticks(&[
        vec![2024.0, 91.0, 100.0, 20.0],
        vec![2024.0, 182.0, 250.0, 50.0],
        vec![2024.0, 273.0, 300.0, 51.0],
    ]);

    // Each quarter is 91 days wide here, so the scale stays 365/91.
    let scale = 365.0 / 91.0;
    assert_close(&out[1], &[150.0 * scale, 30.0 * scale], "Q2 increment");
    assert_close(&out[2], &[50.0 * scale, 1.0 * scale], "Q3 increment");
}

/// A change of year restarts the YTD baseline: the new year's first tick is
/// annualized from zero over its own `day_of_year`, never differenced against
/// the previous year's closing YTD (which would come out negative).
#[test]
fn annualize_restarts_on_a_new_year() {
    let out = annualize_ticks(&[vec![2024.0, 365.0, 400.0], vec![2025.0, 31.0, 10.0]]);

    assert_eq!(out[0], vec![400.0], "a full year scales by 365/365 = 1");
    assert_close(&out[1], &[10.0 * 365.0 / 31.0], "restarted from zero");
    assert!(out[1][0] > 0.0, "must not difference against 400");
}

/// Day-of-year boundaries: day 1 is the shortest possible window and scales by
/// the full 365, and day 365 is a whole year and is the identity.
#[test]
fn annualize_day_of_year_boundaries() {
    let first_day = annualize_ticks(&[vec![2024.0, 1.0, 5.0, -2.0]]);
    assert_eq!(first_day[0], vec![5.0 * 365.0, -2.0 * 365.0]);

    let full_year = annualize_ticks(&[vec![2023.0, 365.0, 7.0, -3.0]]);
    assert_eq!(full_year[0], vec![7.0, -3.0]);
}

/// The scaling constant is a flat 365 with no leap-year correction. 2024 *is*
/// a leap year, yet its day 366 scales by 365/366 (< 1) rather than being the
/// identity — pinned deliberately, since it is the kind of thing a future
/// "fix" would change silently.
#[test]
fn annualize_ignores_leap_years() {
    let out = annualize_ticks(&[vec![2024.0, 366.0, 366.0]]);

    assert_close(&out[0], &[366.0 * 365.0 / 366.0], "flat 365-day year");
    assert!(
        out[0][0] < 366.0,
        "a leap-aware scaling would return the YTD unchanged, got {}",
        out[0][0]
    );
}

/// A tick that does not advance the calendar has no window to annualize over,
/// so the whole output is NaN. The YTD baseline is still advanced though, so
/// the next good tick differences against the *rejected* row rather than
/// against the last row that produced a number.
#[test]
fn annualize_is_nan_when_no_days_elapsed() {
    let out = annualize_ticks(&[
        vec![2024.0, 91.0, 100.0],
        vec![2024.0, 91.0, 120.0],  // same day → zero elapsed
        vec![2024.0, 60.0, 130.0],  // calendar moves backwards
        vec![2024.0, 151.0, 200.0], // 91 days after the rejected day-60 row
    ]);

    assert!(out[1][0].is_nan(), "same day: {:?}", out[1]);
    assert!(out[2][0].is_nan(), "day went backwards: {:?}", out[2]);
    assert_close(
        &out[3],
        &[(200.0 - 130.0) * 365.0 / 91.0],
        "baseline advanced through the NaN ticks",
    );

    // Day 0 of a fresh year is the same degenerate case.
    let day_zero = annualize_ticks(&[vec![2024.0, 0.0, 5.0]]);
    assert!(day_zero[0][0].is_nan(), "day 0: {:?}", day_zero[0]);
}

// ===========================================================================
// forward_adjust
// ===========================================================================

/// One generation of a `forward_adjust` graph: the price to poke, and the
/// dividend row `[share_dividends, cash_dividends]` to poke alongside it.
/// `None` leaves that source un-poked, so its input does not notify.
type Tick = (Option<f64>, Option<[f64; 2]>);

/// Drives a rank-0-price `forward_adjust` over `ticks`, returning the output
/// value plus the cumulative number of generations in which the operator
/// actually notified (measured by a [`count`] probe on its output edge).
fn run_forward_adjust(output_prices: bool, ticks: &[Tick]) -> Vec<(f64, usize)> {
    let mut b = Builder::new();
    let (price, pricev) = b.source(array::scalar(f64::NAN));
    let (div, divv) = b.source(array::constant(arr1([0.0, 0.0])));
    let fa = b.segment(
        stocks::forward_adjust::<0, 1>().with_output_prices(output_prices),
        (pricev, divv),
    );
    let notified = b.segment(count::<0>(), fa);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    ticks
        .iter()
        .enumerate()
        .map(|(i, &(p, d))| {
            if let Some(p) = p {
                *g.state_mut(price) = scalar(p);
            }
            if let Some(d) = d {
                *g.state_mut(div) = arr1(d);
            }
            g.stabilize(&mut pool, &nano(i as i64));
            (vals(g.view(fa))[0], g.view(notified))
        })
        .collect()
}

/// Before the first price tick the output buffer holds the operator's identity
/// element — 0.0 in price mode, 1.0 in factor mode — and a generation that
/// pokes only the dividend never writes it or notifies.
#[test]
fn forward_adjust_initial_output_is_the_identity_element() {
    let prices = run_forward_adjust(true, &[(None, Some([0.0, 0.0]))]);
    assert_eq!(prices[0], (0.0, 0), "price mode starts at 0.0");

    let factors = run_forward_adjust(false, &[(None, Some([0.0, 0.0]))]);
    assert_eq!(factors[0], (1.0, 0), "factor mode starts at 1.0");
}

/// A price tick with no corporate action leaves the running factor at 1, so
/// the output is the raw price and the node notifies every time.
#[test]
fn forward_adjust_price_only_tick_is_unadjusted() {
    let out = run_forward_adjust(true, &[(Some(10.0), None), (Some(11.0), None)]);

    assert_eq!(out[0], (10.0, 1));
    assert_eq!(out[1], (11.0, 2));
}

/// A cash dividend `c` paid against the previous close `p` multiplies the
/// running factor by `p / (p − c)`, so the ex-dividend quote re-adjusts to the
/// cum-dividend level: 9.5 after a 0.5 dividend on a 10.0 close is 10.0 again.
#[test]
fn forward_adjust_cash_dividend_restores_the_cum_dividend_level() {
    let out = run_forward_adjust(true, &[(Some(10.0), None), (Some(9.5), Some([0.0, 0.5]))]);

    assert_eq!(out[0].0, 10.0, "no action yet");
    assert_close(
        &[out[1].0],
        &[9.5 * (10.0 / (10.0 - 0.5))],
        "cash factor p / (p - c)",
    );
    assert_close(&[out[1].0], &[10.0], "back to the pre-dividend level");
}

/// A share dividend of ratio `s` multiplies the factor by `1 + s` and does not
/// consult the previous close at all: a 1-for-1 bonus issue halves the quote
/// and doubles the factor.
#[test]
fn forward_adjust_share_dividend_scales_by_one_plus_the_ratio() {
    let out = run_forward_adjust(true, &[(Some(10.0), None), (Some(5.0), Some([1.0, 0.0]))]);

    assert_eq!(out[1].0, 10.0, "5.0 × (1 + 1) is exact");
}

/// Cash and share legs in the same dividend row compose multiplicatively, cash
/// first (against the raw previous close, *not* the share-adjusted one) and
/// share on top: `factor *= p / (p − c)` then `factor *= 1 + s`.
#[test]
fn forward_adjust_applies_cash_then_share_together() {
    let (prev, cash, share) = (10.0_f64, 0.5_f64, 0.1_f64);
    let factor = (prev / (prev - cash)) * (1.0 + share);
    // What the quote does on the ex-date under both actions at once.
    let ex = (prev - cash) / (1.0 + share);

    let out = run_forward_adjust(true, &[(Some(prev), None), (Some(ex), Some([share, cash]))]);

    assert_close(&[out[1].0], &[ex * factor], "combined factor");
    assert_close(
        &[out[1].0],
        &[prev],
        "the pair restores the pre-action level",
    );
}

/// Successive corporate actions compound into a single running product, each
/// cash leg divided by the close that immediately preceded *it* — so the third
/// action here divides by 50, not by the original 100.
#[test]
fn forward_adjust_compounds_successive_factors() {
    let ticks: Vec<Tick> = vec![
        (Some(100.0), None),
        (Some(98.0), Some([0.0, 2.0])), // cash 2 against a 100 close
        (Some(101.0), None),
        (Some(50.0), Some([1.0, 0.0])), // 1-for-1 bonus issue
        (Some(49.0), Some([0.0, 1.0])), // cash 1 against a 50 close
        (Some(52.0), None),
    ];

    // The factor after each action, written as the running product.
    let f1 = 100.0 / 98.0;
    let f2 = f1 * 2.0;
    let f3 = f2 * (50.0 / 49.0);
    let expected = [
        100.0,
        98.0 * f1,
        101.0 * f1,
        50.0 * f2,
        49.0 * f3,
        52.0 * f3,
    ];

    let out = run_forward_adjust(true, &ticks);
    let got: Vec<f64> = out.iter().map(|&(v, _)| v).collect();
    assert_close(&got, &expected, "compounded factors");
    assert_close(
        &[got[1]],
        &[100.0],
        "the cash action undoes the 100 → 98 drop",
    );
    assert_eq!(out[5].1, 6, "every generation had a price and so notified");
}

/// `with_output_prices` selects between the two things the operator can emit.
/// `true` (the default) gives the adjusted price, `false` gives the bare
/// running factor; the two wirings differ by exactly the raw price.
#[test]
fn forward_adjust_with_output_prices_toggles_price_versus_factor() {
    let ticks: Vec<Tick> = vec![
        (Some(100.0), None),
        (Some(98.0), Some([0.0, 2.0])),
        (Some(50.0), Some([1.0, 0.0])),
    ];

    let prices = run_forward_adjust(true, &ticks);
    let factors = run_forward_adjust(false, &ticks);

    assert_eq!(factors[0].0, 1.0, "no action yet");
    assert_close(&[factors[1].0], &[100.0 / 98.0], "after the cash dividend");
    assert_close(
        &[factors[2].0],
        &[2.0 * 100.0 / 98.0],
        "after the bonus issue",
    );
    for (i, (p, f)) in prices.iter().zip(&factors).enumerate() {
        let raw = ticks[i].0.expect("every tick here carries a price");
        assert_close(&[p.0], &[raw * f.0], "adjusted price == raw price × factor");
    }
}

/// The two inputs are read independently via their notify flags. A generation
/// in which only the dividend notifies banks the factor but writes no output
/// and does not notify downstream — the banked factor lands on the next price
/// tick instead.
#[test]
fn forward_adjust_dividend_without_a_price_defers_to_the_next_price_tick() {
    let out = run_forward_adjust(
        true,
        &[
            (Some(10.0), None),
            (None, Some([0.0, 0.5])), // dividend alone
            (Some(9.5), None),        // the banked factor applies here
        ],
    );

    assert_eq!(out[0], (10.0, 1));
    assert_eq!(
        out[1],
        (10.0, 1),
        "no price → stale output and no notification"
    );
    assert_close(
        &[out[2].0],
        &[10.0],
        "banked factor applied to the next price",
    );
    assert_eq!(out[2].1, 2, "the price tick notifies");
}

/// A dividend arriving before any price has been seen has no previous close to
/// divide by, so it is dropped entirely rather than poisoning the factor with
/// NaN — including its share leg, which would otherwise be well-defined.
#[test]
fn forward_adjust_ignores_a_dividend_before_the_first_price() {
    let out = run_forward_adjust(false, &[(Some(10.0), Some([1.0, 0.5])), (Some(10.0), None)]);

    assert_eq!(out[0].0, 1.0, "factor untouched on the very first tick");
    assert_eq!(out[1].0, 1.0, "and it stays untouched afterwards");
}

/// A cash dividend at or above the previous close would drive the factor
/// negative or infinite; the operator asserts the precondition instead of
/// silently producing garbage.
#[test]
#[should_panic(expected = "prev_price > cash_dividends")]
fn forward_adjust_rejects_a_dividend_above_the_previous_close() {
    run_forward_adjust(true, &[(Some(1.0), None), (Some(0.1), Some([0.0, 1.0]))]);
}

/// The operator is const-generic over the two input ranks and the output
/// mirrors the *price* rank: a rank-1 `[1]` price row with a rank-2 `[1, 2]`
/// dividend block produces a rank-1 `[1]` output, arithmetically identical to
/// the rank-0 wiring above.
#[test]
fn forward_adjust_mirrors_the_price_rank() {
    let mut b = Builder::new();
    let (price, pricev) = b.source(array::constant(arr1([f64::NAN])));
    let (div, divv) = b.source(array::constant(arr([1, 2], vec![0.0, 0.0])));
    let fa = b.segment(stocks::forward_adjust::<1, 2>(), (pricev, divv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(price) = arr1([10.0]);
    g.stabilize(&mut pool, &nano(0));
    assert_eq!(
        g.view(fa).extents(),
        [1],
        "output mirrors the price extents"
    );
    assert_eq!(vals(g.view(fa)), vec![10.0]);

    *g.state_mut(price) = arr1([9.5]);
    *g.state_mut(div) = arr([1, 2], vec![0.0, 0.5]);
    g.stabilize(&mut pool, &nano(1));
    assert_close(
        &vals(g.view(fa)),
        &[10.0],
        "rank-1 price row, rank-2 dividend",
    );
}
