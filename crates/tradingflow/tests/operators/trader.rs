//! `operators::trader::fixed` — the backtest executor.
//!
//! One engine drives every test: a `[n]` book fed by three event streams —
//! quotes on a price signal, dividends on a per-stock signal array, and target
//! weights on a rebalance signal. Each test is a script of generations (see
//! [`Step`]), because everything interesting here is stateful across
//! generations: the delayed-execution latch, the retained marks, and the book
//! itself.
//!
//! The quote conventions under test are the ones the engine documents: a side
//! that cannot trade is `±inf` (never `NaN`, which is a data error), a
//! suspended stock quotes `(-inf, +inf)` and keeps its previous mark, and a
//! `flags` element of `false` is a delisting that marks the stock to zero.

use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::trader::fixed::{Exec, Fixed, benchmark, random, simple};
use tradingflow::operators::{array, signal};

use crate::harness::*;

const INF: f64 = f64::INFINITY;

/// One generation of a trader script. A field left `None` leaves that stream
/// quiet: quotes, dividends and targets each pulse their own signal only when
/// supplied, which is what makes the cadence interactions testable.
#[derive(Clone, Default)]
struct Step {
    /// Pokes the quotes **and** pulses the price signal.
    quotes: Option<(Vec<f64>, Vec<f64>)>,
    /// Pokes the listing flags. A level, not an event — no signal.
    flags: Option<Vec<bool>>,
    /// Pokes the dividend components **and** pulses every stock's dividend
    /// signal. A stock with `(0, 0)` is an exact no-op, so pulsing the whole
    /// cross-section is the canonical way to spell "these stocks paid".
    dividend: Option<(Vec<f64>, Vec<f64>)>,
    /// Pokes the target weights **and** pulses the rebalance signal.
    target: Option<Vec<f64>>,
    /// Pokes the target weights *without* pulsing — the wire drifts under a
    /// trader that has already latched a rebalance.
    weights: Option<Vec<f64>>,
}

impl Step {
    fn quote(bids: &[f64], asks: &[f64]) -> Self {
        Self {
            quotes: Some((bids.to_vec(), asks.to_vec())),
            ..Self::default()
        }
    }

    fn nothing() -> Self {
        Self::default()
    }

    fn then_target(mut self, w: &[f64]) -> Self {
        self.target = Some(w.to_vec());
        self
    }

    fn then_weights(mut self, w: &[f64]) -> Self {
        self.weights = Some(w.to_vec());
        self
    }

    fn then_flags(mut self, f: &[bool]) -> Self {
        self.flags = Some(f.to_vec());
        self
    }

    fn then_dividend(mut self, share: &[f64], cash: &[f64]) -> Self {
        self.dividend = Some((share.to_vec(), cash.to_vec()));
        self
    }
}

/// What the book read at the end of one generation.
#[derive(Clone, Debug)]
struct Obs {
    positions: Vec<f64>,
    cash: f64,
    net: f64,
}

/// Drives `trader` over `steps`, returning one [`Obs`] per generation.
///
/// The wiring mirrors a real backtest: three independent signals (price,
/// dividend, rebalance) so a script can pulse any subset in a generation, and
/// quotes/flags/weights as ordinary retained state arrays.
fn run<E: Exec>(trader: Fixed<E>, n: usize, steps: &[Step]) -> Vec<Obs> {
    let mut b = Builder::new();

    let (price_tick, price_signal) = b.source(signal());
    let (flags_h, flags_v) = b.source(array::constant(arr([n], vec![true; n])));
    let (bids_h, bids_v) = b.source(array::constant(arr1(vec![-INF; n])));
    let (asks_h, asks_v) = b.source(array::constant(arr1(vec![INF; n])));

    // One manual signal per stock, stacked into the `[n]` dividend signal array.
    let div_ticks: Vec<_> = (0..n).map(|_| b.source(signal())).collect();
    let div_signals: Vec<_> = div_ticks.iter().map(|&(_, c)| c).collect();
    let div_signal = b.segment(
        signal::as_signal_map(array::stack::<bool, 0, 1>(0)),
        &div_signals[..],
    );
    let (share_h, share_v) = b.source(array::constant(arr1(vec![0.0; n])));
    let (cash_h, cash_v) = b.source(array::constant(arr1(vec![0.0; n])));

    let (reb_tick, reb_signal) = b.source(signal());
    let (w_h, w_v) = b.source(array::constant(arr1(vec![0.0; n])));

    let (pos, cash_out, net_out) = b.segment(
        trader,
        (
            (price_signal, flags_v, bids_v, asks_v),
            (div_signal, share_v, cash_v),
            (reb_signal, w_v),
        ),
    );

    let mut g = b.build();
    let mut pool = Pool::new(0);
    let mut out = Vec::new();
    for (i, s) in steps.iter().enumerate() {
        if let Some(f) = &s.flags {
            *g.state_mut(flags_h) = arr([n], f.clone());
        }
        if let Some((bd, ak)) = &s.quotes {
            *g.state_mut(bids_h) = arr1(bd.clone());
            *g.state_mut(asks_h) = arr1(ak.clone());
            let _ = g.state_mut(price_tick);
        }
        if let Some((sh, ca)) = &s.dividend {
            *g.state_mut(share_h) = arr1(sh.clone());
            *g.state_mut(cash_h) = arr1(ca.clone());
            for &(h, _) in &div_ticks {
                let _ = g.state_mut(h);
            }
        }
        if let Some(w) = &s.weights {
            *g.state_mut(w_h) = arr1(w.clone());
        }
        if let Some(w) = &s.target {
            *g.state_mut(w_h) = arr1(w.clone());
            let _ = g.state_mut(reb_tick);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        out.push(Obs {
            positions: vals(g.view(pos)),
            cash: *g.view(cash_out),
            net: *g.view(net_out),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Rebalance scheduling
// ---------------------------------------------------------------------------

/// The undelayed engine executes in the generation it is armed, provided that
/// generation carries a price pulse: weights become positions at the quoted
/// prices, and net value is conserved through a frictionless fill.
#[test]
fn an_undelayed_rebalance_executes_against_the_same_pulse() {
    let out = run(
        benchmark(false, 1000.0),
        2,
        &[Step::quote(&[10.0, 20.0], &[10.0, 20.0]).then_target(&[0.5, 0.5])],
    );

    assert_close(&out[0].positions, &[50.0, 25.0], "500 into each leg");
    assert_close(&[out[0].cash], &[0.0], "fully invested");
    assert_close(&[out[0].net], &[1000.0], "frictionless fill conserves NAV");
}

/// `delayed` moves the same execution to the *next* price pulse — the
/// look-ahead-safe schedule for weights computed from the prices that armed
/// the trader. The armed generation must leave the book untouched.
#[test]
fn a_delayed_rebalance_waits_for_the_next_price_pulse() {
    let out = run(
        benchmark(true, 1000.0),
        2,
        &[
            Step::quote(&[10.0, 20.0], &[10.0, 20.0]).then_target(&[0.5, 0.5]),
            Step::quote(&[10.0, 20.0], &[10.0, 20.0]),
        ],
    );

    assert_close(&out[0].positions, &[0.0, 0.0], "armed, not executed");
    assert_close(
        &[out[0].cash],
        &[1000.0],
        "cash untouched on the arming tick",
    );
    assert_close(&out[1].positions, &[50.0, 25.0], "executed one pulse later");
}

/// The latched weights are the ones sampled *at the pulse*: a weights wire
/// that drifts between the arming pulse and the executing price pulse must
/// not change what gets traded. (Reading the live input instead would have
/// bought the drifted target here.)
#[test]
fn a_delayed_rebalance_executes_the_weights_latched_at_the_pulse() {
    let out = run(
        benchmark(true, 1000.0),
        2,
        &[
            Step::quote(&[10.0, 20.0], &[10.0, 20.0]).then_target(&[1.0, 0.0]),
            Step::quote(&[10.0, 20.0], &[10.0, 20.0]).then_weights(&[0.0, 1.0]),
        ],
    );

    assert_close(
        &out[1].positions,
        &[100.0, 0.0],
        "the latched [1, 0], not the drifted [0, 1]",
    );
}

/// A rebalance pulse on a generation with no price pulse has no market to
/// trade in, so it waits — the arming survives arbitrarily many quiet
/// generations rather than being dropped.
#[test]
fn a_rebalance_without_a_price_pulse_waits_for_one() {
    let out = run(
        benchmark(false, 1000.0),
        1,
        &[
            Step::nothing().then_target(&[1.0]),
            Step::nothing(),
            Step::quote(&[10.0], &[10.0]),
        ],
    );

    assert_close(&out[0].positions, &[0.0], "no market yet");
    assert_close(&out[1].positions, &[0.0], "still waiting");
    assert_close(
        &out[2].positions,
        &[100.0],
        "executes on the first price pulse",
    );
}

// ---------------------------------------------------------------------------
// Marking
// ---------------------------------------------------------------------------

/// A suspended stock quotes `(-inf, +inf)`: there is no price to mark at, so
/// the book keeps the previous mark and net value is unchanged. A rebalance
/// during the suspension cannot trade it.
#[test]
fn a_suspended_stock_retains_its_mark_and_cannot_trade() {
    let out = run(
        benchmark(false, 1000.0),
        1,
        &[
            Step::quote(&[10.0], &[10.0]).then_target(&[1.0]),
            Step::quote(&[-INF], &[INF]),
            Step::quote(&[-INF], &[INF]).then_target(&[0.0]),
        ],
    );

    assert_close(&[out[1].net], &[1000.0], "marked at the last quote");
    assert_close(&out[2].positions, &[100.0], "no exit while suspended");
}

/// A limit-down day is one-sided: the bid is `-inf` (nobody will buy) but the
/// ask is the floor price, so the book marks there — a real mark-down, not a
/// retained one — and the position is still unsellable.
#[test]
fn a_limit_down_stock_marks_at_the_ask() {
    let out = run(
        benchmark(false, 1000.0),
        1,
        &[
            Step::quote(&[10.0], &[10.0]).then_target(&[1.0]),
            Step::quote(&[-INF], &[9.0]),
        ],
    );

    assert_close(
        &[out[1].net],
        &[900.0],
        "100 shares marked at the 9.0 floor",
    );
}

/// A cleared `flags` element is a delisting: the stock marks at zero (a total
/// loss, since there is no exit price), the shares stay on the book, and no
/// order against it is executed even while quotes are still finite.
#[test]
fn a_delisted_stock_marks_to_zero_and_blocks_orders() {
    let out = run(
        benchmark(false, 1000.0),
        1,
        &[
            Step::quote(&[10.0], &[10.0]).then_target(&[1.0]),
            Step::quote(&[10.0], &[10.0]).then_flags(&[false]),
            Step::quote(&[10.0], &[10.0])
                .then_flags(&[false])
                .then_target(&[0.0]),
        ],
    );

    assert_close(&[out[1].net], &[0.0], "written off");
    assert_close(&out[1].positions, &[100.0], "shares still held");
    assert_close(&out[2].positions, &[100.0], "and untradable");
}

// ---------------------------------------------------------------------------
// Dividends
// ---------------------------------------------------------------------------

/// Cash dividends credit the cash account and share dividends scale the
/// position; neither is reinvested on the spot. Both leave net value
/// continuous across the ex-date, which is the invariant that pins the
/// convention: a 1.00 cash dividend against a 10.00 close moves the quote to
/// 9.00, and a 1-for-1 bonus issue halves it.
#[test]
fn dividends_credit_cash_and_scale_positions_continuously() {
    let out = run(
        benchmark(false, 1000.0),
        1,
        &[
            Step::quote(&[10.0], &[10.0]).then_target(&[1.0]),
            Step::quote(&[9.0], &[9.0]).then_dividend(&[0.0], &[1.0]),
            Step::quote(&[4.5], &[4.5]).then_dividend(&[1.0], &[0.0]),
        ],
    );

    assert_close(&[out[1].cash], &[100.0], "100 shares × 1.00 to cash");
    assert_close(
        &out[1].positions,
        &[100.0],
        "cash dividends do not reinvest",
    );
    assert_close(
        &[out[1].net],
        &[1000.0],
        "continuous across the cash ex-date",
    );
    assert_close(&out[2].positions, &[200.0], "1-for-1 doubles the position");
    assert_close(
        &[out[2].net],
        &[1000.0],
        "continuous across the share ex-date",
    );
}

// ---------------------------------------------------------------------------
// Execution policies
// ---------------------------------------------------------------------------

/// `simple` rounds each target to a whole number of lots, so the residue
/// stays in cash: a 1000.00 target at 3.00 a share buys 300 shares, not
/// 333.33.
#[test]
fn simple_rounds_targets_to_whole_lots() {
    let out = run(
        simple(false, 1000.0, 0.0, 0.0, 100.0),
        1,
        &[Step::quote(&[3.0], &[3.0]).then_target(&[1.0])],
    );

    assert_close(&out[0].positions, &[300.0], "333.33 rounds to three lots");
    assert_close(&[out[0].cash], &[100.0], "the residue stays in cash");
}

/// Each fill is charged the *larger* of the flat and proportional fee, and
/// the charge comes out of cash — which a fully-invested target pushes
/// negative, the documented slight leverage.
#[test]
fn fees_charge_the_larger_of_the_flat_and_proportional_terms() {
    let flat = run(
        simple(false, 1000.0, 5.0, 0.001, 1.0),
        1,
        &[Step::quote(&[10.0], &[10.0]).then_target(&[1.0])],
    );
    assert_close(&[flat[0].cash], &[-5.0], "flat 5.00 beats 0.1% of 1000");

    let proportional = run(
        simple(false, 1000.0, 5.0, 0.01, 1.0),
        1,
        &[Step::quote(&[10.0], &[10.0]).then_target(&[1.0])],
    );
    assert_close(
        &[proportional[0].cash],
        &[-10.0],
        "1% of 1000 beats the flat 5.00",
    );
}

/// `random` holds an equal-value subset of exactly `portfolio_size` names
/// drawn from the weighted candidates, and the draw is a pure function of the
/// seed — the property that makes a stochastic baseline reproducible.
#[test]
fn random_holds_a_seeded_equal_value_subset() {
    let script = [Step::quote(&[10.0; 5], &[10.0; 5]).then_target(&[0.2; 5])];
    let first = run(random(false, 1000.0, 0.0, 0.0, 1.0, 2, 7), 5, &script);
    let again = run(random(false, 1000.0, 0.0, 0.0, 1.0, 2, 7), 5, &script);

    let held: Vec<f64> = first[0]
        .positions
        .iter()
        .copied()
        .filter(|&x| x != 0.0)
        .collect();
    assert_eq!(
        held.len(),
        2,
        "exactly `portfolio_size` names: {:?}",
        first[0]
    );
    assert_close(&held, &[50.0, 50.0], "500 of value in each");
    assert_eq!(first[0].positions, again[0].positions, "seed-deterministic");
}

/// The engine drops an order aimed at a side that quotes infinity, whatever
/// the policy asked for — the last line of defence behind each policy's own
/// price guard.
#[test]
fn orders_into_an_infinite_side_are_dropped() {
    /// Always tries to sell one share of stock 0, price be damned.
    struct AlwaysSell;
    impl Exec for AlwaysSell {
        fn orders(
            &mut self,
            _: &[f64],
            _: &[f64],
            _: &[f64],
            _: f64,
            _: &[f64],
        ) -> Vec<(usize, f64)> {
            vec![(0, -1.0)]
        }
    }

    let out = run(
        Fixed::new(AlwaysSell, false, 0.0, 0.0, 0.0),
        1,
        &[
            // Unsellable: the sell is dropped and the book stays flat.
            Step::quote(&[-INF], &[9.0]).then_target(&[0.0]),
            // Sellable: the same order goes through (short, since the book is
            // flat — the engine sizes nothing itself).
            Step::quote(&[9.0], &[9.0]).then_target(&[0.0]),
        ],
    );

    assert_close(&out[0].positions, &[0.0], "dropped into a -inf bid");
    assert_close(&out[1].positions, &[-1.0], "executed against a real bid");
    assert_close(&[out[1].cash], &[9.0], "proceeds credited at the bid");
}

/// A `NaN` quote is a data error, not a quiet stock: quiescence is the signal,
/// and an untradable side is `±inf`, so the engine refuses to guess.
#[test]
#[should_panic(expected = "quotes must not be NaN")]
fn a_nan_quote_is_a_data_error() {
    run(
        benchmark(false, 1000.0),
        1,
        &[Step::quote(&[f64::NAN], &[10.0])],
    );
}
