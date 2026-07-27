//! Integration tests for [`tradingflow::operators::event`] — the operators
//! whose behaviour is defined in terms of the *notification flags* rather than
//! the values on their edges.
//!
//! A wire carries `(notify, value)`. `notify == true` means "a new event
//! arrived this generation, and `value` is its payload"; `notify == false`
//! means "nothing arrived, and `value` is still the payload of the last event".
//! Everything in this module is an assertion about which of those two a node
//! produces, so the tests here mostly probe *whether a node recomputed*, not
//! what it computed. The [`count`] probe answers exactly that question: it is
//! an [`Operator`](tradingflow::graph::Operator), so the blanket
//! `Operator -> Segment` impl only calls its `compute` when at least one input
//! notified — its value is the number of notifications it has received, and it
//! is a far sharper instrument than inferring recomputation from output values
//! (which are *carried*, not cleared, when a node does not run).
//!
//! Two graph-level facts the tests below lean on:
//!
//! * A node whose upstream cone contains no poked source is never scheduled at
//!   all, so its output value stays byte-identical and its flag stays clear.
//! * All notify flags are cleared when the graph is built, so the first
//!   generation is not special: an unpoked source does not "notify" just
//!   because its initial output was published at build time.

use tradingflow::data::ArrayView;
use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::{array, elem, event, series};

use crate::harness::*;

/// The predicate shared by the [`event::filter`] tests: a rank-0 sample passes
/// when it exceeds three.
fn over_three(a: ArrayView<'_, f64, 0>) -> bool {
    a.to_contiguous()[0] > 3.0
}

// ---------------------------------------------------------------------------
// filter — suppressing a notification
// ---------------------------------------------------------------------------

/// The load-bearing cutoff proof: `filter` suppresses the notification when its
/// predicate fails, and a suppressed tick must not advance anything downstream.
/// A `series::record_all` behind the filter therefore records the passing
/// values *only*, each stamped with its own tick — if a suppressed tick leaked
/// a notification the record would grow an extra row and the whole event model
/// (filter as a stream cutoff, not a value mask) would be broken.
#[test]
fn filter_records_only_passing_values() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::scalar(0.0_f64));
    let kept = b.segment(event::filter(over_three), srcv);
    let rec = b.segment(series::record_all(), kept);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, v) in [1.0, 5.0, 2.0, 10.0].into_iter().enumerate() {
        *g.state_mut(src) = scalar(v);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }

    assert_eq!(
        series_vals(g.view(rec)),
        vec![5.0, 10.0],
        "only the passing values are recorded"
    );
    assert_eq!(
        g.view(rec).instants(),
        &[nano(2), nano(4)],
        "each passing value keeps the instant of its own tick"
    );
}

/// The same cutoff seen through a recompute probe rather than through values:
/// the `count` behind the filter advances on passing ticks and *only* on
/// passing ticks. This is the precise statement of "a suppressed tick does not
/// advance anything downstream" — a stateful consumer that merely ignored the
/// suppressed value (rather than never running) would still fail here.
#[test]
fn filter_suppression_does_not_advance_downstream() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::scalar(0.0_f64));
    let kept = b.segment(event::filter(over_three), srcv);
    let probe = b.segment(count::<0>(), kept);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (poked value, expected number of downstream recomputes so far)
    let ticks: &[(f64, usize)] = &[
        (1.0, 0),
        (5.0, 1),
        (2.0, 1),
        (10.0, 2),
        (4.0, 3),
        (0.5, 3),
        (3.0, 3), // the predicate is strict: 3.0 does not pass `> 3.0`
    ];
    for (i, &(v, expected)) in ticks.iter().enumerate() {
        *g.state_mut(src) = scalar(v);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), expected, "tick {i}: poked {v}");
    }
}

/// The flip side of the cutoff under value-level events: a suppressed (or
/// merely scheduled-but-quiet) `filter` presents the quiescent all-NaN form,
/// never the rejected value and never a stale copy of the last passing one —
/// the port-level "carry the last passing value" contract of the old flag
/// model is gone by design, and holding a last-known value is an explicit
/// downstream `hold`/record now.
#[test]
fn filter_is_quiescent_between_passing_events() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::scalar(0.0_f64));
    let kept = b.segment(event::filter(over_three), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, v) in [1.0, 5.0, 2.0, 10.0].into_iter().enumerate() {
        *g.state_mut(src) = scalar(v);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert!(
            vals(g.view(kept))[0].is_nan(),
            "tick {i}: the wire is quiescent after every generation"
        );
    }
}

// ---------------------------------------------------------------------------
// clock — forwarding a notification as a bare pulse
// ---------------------------------------------------------------------------

/// `clock` forwards its input's notification as a `UnitPort` pulse and drops
/// the payload — including the payload's *shape*: a rank-1 `[3]` array source
/// here drives a clock that gates a rank-0 stream, which only type-checks
/// because nothing but the edge survives the conversion. The pulse fires
/// exactly on the generations the beat source was poked: not when an unrelated
/// source moves, not on an idle generation.
#[test]
fn clock_pulses_exactly_when_its_input_notifies() {
    let mut b = Builder::new();
    let (beat, beatv) = b.source(array::from_parts([3], vec![0.0_f64; 3].into()));
    let (data, datav) = b.source(array::scalar(0.0_f64));
    let pulse = b.segment(event::as_clock(), beatv);
    let sampled = b.segment(event::sample(), (pulse, datav));
    let probe = b.segment(count::<0>(), sampled);
    let rec = b.segment(series::record_all(), sampled);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (beat poke, data poke, expected pulses so far, expected last sample)
    type Tick = (Option<[f64; 3]>, Option<f64>, usize, f64);
    let ticks: &[Tick] = &[
        (Some([1.0, 1.0, 1.0]), Some(10.0), 1, 10.0),
        (None, Some(20.0), 1, 10.0), // data alone is not a beat
        (Some([2.0, 2.0, 2.0]), None, 2, 20.0), // beat alone pulses
        (None, None, 2, 20.0),       // idle
        (Some([3.0, 3.0, 3.0]), Some(30.0), 3, 30.0),
    ];
    for (i, &(bp, dp, pulses, value)) in ticks.iter().enumerate() {
        if let Some(row) = bp {
            *g.state_mut(beat) = arr([3], row);
        }
        if let Some(v) = dp {
            *g.state_mut(data) = scalar(v);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), pulses, "tick {i}: pulse count");
        // The sampled wire itself is quiescent (NaN) after every generation;
        // the gated record holds the event history, so its last row is the
        // latest sampled value.
        let last = series_vals(g.view(rec)).last().copied().unwrap_or(f64::NAN);
        assert_close(&[last], &[value], &format!("tick {i}"));
    }
}

/// `clock` forwards the *flag*, so a suppression upstream of it is a
/// suppression downstream of it: a `filter -> clock -> resample` chain samples
/// only on the ticks the predicate accepted. This is the composition that makes
/// the event operators useful — "resample stream `x` whenever `y` passes a
/// test" is spelled by putting a clock between the two — and it only works if
/// `clock` refuses to invent a pulse for a quiet input.
#[test]
fn clock_relays_a_filters_suppression() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::scalar(0.0_f64));
    let kept = b.segment(event::filter(over_three), srcv);
    let pulse = b.segment(event::as_clock(), kept);
    let sampled = b.segment(event::sample(), (pulse, srcv));
    let probe = b.segment(count::<0>(), sampled);
    let rec = b.segment(series::record_all(), sampled);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (poked value, expected pulses so far, expected last sample)
    let nan = f64::NAN;
    let ticks: &[(f64, usize, f64)] = &[
        (1.0, 0, nan), // rejected: no pulse, the sampler never ran
        (5.0, 1, 5.0),
        (2.0, 1, 5.0),
        (10.0, 2, 10.0),
        (4.0, 3, 4.0),
        (0.5, 3, 4.0),
    ];
    for (i, &(v, pulses, value)) in ticks.iter().enumerate() {
        *g.state_mut(src) = scalar(v);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), pulses, "tick {i}: poked {v}");
        let last = series_vals(g.view(rec)).last().copied().unwrap_or(f64::NAN);
        assert_close(&[last], &[value], &format!("tick {i}"));
    }
}

// ---------------------------------------------------------------------------
// resample — emitting on a clock pulse only
// ---------------------------------------------------------------------------

/// `resample` emits its data input when — and only when — the leading clock
/// port notifies. Both directions matter: a pulse emits even though the data
/// port is quiet (the point of resampling: lift a slow stream onto a fast
/// grid), and a data arrival is silent without a pulse (the data stream must
/// not smuggle its own tick onto the output).
#[test]
fn resample_emits_only_on_a_clock_pulse() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::scalar(0.0_f64));
    let (tick, tickv) = b.source(clock());
    let sampled = b.segment(event::sample(), (tickv, srcv));
    let probe = b.segment(count::<0>(), sampled);
    let rec = b.segment(series::record_all(), sampled);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (pulse?, data poke, expected emissions so far, expected last sample)
    let nan = f64::NAN;
    type Tick = (bool, Option<f64>, usize, f64);
    let ticks: &[Tick] = &[
        (false, Some(1.0), 0, nan), // data alone: silent, nothing sampled yet
        (true, None, 1, 1.0),       // pulse alone: emits the data from tick 0
        (true, Some(2.0), 2, 2.0),  // pulse + data: emits the fresh value
        (false, None, 2, 2.0),      // idle
        (false, Some(3.0), 2, 2.0), // data alone again: still silent
        (true, None, 3, 3.0),       // pulse: emits the value from tick 4
    ];
    for (i, &(pulse, poke, emissions, value)) in ticks.iter().enumerate() {
        if let Some(v) = poke {
            *g.state_mut(src) = scalar(v);
        }
        if pulse {
            // Touching the clock cell's state marks it dirty; the value is
            // irrelevant, the notification is the whole payload.
            let _ = g.state_mut(tick);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), emissions, "tick {i}: emission count");
        let last = series_vals(g.view(rec)).last().copied().unwrap_or(f64::NAN);
        assert_close(&[last], &[value], &format!("tick {i}"));
    }
}

/// Regression, pinned on its own because it was a real bug: a data arrival with
/// no clock pulse must be *completely* silent. The node is scheduled (its cone
/// is dirty, so `Resample::compute` does run) — the silence has to come from
/// the operator emitting the quiescent all-NaN form, not from the scheduler
/// skipping it. The trailing pulse keeps the test honest: without it a
/// disconnected graph would also "pass".
#[test]
fn resample_stays_silent_without_a_clock_pulse() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::scalar(0.0_f64));
    let (tick, tickv) = b.source(clock());
    let sampled = b.segment(event::sample(), (tickv, srcv));
    let probe = b.segment(count::<0>(), sampled);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, v) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
        *g.state_mut(src) = scalar(v);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(
            g.view(probe),
            0,
            "tick {i}: data without a pulse propagated"
        );
        assert!(
            vals(g.view(sampled))[0].is_nan(),
            "tick {i}: output moved without a pulse",
        );
    }

    let _ = g.state_mut(tick);
    g.stabilize(&mut pool, &nano(5));
    assert_eq!(g.view(probe), 1, "the sampler is actually wired up");
}

/// A pulse emits the latest data value even though that value arrived in an
/// earlier generation, and it emits on *every* pulse — including when the data
/// has not changed since the last one. That is what makes `resample` a
/// grid-builder rather than a change-detector: the downstream record gets one
/// row per pulse, stamped with the pulse's instant (not the instant the data
/// arrived), so a slow stream sampled on a fast clock produces a dense series
/// with repeated values.
#[test]
fn resample_emits_the_latest_value_from_an_earlier_generation() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::scalar(0.0_f64));
    let (tick, tickv) = b.source(clock());
    let sampled = b.segment(event::sample(), (tickv, srcv));
    let rec = b.segment(series::record_all(), sampled);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Two data-only generations: nothing is emitted, nothing is recorded.
    for (i, v) in [1.0, 2.0].into_iter().enumerate() {
        *g.state_mut(src) = scalar(v);
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }
    assert_eq!(g.view(rec).len(), 0, "data alone must not record");

    // Two pulse-only generations: both emit the value from generation 2.
    for t in 3..=4 {
        let _ = g.state_mut(tick);
        g.stabilize(&mut pool, &nano(t));
    }
    assert_eq!(
        series_vals(g.view(rec)),
        vec![2.0, 2.0],
        "every pulse re-emits the latest data"
    );
    assert_eq!(
        g.view(rec).instants(),
        &[nano(3), nano(4)],
        "rows are stamped with the pulse, not with the data's arrival"
    );
}

// ---------------------------------------------------------------------------
// joins over state vs. event edges — NaN-filling the inputs with no events
// ---------------------------------------------------------------------------

/// There is one join operator now, and the carry/sync contrast lives in the
/// *wiring*: a join over state cells reads each input's retained value (the
/// carry), while the same join over `as_event`-badged edges reads NaN for
/// every input whose source did not fire (the sync fill). Which one you want
/// depends on the question: the state join answers "what is the latest value
/// of each element", the event join answers "what arrived on this tick", and
/// silently getting the first when you meant the second is a stale-data bug
/// that a value assertion on a single generation cannot see.
#[test]
fn a_join_over_event_edges_nan_fills_while_a_state_join_carries() {
    let nan = f64::NAN;
    let mut b = Builder::new();
    let (s0, s0v) = b.source(array::scalar(0.0_f64));
    let (s1, s1v) = b.source(array::scalar(0.0_f64));
    let (s2, s2v) = b.source(array::scalar(0.0_f64));
    let e0 = b.segment(event::as_event(), s0v);
    let e1 = b.segment(event::as_event(), s1v);
    let e2 = b.segment(event::as_event(), s2v);
    let stacked = b.segment(array::stack::<f64, 0, 1>(0), &[s0v, s1v, s2v][..]);
    let synced = b.segment(array::stack::<f64, 0, 1>(0), &[e0, e1, e2][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (per-source poke, expected state join, expected event join)
    type Tick = ([Option<f64>; 3], [f64; 3], [f64; 3]);
    let ticks: &[Tick] = &[
        (
            [Some(1.0), Some(2.0), Some(3.0)],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ),
        ([None, Some(20.0), None], [1.0, 20.0, 3.0], [nan, 20.0, nan]),
        (
            [Some(10.0), None, None],
            [10.0, 20.0, 3.0],
            [10.0, nan, nan],
        ),
        (
            [Some(11.0), Some(21.0), Some(31.0)],
            [11.0, 21.0, 31.0],
            [11.0, 21.0, 31.0],
        ),
    ];
    for (i, (poke, want_stack, want_sync)) in ticks.iter().enumerate() {
        for (h, v) in [s0, s1, s2].into_iter().zip(poke) {
            if let Some(v) = v {
                *g.state_mut(h) = scalar(*v);
            }
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_close(
            &vals(g.view(stacked)),
            want_stack,
            &format!("tick {i}: stack"),
        );
        assert_close(&vals(g.view(synced)), want_sync, &format!("tick {i}: sync"));
        if poke.iter().all(Option::is_some) {
            // When every input notified there is nothing to NaN-fill, so the
            // two joins must agree down to the bit.
            assert_same_bits(
                g.view(stacked),
                g.view(synced),
                &format!("tick {i}: all inputs notified"),
            );
        }
    }
}

/// The same contrast for `concat`, on rank-1 inputs joined along an existing
/// axis: the NaN fill covers the whole contribution of a quiet input (both of
/// its elements), not just a scalar slot, so the fill is expressed in terms of
/// each input's chunk of the output rather than element-by-element.
#[test]
fn a_concat_over_event_edges_nan_fills_while_a_state_concat_carries() {
    let nan = f64::NAN;
    let mut b = Builder::new();
    let (a, av) = b.source(array::from_parts([2], vec![0.0_f64; 2].into()));
    let (c, cv) = b.source(array::from_parts([2], vec![0.0_f64; 2].into()));
    let ea = b.segment(event::as_event(), av);
    let ec = b.segment(event::as_event(), cv);
    let joined = b.segment(array::concat::<f64, 1>(0), &[av, cv][..]);
    let synced = b.segment(array::concat::<f64, 1>(0), &[ea, ec][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (poke of a, poke of c, expected state join, expected event join)
    type Tick = (Option<[f64; 2]>, Option<[f64; 2]>, [f64; 4], [f64; 4]);
    let ticks: &[Tick] = &[
        (
            Some([1.0, 2.0]),
            Some([3.0, 4.0]),
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ),
        (
            Some([10.0, 20.0]),
            None,
            [10.0, 20.0, 3.0, 4.0],
            [10.0, 20.0, nan, nan],
        ),
        (
            None,
            Some([30.0, 40.0]),
            [10.0, 20.0, 30.0, 40.0],
            [nan, nan, 30.0, 40.0],
        ),
    ];
    for (i, (pa, pc, want_concat, want_sync)) in ticks.iter().enumerate() {
        if let Some(row) = pa {
            *g.state_mut(a) = arr1(*row);
        }
        if let Some(row) = pc {
            *g.state_mut(c) = arr1(*row);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_close(
            &vals(g.view(joined)),
            want_concat,
            &format!("tick {i}: concat"),
        );
        assert_close(&vals(g.view(synced)), want_sync, &format!("tick {i}: sync"));
    }
}

// ---------------------------------------------------------------------------
// The carry contract and coalescing
// ---------------------------------------------------------------------------

/// The general carry contract, in both halves. A join reads *every* input each
/// generation, including the ones that did not notify, so poking one source
/// produces an output that mixes the fresh value with the carried ones. And an
/// idle generation — no source poked at all — never schedules the join, so its
/// output stays byte-identical to the previous one: the assertion is on raw bit
/// patterns because "unchanged" has to include *which* `NaN` is sitting in the
/// buffer, and because a join that quietly recomputed from stale inputs would
/// still produce equal-looking floats.
///
/// The second half also pins the one thing about the event join that is easy
/// to get backwards: it NaN-fills per *input*, not per generation, so an idle
/// generation (the join never scheduled) carries its previous output (NaNs
/// included) rather than blanking the whole cross-section.
#[test]
fn a_join_carries_unnotified_inputs_and_freezes_when_idle() {
    let nan = f64::NAN;
    let mut b = Builder::new();
    let (s0, s0v) = b.source(array::scalar(0.0_f64));
    let (s1, s1v) = b.source(array::scalar(0.0_f64));
    let (s2, s2v) = b.source(array::scalar(0.0_f64));
    let e0 = b.segment(event::as_event(), s0v);
    let e1 = b.segment(event::as_event(), s1v);
    let e2 = b.segment(event::as_event(), s2v);
    let stacked = b.segment(array::stack::<f64, 0, 1>(0), &[s0v, s1v, s2v][..]);
    let synced = b.segment(array::stack::<f64, 0, 1>(0), &[e0, e1, e2][..]);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Generation 1: all three fire, and one of them carries a NaN payload of
    // its own so the byte-identity assertion below has a NaN to preserve.
    *g.state_mut(s0) = scalar(1.0);
    *g.state_mut(s1) = scalar(nan);
    *g.state_mut(s2) = scalar(3.0);
    g.stabilize(&mut pool, &nano(1));
    assert_close(&vals(g.view(stacked)), &[1.0, nan, 3.0], "gen 1: stack");
    assert_close(&vals(g.view(synced)), &[1.0, nan, 3.0], "gen 1: sync");

    // Generation 2: only `s2` fires. The plain join still reads `s0` and `s1`
    // and carries both (including the NaN that is a genuine payload); the
    // `_sync` join overwrites them with the missing marker.
    *g.state_mut(s2) = scalar(7.0);
    g.stabilize(&mut pool, &nano(2));
    assert_close(
        &vals(g.view(stacked)),
        &[1.0, nan, 7.0],
        "gen 2: stack carries the quiet inputs",
    );
    assert_close(
        &vals(g.view(synced)),
        &[nan, nan, 7.0],
        "gen 2: sync fills the quiet inputs",
    );

    // Generations 3 and 4 are idle: neither join is scheduled at all, so both
    // outputs must be byte-for-byte what generation 2 left behind.
    let frozen_stack = bits(g.view(stacked));
    let frozen_sync = bits(g.view(synced));
    for t in 3..=4 {
        g.stabilize(&mut pool, &nano(t));
        assert_eq!(
            bits(g.view(stacked)),
            frozen_stack,
            "gen {t}: stack drifted"
        );
        assert_eq!(bits(g.view(synced)), frozen_sync, "gen {t}: sync drifted");
    }

    // And the graph is still live afterwards — the freeze is idleness, not a
    // node that fell out of the schedule permanently.
    *g.state_mut(s0) = scalar(9.0);
    g.stabilize(&mut pool, &nano(5));
    assert_close(&vals(g.view(stacked)), &[9.0, nan, 7.0], "gen 5: stack");
    assert_close(&vals(g.view(synced)), &[9.0, nan, nan], "gen 5: sync");
}

/// Two sources poked before the same `stabilize` produce **one** pass over the
/// union of their cones, not one pass each: a join downstream of both recomputes
/// exactly once and emits exactly one notification for the generation. This is
/// what makes a generation the unit of time in the graph — if the two pokes were
/// drained separately the join would first fire on a half-updated input pair,
/// and every downstream record would carry a spurious intermediate row.
#[test]
fn coalesced_pokes_recompute_the_union_of_cones_once() {
    let mut b = Builder::new();
    let (a, av) = b.source(array::scalar(0.0_f64));
    let (c, cv) = b.source(array::scalar(0.0_f64));
    let sum = b.segment(elem::add(), (av, cv));
    let probe = b.segment(count::<0>(), sum);
    let rec = b.segment(series::record_all(), sum);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (poke of a, poke of c, expected recomputes so far, expected sum)
    type Tick = (Option<f64>, Option<f64>, usize, f64);
    let ticks: &[Tick] = &[
        (Some(10.0), Some(100.0), 1, 110.0), // both: one recompute, not two
        (Some(20.0), Some(200.0), 2, 220.0),
        (Some(30.0), None, 3, 230.0), // one: the join re-reads the carried `c`
        (None, None, 3, 230.0),       // idle: no recompute at all
        (None, Some(300.0), 4, 330.0),
    ];
    for (i, &(pa, pc, recomputes, sum_value)) in ticks.iter().enumerate() {
        if let Some(v) = pa {
            *g.state_mut(a) = scalar(v);
        }
        if let Some(v) = pc {
            *g.state_mut(c) = scalar(v);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), recomputes, "tick {i}: recompute count");
        assert_close(&vals(g.view(sum)), &[sum_value], &format!("tick {i}"));
        assert_eq!(g.view(rec).len(), recomputes, "tick {i}: recorded rows");
    }

    assert_eq!(
        g.view(rec).instants(),
        &[nano(1), nano(2), nano(3), nano(5)],
        "one row per recomputing generation, and none for the idle one"
    );
}
