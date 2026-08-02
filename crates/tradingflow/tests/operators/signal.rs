//! Integration tests for [`tradingflow::operators::event`] — the signal
//! derivation operators, and the wiring idioms of the signal + state event
//! model.
//!
//! An event stream is a `(signal, values)` pair of wires. The signal marks the
//! generations on which the values wire carries a new batch; between pulses
//! the values wire retains its last batch, so a consumer that ignores the
//! signal reads a held state value. Sampling a wire onto another cadence is
//! *pairing* it with a different signal — there is no sampling operator —
//! and stateful consumers (`record_on` and friends) take the signal as an
//! explicit input and append once per pulse.
//!
//! Two graph-level facts the tests below lean on:
//!
//! * A node whose upstream cone contains no poked source is never scheduled at
//!   all, so its output value stays byte-identical; a scheduled signal node
//!   emits its pulse for that generation and resets to `false` afterwards.
//! * The reset pass runs after every generation, so `g.view` on a signal edge
//!   always reads `false` — pulse assertions go through a downstream counter.

use tradingflow::data::ArrayView;
use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::{array, elem, series, signal};

use crate::harness::*;

/// The predicate shared by the [`event::filter`] tests: a rank-0 sample passes
/// when it exceeds three.
fn over_three(a: ArrayView<'_, f64, 0>) -> bool {
    a.to_contiguous()[0] > 3.0
}

// ---------------------------------------------------------------------------
// filter — suppressing a pulse
// ---------------------------------------------------------------------------

/// The load-bearing cutoff proof: `filter` suppresses the signal when its
/// predicate fails, and a suppressed tick must not advance anything
/// downstream. A `series::record_all` on the filtered signal therefore records
/// the passing values *only*, each stamped with its own tick — if a suppressed
/// tick leaked a pulse the record would grow an extra row and the whole event
/// model (filter as a stream cutoff, not a value mask) would be broken.
#[test]
fn filter_records_only_passing_values() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let kept = b.segment(signal::filter(over_three), srcv);
    let rec = b.segment(series::record_all(), (kept, srcv.1));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, v) in [1.0, 5.0, 2.0, 10.0].into_iter().enumerate() {
        *g.state_mut(src) = v.into();
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

/// The same cutoff seen through a pulse counter rather than through values:
/// the `count` behind the filter advances on passing ticks and *only* on
/// passing ticks. This is the precise statement of "a suppressed tick does not
/// advance anything downstream" for every pulse-gated consumer.
#[test]
fn filter_suppression_does_not_advance_downstream() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let kept = b.segment(signal::filter(over_three), srcv);
    let probe = b.segment(count::<0>(), (kept, srcv.1));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (poked value, expected number of passing pulses so far)
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
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), expected, "tick {i}: poked {v}");
    }
}

/// A signal edge is quiescent between generations: the reset pass lowers it, so
/// `g.view` reads `false` after every stabilize — passing tick or not. The
/// pulse is consumed in-generation by scheduled consumers only.
#[test]
fn a_signal_edge_reads_low_after_every_generation() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let kept = b.segment(signal::filter(over_three), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, v) in [1.0, 5.0, 2.0, 10.0].into_iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert!(!*g.view(srcv.0), "tick {i}: the source signal reads low");
        assert!(!*g.view(kept), "tick {i}: the filtered signal reads low");
    }
}

// ---------------------------------------------------------------------------
// signal — deriving a cadence from a wire
// ---------------------------------------------------------------------------

/// `signal` pulses exactly on the generations its input's producer ran — not
/// when an unrelated source moves, not on an idle generation. Pairing that
/// pulse with an unrelated data wire *is* sampling: the record gets the data
/// wire's current value once per beat, and the payload's shape is dropped
/// entirely (a rank-1 beat source drives a rank-0 sample here).
#[test]
fn signal_pulses_exactly_when_its_input_notifies() {
    let mut b = Builder::new();
    let (beat, (pulse, _beatv)) = b.source(cell([0.0_f64; 3]));
    let (data, datav) = b.source(array::constant(0.0_f64));
    let probe = b.segment(count::<0>(), (pulse, datav));
    let rec = b.segment(series::record_all(), (pulse, datav));
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
            *g.state_mut(beat) = row.into();
        }
        if let Some(v) = dp {
            *g.state_mut(data) = v.into();
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), pulses, "tick {i}: pulse count");
        let last = series_vals(g.view(rec)).last().copied().unwrap_or(f64::NAN);
        assert_close(&[last], &[value], &format!("tick {i}"));
    }
}

/// A filtered signal composes: "sample stream `x` whenever `y` passes a test"
/// is spelled by pairing `x`'s values with the filtered signal of `y`. It only
/// works because `filter` refuses to invent a pulse for a rejected tick.
#[test]
fn a_filtered_signal_samples_another_wire() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let kept = b.segment(signal::filter(over_three), srcv);
    let probe = b.segment(count::<0>(), (kept, srcv.1));
    let rec = b.segment(series::record_all(), (kept, srcv.1));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (poked value, expected pulses so far, expected last sample)
    let nan = f64::NAN;
    let ticks: &[(f64, usize, f64)] = &[
        (1.0, 0, nan), // rejected: no pulse, nothing recorded
        (5.0, 1, 5.0),
        (2.0, 1, 5.0),
        (10.0, 2, 10.0),
        (4.0, 3, 4.0),
        (0.5, 3, 4.0),
    ];
    for (i, &(v, pulses, value)) in ticks.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), pulses, "tick {i}: poked {v}");
        let last = series_vals(g.view(rec)).last().copied().unwrap_or(f64::NAN);
        assert_close(&[last], &[value], &format!("tick {i}"));
    }
}

// ---------------------------------------------------------------------------
// pairing with a foreign signal — the sampling idiom
// ---------------------------------------------------------------------------

/// A record driven by a manual signal appends when — and only when — the signal
/// pulses. Both directions matter: a pulse appends even though the data wire
/// was quiet that generation (the point of sampling: lift a slow stream onto a
/// fast grid), and a data arrival is silent without a pulse (the data stream
/// must not smuggle its own tick onto the output).
#[test]
fn a_paired_signal_gates_the_record() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(0.0_f64));
    let (tick, tickv) = b.source(signal());
    let probe = b.segment(count::<0>(), (tickv, srcv));
    let rec = b.segment(series::record_all(), (tickv, srcv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (pulse?, data poke, expected rows so far, expected last sample)
    let nan = f64::NAN;
    type Tick = (bool, Option<f64>, usize, f64);
    let ticks: &[Tick] = &[
        (false, Some(1.0), 0, nan), // data alone: silent, nothing sampled yet
        (true, None, 1, 1.0),       // pulse alone: samples the data from tick 0
        (true, Some(2.0), 2, 2.0),  // pulse + data: samples the fresh value
        (false, None, 2, 2.0),      // idle
        (false, Some(3.0), 2, 2.0), // data alone again: still silent
        (true, None, 3, 3.0),       // pulse: samples the value from tick 4
    ];
    for (i, &(pulse, poke, emissions, value)) in ticks.iter().enumerate() {
        if let Some(v) = poke {
            *g.state_mut(src) = v.into();
        }
        if pulse {
            // Touching the signal cell's state marks it dirty; the value is
            // irrelevant, the pulse is the whole payload.
            let _ = g.state_mut(tick);
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(probe), emissions, "tick {i}: pulse count");
        let last = series_vals(g.view(rec)).last().copied().unwrap_or(f64::NAN);
        assert_close(&[last], &[value], &format!("tick {i}"));
    }
}

/// Regression, pinned on its own because its ancestor was a real bug: a data
/// arrival with no signal must be *completely* silent. The consumers are
/// scheduled (their cone is dirty, so their `compute` does run) — the silence
/// has to come from the pulse-gate inside the consumer, not from the scheduler
/// skipping it. The trailing pulse keeps the test honest: without it a
/// disconnected graph would also "pass".
#[test]
fn data_without_a_pulse_is_completely_silent() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(0.0_f64));
    let (tick, tickv) = b.source(signal());
    let probe = b.segment(count::<0>(), (tickv, srcv));
    let rec = b.segment(series::record_all(), (tickv, srcv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (i, v) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(
            g.view(probe),
            0,
            "tick {i}: data without a pulse propagated"
        );
        assert_eq!(g.view(rec).len(), 0, "tick {i}: row appended without pulse");
    }

    let _ = g.state_mut(tick);
    g.stabilize(&mut pool, &nano(5));
    assert_eq!(g.view(probe), 1, "the consumers are actually wired up");
}

/// A pulse samples the latest data value even though that value arrived in an
/// earlier generation, and it samples on *every* pulse — including when the
/// data has not changed since the last one. That is what makes signal pairing a
/// grid-builder rather than a change-detector: the record gets one row per
/// pulse, stamped with the pulse's instant (not the instant the data arrived),
/// so a slow stream sampled on a fast signal produces a dense series with
/// repeated values.
#[test]
fn a_pulse_samples_the_latest_value_from_an_earlier_generation() {
    let mut b = Builder::new();
    let (src, srcv) = b.source(array::constant(0.0_f64));
    let (tick, tickv) = b.source(signal());
    let rec = b.segment(series::record_all(), (tickv, srcv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Two data-only generations: nothing is sampled, nothing is recorded.
    for (i, v) in [1.0, 2.0].into_iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }
    assert_eq!(g.view(rec).len(), 0, "data alone must not record");

    // Two pulse-only generations: both sample the value from generation 2.
    for t in 3..=4 {
        let _ = g.state_mut(tick);
        g.stabilize(&mut pool, &nano(t));
    }
    assert_eq!(
        series_vals(g.view(rec)),
        vec![2.0, 2.0],
        "every pulse re-samples the latest data"
    );
    assert_eq!(
        g.view(rec).instants(),
        &[nano(3), nano(4)],
        "rows are stamped with the pulse, not with the data's arrival"
    );
}

// ---------------------------------------------------------------------------
// The carry contract and coalescing
// ---------------------------------------------------------------------------

/// The carry contract, in both halves. A join reads *every* input each
/// generation, including the ones whose sources did not fire, so poking one
/// source produces an output that mixes the fresh value with the carried ones
/// — "which inputs are fresh" lives on the signals, not in the values. And an
/// idle generation — no source poked at all — never schedules the join, so
/// its output stays byte-identical to the previous one: the assertion is on
/// raw bit patterns because "unchanged" has to include *which* `NaN` is
/// sitting in the buffer.
#[test]
fn a_join_carries_unfired_inputs_and_freezes_when_idle() {
    let nan = f64::NAN;
    let mut b = Builder::new();
    let (s0, (c0, s0v)) = b.source(cell(0.0_f64));
    let (s1, s1v) = b.source(array::constant(0.0_f64));
    let (s2, (c2, s2v)) = b.source(cell(0.0_f64));
    let stacked = b.segment(array::stack::<f64, 0, 1>(0), &[s0v, s1v, s2v][..]);
    let pulses0 = b.segment(count::<0>(), (c0, s0v));
    let pulses2 = b.segment(count::<0>(), (c2, s2v));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Generation 1: all three fire, and one of them carries a NaN payload of
    // its own so the byte-identity assertion below has a NaN to preserve.
    *g.state_mut(s0) = (1.0).into();
    *g.state_mut(s1) = (nan).into();
    *g.state_mut(s2) = (3.0).into();
    g.stabilize(&mut pool, &nano(1));
    assert_close(&vals(g.view(stacked)), &[1.0, nan, 3.0], "gen 1: stack");
    assert_eq!((g.view(pulses0), g.view(pulses2)), (1, 1), "gen 1: pulses");

    // Generation 2: only `s2` fires. The join still reads `s0` and `s1` and
    // carries both (including the NaN that is a genuine payload); only `s2`'s
    // signals.
    *g.state_mut(s2) = (7.0).into();
    g.stabilize(&mut pool, &nano(2));
    assert_close(
        &vals(g.view(stacked)),
        &[1.0, nan, 7.0],
        "gen 2: stack carries the quiet inputs",
    );
    assert_eq!((g.view(pulses0), g.view(pulses2)), (1, 2), "gen 2: pulses");

    // Generations 3 and 4 are idle: the join is not scheduled at all, so its
    // output must be byte-for-byte what generation 2 left behind.
    let frozen_stack = bits(g.view(stacked));
    for t in 3..=4 {
        g.stabilize(&mut pool, &nano(t));
        assert_eq!(
            bits(g.view(stacked)),
            frozen_stack,
            "gen {t}: stack drifted"
        );
        assert_eq!(
            (g.view(pulses0), g.view(pulses2)),
            (1, 2),
            "gen {t}: pulses drifted"
        );
    }

    // And the graph is still live afterwards — the freeze is idleness, not a
    // node that fell out of the schedule permanently.
    *g.state_mut(s0) = (9.0).into();
    g.stabilize(&mut pool, &nano(5));
    assert_close(&vals(g.view(stacked)), &[9.0, nan, 7.0], "gen 5: stack");
    assert_eq!((g.view(pulses0), g.view(pulses2)), (2, 2), "gen 5: pulses");
}

/// The synchronized join in the factored model: `stack` over the value wires
/// carries the quiet legs' retained values, while `signal_stack` over the same
/// legs' signals says *which* slots actually fired this generation — the
/// event-granular batch face (values where the signal element pulsed, NaN
/// elsewhere) is their pairing. Which one you want depends on the question:
/// the state join answers "what is the latest value of each leg", the batch
/// face answers "what arrived on this tick", and silently getting the first
/// when you meant the second is a stale-data bug a single-generation value
/// assertion cannot see.
#[test]
fn a_signal_stack_join_nan_fills_while_a_state_join_carries() {
    let nan = f64::NAN;
    let mut b = Builder::new();
    let (s0, (s0v_signal, s0v)) = b.source(cell(0.0_f64));
    let (s1, (s1v_signal, s1v)) = b.source(cell(0.0_f64));
    let (s2, (s2v_signal, s2v)) = b.source(cell(0.0_f64));
    let stacked = b.segment(array::stack::<f64, 0, 1>(0), &[s0v, s1v, s2v][..]);
    let signals = b.segment(
        signal::as_signal_map(array::stack::<bool, 0, 1>(0)),
        &[s0v_signal, s1v_signal, s2v_signal][..],
    );
    let synced = b.segment(batch_face(), (signals, stacked));
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
                *g.state_mut(h) = (*v).into();
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
            // When every input fired there is nothing to NaN-fill, so the
            // two joins must agree down to the bit.
            assert_same_bits(
                g.view(stacked),
                g.view(synced),
                &format!("tick {i}: all inputs fired"),
            );
        }
    }
}

/// The same contrast for `concat`, on rank-1 inputs joined along an existing
/// axis: the NaN fill covers the whole contribution of a quiet input (both of
/// its elements), not just a scalar slot, so the joined signal array is
/// expressed in terms of each input's chunk of the output rather than
/// element-by-element — an `as_signal_map` over the pair of leg signals.
#[test]
fn a_signal_concat_nan_fills_while_a_state_concat_carries() {
    let nan = f64::NAN;
    let mut b = Builder::new();
    let (a, (av_signal, av)) = b.source(cell([0.0_f64; 2]));
    let (c, (cv_signal, cv)) = b.source(cell([0.0_f64; 2]));
    let joined = b.segment(array::concat::<f64, 1>(0), &[av, cv][..]);
    let signals = b.segment(
        signal::as_signal_map(array::array_binary_map(
            |a: ArrayView<bool, 0>, c: ArrayView<bool, 0>| {
                tradingflow::data::Array::from_parts([4], vec![*a, *a, *c, *c].into())
            },
        )),
        (av_signal, cv_signal),
    );
    let synced = b.segment(batch_face(), (signals, joined));
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
            *g.state_mut(a) = (*row).into();
        }
        if let Some(row) = pc {
            *g.state_mut(c) = (*row).into();
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

/// Two sources poked before the same `stabilize` produce **one** pass over the
/// union of their cones, not one pass each: a join downstream of both recomputes
/// exactly once and its derived signals exactly once for the generation.
/// This is what makes a generation the unit of time in the graph — if the two
/// pokes were drained separately the join would first fire on a half-updated
/// input pair, and every downstream record would carry a spurious intermediate
/// row.
#[test]
fn coalesced_pokes_recompute_the_union_of_cones_once() {
    let mut b = Builder::new();
    let (a, (sig_a, av)) = b.source(cell(0.0_f64));
    let (c, (sig_c, cv)) = b.source(cell(0.0_f64));
    let sum = b.segment(elem::add(), (av, cv));
    let probe = b.segment(runs::<0>(), sum);
    // The union of the two source cadences, spelled explicitly.
    let sum_signal = b.segment(signal::or(), (sig_a, sig_c));
    let rec = b.segment(series::record_all(), (sum_signal, sum));
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
            *g.state_mut(a) = v.into();
        }
        if let Some(v) = pc {
            *g.state_mut(c) = v.into();
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
