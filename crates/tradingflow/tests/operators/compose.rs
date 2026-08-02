//! Cross-cutting operator tests: everything that only breaks when operators are
//! put *together*.
//!
//! The single-operator modules pin what each operator computes in isolation.
//! This module pins the four properties that only a composition can violate:
//!
//! 1. **Fusion transparency.** A `segment!` collapses a chain into one graph
//!    node, so the whole chain reruns as a unit and the inner operators see
//!    signals produced *inside* the node rather than at scheduler
//!    boundaries. The invariant is that this is unobservable: a fused segment
//!    is tick-for-tick bit-identical to the same operators as separate nodes.
//! 2. **Composed signals.** The design-brief moving-average crossover, written
//!    against an independent reference model, over a path long enough that the
//!    private records compact underneath the windows.
//! 3. **Graph structure.** Rejoining cones and coalesced multi-source
//!    generations: a node reached by two paths must run exactly once.
//! 4. **Parallel execution.** `Pool::new(N > 0)` runs the cone on real worker
//!    threads; the results must not depend on `N`.

use tradingflow::data::ArrayView;
use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::{array, elem, rolling, series, signal};
use tradingflow::ports::{ArrayPort, SignalPort};
use tradingflow::segment;

use crate::harness::*;

/// The gate predicate used by the fusion tests: a row survives if it carries
/// any real data at all.
fn any_finite(a: ArrayView<'_, f64, 1>) -> bool {
    a.to_contiguous().iter().any(|x| x.is_finite())
}

/// Reference model for the moving-average crossover
/// `MA(x, fast) − MA(x, slow) > 0 AND NOT LAG(MA(x, fast) − MA(x, slow), 1) > 0`.
///
/// Each MA is the mean of the ticks seen so far, capped at its window (partial
/// windows are emitted — the operators do no warm-up gating), and the spread's
/// 1-tick lag reads the previously computed spread.
///
/// `!(prev > 0.0)` is the `NOT LAG(..) > 0` of the formula, and the negation is
/// load-bearing rather than a clumsy spelling of `prev <= 0.0`: `prev` is `NaN`
/// on the first tick (the lag's fill value), and `!(NaN > 0.0)` is `true` —
/// which is what `elem::not() @ (elem::gt() @ (nan, 0))` produces — whereas
/// `prev <= 0.0` would be `false`.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn crossover_reference(path: &[f64], fast: usize, slow: usize) -> Vec<bool> {
    let mean = |t: usize, w: usize| -> f64 {
        let w = w.min(t);
        path[t - w..t].iter().sum::<f64>() / w as f64
    };
    let mut prev = f64::NAN;
    let mut out = Vec::with_capacity(path.len());
    for t in 1..=path.len() {
        let d = mean(t, fast) - mean(t, slow);
        out.push(d > 0.0 && !(prev > 0.0));
        prev = d;
    }
    out
}

// ---------------------------------------------------------------------------
// `segment!` fusion
// ---------------------------------------------------------------------------

/// A fused arithmetic/rolling chain is tick-for-tick bit-identical to the same
/// operators wired as separate nodes.
///
/// Worth pinning because fusion changes *when* the inner operators run:
/// unfused, the scheduler skips a whole node whose upstream sources are clean;
/// fused, the node always runs as a unit and each inner signal-gated operator
/// must ignore the generations without its pulse. The two sources are poked
/// independently so that most generations exercise one of those cases, and the
/// table carries `NaN` rows so the comparison is over exact bit patterns
/// rather than "both are some NaN".
#[test]
fn fused_rolling_chain_matches_unfused_nodes() {
    let nan = f64::NAN;

    let fused = segment!(
        |cx: SignalPort<0>, x: ArrayPort<f64, 1>, cy: SignalPort<0>, y: ArrayPort<f64, 1>|
            -> ArrayPort<f64, 1> {
            let d = elem::sub()
                @ (rolling::mean(3, 1) @ (cx, x), rolling::mean(5, 1) @ (cy, y));
            let cd = signal::or() @ (cx, cy);
            let prev = rolling::lag(1) @ (cd, d);
            elem::abs() @ (elem::add() @ (d, prev))
        }
    );

    let mut b = Builder::new();
    let (sx, (cx, xv)) = b.source(cell([nan; 2]));
    let (sy, (cy, yv)) = b.source(cell([nan; 2]));

    // Reference: the same chain as separate nodes.
    let fast = b.segment(rolling::mean(3, 1), (cx, xv));
    let slow = b.segment(rolling::mean(5, 1), (cy, yv));
    let d = b.segment(elem::sub(), (fast, slow));
    let cd = b.segment(signal::or(), (cx, cy));
    let prev = b.segment(rolling::lag(1), (cd, d));
    let sum = b.segment(elem::add(), (d, prev));
    let plain = b.segment(elem::abs(), sum);

    let f_out = b.segment(fused, (cx, xv, cy, yv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // `None` = that source is not poked this generation. A hand-written head
    // covers the interesting generations; the deterministic tail is long enough
    // (with a staggered poke schedule) to fill and slide both windows several
    // times over, so the comparison is sensitive to every operator's state.
    type Tick = (Option<[f64; 2]>, Option<[f64; 2]>);
    let head: Vec<Tick> = vec![
        (Some([1.0, 10.0]), Some([2.0, 20.0])),
        (Some([3.0, 30.0]), None), // only x fires
        (None, Some([4.0, 40.0])), // only y fires
        (None, None),              // idle generation
        (Some([nan, 50.0]), None), // NaN on one element only
        (Some([5.0, 60.0]), Some([nan, nan])),
        (None, None), // idle generation after a NaN tick
        (Some([6.0, 70.0]), Some([7.0, 80.0])),
        (Some([8.0, 90.0]), None),
    ];
    let (px, py) = (quarter_path(3, 24), quarter_path(4, 24));
    let tail = (0..24).map(|i| -> Tick {
        (
            (i % 4 != 3).then(|| [px[i], px[i] + 0.5]),
            (i % 3 != 2).then(|| [py[i], py[i] - 0.25]),
        )
    });
    let ticks: Vec<Tick> = head.into_iter().chain(tail).collect();
    let mut seen = Vec::new();
    for (i, (x, y)) in ticks.iter().enumerate() {
        if let Some(x) = x {
            *g.state_mut(sx) = x.to_vec().into();
        }
        if let Some(y) = y {
            *g.state_mut(sy) = y.to_vec().into();
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_same_bits(g.view(plain), g.view(f_out), &format!("tick {i}"));
        seen.push(vals(g.view(plain)));
    }
    // Guard against a vacuous pass: two chains that both emit a constant (or
    // only `NaN`) would agree bit-for-bit while proving nothing.
    assert!(
        seen.iter().flatten().any(|v| v.is_finite()) && seen.iter().any(|r| *r != seen[0]),
        "the tick table must drive varying, finite output: {seen:?}"
    );
}

/// A fused segment that *gates* (`event::filter`) and returns **two** outputs is
/// tick-for-tick bit-identical to the same operators as separate nodes.
///
/// This is the fusion case most likely to break. A gate's whole job is to
/// withhold a pulse, and inside a fused node that pulse never reaches the
/// scheduler — it is consumed by the next inner operator instead. A
/// multi-output segment additionally makes the macro's `route` project two
/// wires out of one environment, one of which (`select_at`) is a zero-copy
/// view derived from the other. The table drives all four gate states (both
/// pass, either one blocked, both blocked) plus idle generations.
#[test]
fn fused_gated_multi_output_matches_unfused_nodes() {
    let nan = f64::NAN;

    let fused = segment!(
        |cp: SignalPort<0>, p: ArrayPort<f64, 1>, cq: SignalPort<0>, q: ArrayPort<f64, 1>|
            -> (ArrayPort<f64, 1>, ArrayPort<f64, 0>) {
            let gp = signal::filter(any_finite) @ (cp, p);
            let gq = signal::filter(any_finite) @ (cq, q);
            let cs = signal::and() @ (gp, gq);
            let m = rolling::mean(3, 1) @ (cs, elem::add() @ (p, q));
            (m, array::select_at(0, 0) @ m)
        }
    );

    let mut b = Builder::new();
    let (sp, (cp, pv)) = b.source(cell([nan; 2]));
    let (sq, (cq, qv)) = b.source(cell([nan; 2]));

    // Reference: the same chain as separate nodes.
    let gp = b.segment(signal::filter(any_finite), (cp, pv));
    let gq = b.segment(signal::filter(any_finite), (cq, qv));
    let cs = b.segment(signal::and(), (gp, gq));
    let s = b.segment(elem::add(), (pv, qv));
    let m = b.segment(rolling::mean(3, 1), (cs, s));
    let head = b.segment(array::select_at(0, 0), m);

    let (f_m, f_head) = b.segment(fused, (cp, pv, cq, qv));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // `None` = that source is not poked; an all-NaN row is poked but blocked.
    type Tick = (Option<[f64; 2]>, Option<[f64; 2]>);
    let ticks: &[Tick] = &[
        (Some([1.0, 2.0]), Some([10.0, 20.0])), // both pass
        (Some([nan, nan]), Some([11.0, 21.0])), // p blocked by the gate
        (Some([3.0, 4.0]), Some([nan, nan])),   // q blocked by the gate
        (Some([nan, nan]), Some([nan, nan])),   // both blocked
        (None, None),                           // idle generation
        (Some([5.0, 6.0]), None),               // only p fires
        (None, Some([12.0, 22.0])),             // only q fires
        (Some([7.0, nan]), Some([13.0, 23.0])), // partial NaN still passes
        (Some([9.0, 8.0]), Some([14.0, 24.0])),
    ];
    let mut seen = Vec::new();
    for (i, (p, q)) in ticks.iter().enumerate() {
        if let Some(p) = p {
            *g.state_mut(sp) = p.to_vec().into();
        }
        if let Some(q) = q {
            *g.state_mut(sq) = q.to_vec().into();
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_same_bits(g.view(m), g.view(f_m), &format!("tick {i}: mean"));
        assert_same_bits(g.view(head), g.view(f_head), &format!("tick {i}: head"));
        seen.push(vals(g.view(m)));
    }
    // Guard against a vacuous pass (see the note in the chain test above).
    assert!(
        seen.iter().flatten().any(|v| v.is_finite()) && seen.iter().any(|r| *r != seen[0]),
        "the tick table must drive varying, finite output: {seen:?}"
    );
}

// ---------------------------------------------------------------------------
// Composed signals
// ---------------------------------------------------------------------------

/// The design-brief signal, fused into one node, matches an independent
/// reference model tick for tick — and fires on the *edge*, not the level.
///
/// This is the end-to-end statement the whole framework exists for: two
/// self-recording means, a subtraction, a lag, two comparisons and two boolean
/// connectives collapsed into a single node, run over 60 ticks of a two-element
/// cross-section so the operators' private records compact many times
/// underneath their windows. The reference model is written from the formula
/// and shares no code with the graph. The final assertion guards against the
/// whole comparison degenerating into "two all-false vectors agree".
#[test]
fn ma_crossover_signal_fires_on_the_edge() {
    let (fast, slow) = (2usize, 3usize);

    // A V-shaped prefix (one clean crossing) followed by a pseudo-random tail
    // that keeps both elements crossing back and forth.
    let mut path0 = vec![10.0, 9.0, 8.0, 7.0, 12.0, 14.0, 16.0];
    path0.extend(quarter_path(42, 53));
    let path1: Vec<f64> = path0.iter().map(|&x| 200.25 - x).collect();

    let mut b = Builder::new();
    let (src, (xc, xv)) = b.source(cell([0.0_f64, 0.0]));
    let signal = b.segment(
        segment!(
            |c: SignalPort<0>, x: ArrayPort<f64, 1>| -> ArrayPort<bool, 1> {
                let zeros = array::constant([0.0_f64, 0.0]) @ ();
                let d = elem::sub()
                    @ (rolling::mean(fast, 1) @ (c, x), rolling::mean(slow, 1) @ (c, x));
                let prev = rolling::lag(1) @ (c, d);
                elem::and() @ (
                    elem::gt() @ (d, zeros),
                    elem::not() @ (elem::gt() @ (prev, zeros)),
                )
            }
        ),
        (xc, xv),
    );
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let expect0 = crossover_reference(&path0, fast, slow);
    let expect1 = crossover_reference(&path1, fast, slow);
    let mut fired = 0usize;
    for t in 0..path0.len() {
        *g.state_mut(src) = [path0[t], path1[t]].into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        assert_eq!(
            &*g.view(signal).to_contiguous(),
            &[expect0[t], expect1[t]],
            "tick {t}"
        );
        fired += usize::from(expect0[t]) + usize::from(expect1[t]);
    }
    assert!(
        expect0[4] && fired > 5,
        "path must exercise real crossovers (fired {fired} times)"
    );
}

/// A self-recording composed chain (`rolling::mean(w, c) @ (signal, values)`)
/// agrees bit-for-bit with the hoisted spelling (one shared `series::record_on`
/// feeding `rolling::series_mean(w, c)`).
///
/// The two spellings differ in more than syntax: the self-recording form gives
/// every operator its own private buffer with delayed trimming sized to its own
/// window, while the hoisted form shares one eagerly trimmed record between both
/// windows. Over 40 quarter-valued ticks (windows 4 and 9 against a shared
/// retention of 32, so the fronts compact repeatedly and at different times) the
/// incremental accumulators must still land on identical bits.
#[test]
fn self_recording_chain_matches_hoisted_record() {
    let (fast, slow) = (4usize, 9usize);

    let mut b = Builder::new();
    let (src, (xc, xv)) = b.source(cell([0.0_f64, 0.0]));

    // Self-recording: each `rolling::mean` buffers the live stream itself.
    let live = b.segment(
        segment!(
            |c: SignalPort<0>, x: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
                elem::sub()
                    @ (rolling::mean(fast, 1) @ (c, x), rolling::mean(slow, 1) @ (c, x))
            }
        ),
        (xc, xv),
    );

    // Hoisted: one shared record, two window operators reading it.
    let rec = b.segment(series::record_on(32, false), (xc, xv));
    let h_fast = b.segment(rolling::series_mean(fast, 1), rec);
    let h_slow = b.segment(rolling::series_mean(slow, 1), rec);
    let hoisted = b.segment(elem::sub(), (h_fast, h_slow));

    let mut g = b.build();
    let mut pool = Pool::new(0);

    let (p0, p1) = (quarter_path(1, 40), quarter_path(2, 40));
    for t in 0..40 {
        *g.state_mut(src) = [p0[t], p1[t]].into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        assert_same_bits(g.view(live), g.view(hoisted), &format!("tick {t}"));
    }
    // Guard against a vacuous pass: both spellings emitting `NaN` (or zero)
    // throughout would agree bit-for-bit while proving nothing.
    assert!(
        vals(g.view(live)).iter().all(|v| v.is_finite())
            && vals(g.view(live)).iter().any(|&v| v != 0.0),
        "the spread must be finite and non-degenerate: {:?}",
        vals(g.view(live))
    );
}

// ---------------------------------------------------------------------------
// Graph structure
// ---------------------------------------------------------------------------

/// A rejoining cone `(a + b) * a` — one source read by two operators that meet
/// again downstream — computes the right value, and everything below the rejoin
/// advances exactly one generation per stabilize.
///
/// The diamond is the shape a naive "recompute every consumer of a dirty port"
/// scheduler gets wrong: `a` reaches `mul` both directly and through `add`, so
/// `mul` must be ordered after `add` (otherwise it multiplies by a half-updated
/// sum, which the value column catches) and the cone below it must not be
/// visited twice for the two paths (which the counter column catches). The table
/// also stages generations in which only one leg is poked, so the un-poked leg's
/// stale value has to be carried into the join, and an idle generation, in which
/// nothing below the rejoin may advance at all.
#[test]
fn rejoining_cone_recomputes_once_per_generation() {
    let mut b = Builder::new();
    let (sa, av) = b.source(array::constant(0.0_f64));
    let (sb, bv) = b.source(array::constant(0.0_f64));
    let sum = b.segment(elem::add(), (av, bv));
    let out = b.segment(elem::mul(), (sum, av));
    let runs = b.segment(runs::<0>(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (poke a, poke b, expected (a + b) * a, expected recomputes below the join)
    let ticks: &[(Option<f64>, Option<f64>, f64, usize)] = &[
        (Some(2.0), Some(3.0), 10.0, 1), // (2+3)*2
        (Some(4.0), None, 28.0, 2),      // (4+3)*4, b carried
        (None, Some(10.0), 56.0, 3),     // (4+10)*4, a carried
        (Some(-1.0), Some(1.0), 0.0, 4), // both poked, one generation
        (None, None, 0.0, 4),            // idle: nothing reruns
        (Some(0.5), Some(0.5), 0.5, 5),
        (Some(-2.0), Some(-3.0), 10.0, 6),
    ];
    for (i, &(a, bb, want, want_runs)) in ticks.iter().enumerate() {
        if let Some(a) = a {
            *g.state_mut(sa) = a.into();
        }
        if let Some(bb) = bb {
            *g.state_mut(sb) = bb.into();
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_close(&vals(g.view(out)), &[want], &format!("tick {i}"));
        assert_eq!(
            g.view(runs),
            want_runs,
            "tick {i}: recomputes below the join"
        );
    }
}

/// Two sources poked in the same generation stabilize **once** over the union of
/// their cones: the shared join recomputes a single time, not once per source.
///
/// Coalescing is what makes a synchronous multi-source tick cheap, and getting
/// it wrong is invisible in the output values — the sum is the same either way —
/// so the only way to pin it is with recompute counters on each leg and on the
/// join. The later generations poke one source alone, proving the counters track
/// scheduling rather than just counting generations.
#[test]
fn coalesced_sources_stabilize_once_over_the_union() {
    let mut b = Builder::new();
    let (sa, (sig_a, av)) = b.source(cell(0.0_f64));
    let (sb, (sig_b, bv)) = b.source(cell(0.0_f64));
    let runs_a = b.segment(runs::<0>(), av);
    let runs_b = b.segment(runs::<0>(), bv);
    let sum = b.segment(elem::add(), (av, bv));
    let runs_sum = b.segment(runs::<0>(), sum);
    // The join's cadence is the union of its inputs' — spelled as the `or` of
    // the two source signals rather than inferred from the scheduler.
    let sum_signal = b.segment(signal::or(), (sig_a, sig_b));
    let rec = b.segment(series::record_all(), (sum_signal, sum));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // (poke a, poke b, expected (runs_a, runs_b, runs_sum))
    type CoalesceTick = (Option<f64>, Option<f64>, (usize, usize, usize));
    let ticks: &[CoalesceTick] = &[
        (Some(10.0), Some(100.0), (1, 1, 1)),
        (Some(20.0), Some(200.0), (2, 2, 2)),
        (Some(30.0), None, (3, 2, 3)),  // only a's cone reruns
        (None, Some(300.0), (3, 3, 4)), // only b's cone reruns
    ];
    for (i, &(a, bb, (wa, wb, ws))) in ticks.iter().enumerate() {
        if let Some(a) = a {
            *g.state_mut(sa) = a.into();
        }
        if let Some(bb) = bb {
            *g.state_mut(sb) = bb.into();
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        assert_eq!(g.view(runs_a), wa, "tick {i}: a leg reruns");
        assert_eq!(g.view(runs_b), wb, "tick {i}: b leg reruns");
        assert_eq!(g.view(runs_sum), ws, "tick {i}: join reruns");
    }
    assert_eq!(
        series_vals(g.view(rec)),
        vec![110.0, 220.0, 230.0, 330.0],
        "one recorded row per generation"
    );
}

// ---------------------------------------------------------------------------
// Parallel execution
// ---------------------------------------------------------------------------

/// Builds and runs a mixed cone — a fused `segment!`, the same chain unfused on
/// a second source, an `event::filter` gate, a lag and a rejoin — on a pool of
/// `workers` threads, returning the per-tick bit patterns of three outputs.
///
/// The two sources are poked on different schedules so the dirty cone changes
/// shape from generation to generation, which is what gives the scheduler
/// something to get wrong.
fn run_mixed_cone(workers: usize, len: usize) -> Vec<Vec<u64>> {
    let (p0, p1) = (quarter_path(11, len), quarter_path(23, len));

    let mut b = Builder::new();
    let (sa, (ca, av)) = b.source(cell([0.0_f64; 4]));
    let (sb, (cb, bv)) = b.source(cell([0.0_f64; 4]));

    let spread = b.segment(
        segment!(
            |c: SignalPort<0>, x: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
                elem::sub()
                    @ (rolling::mean(3, 1) @ (c, x), rolling::mean(7, 1) @ (c, x))
            }
        ),
        (ca, av),
    );
    let gated = b.segment(
        signal::filter(|a: ArrayView<f64, 1>| a.to_contiguous()[0] > 0.0),
        (ca, spread),
    );

    let f2 = b.segment(rolling::mean(3, 1), (cb, bv));
    let s2 = b.segment(rolling::mean(7, 1), (cb, bv));
    let d2 = b.segment(elem::sub(), (f2, s2));

    let cj = b.segment(signal::and(), (gated, cb));
    let join = b.segment(elem::add(), (spread, d2));
    let lagged = b.segment(rolling::lag(2), (cj, join));
    let out = b.segment(elem::mul(), (join, lagged));

    let mut g = b.build();
    let mut pool = Pool::new(workers);
    let mut rows = Vec::with_capacity(len);
    for i in 0..len {
        let x = p0[i];
        // Every 7th tick feeds an all-NaN row; source b idles every 3rd tick.
        let row_a = if i % 7 == 6 {
            vec![f64::NAN; 4]
        } else {
            vec![x, x + 0.25, x - 0.5, x * 0.5]
        };
        *g.state_mut(sa) = row_a.into();
        if i % 3 != 2 {
            let y = p1[i];
            *g.state_mut(sb) = [y, y - 1.0, y + 2.0, y * 0.25].into();
        }
        g.stabilize(&mut pool, &nano(i as i64 + 1));
        let mut row = bits(g.view(spread));
        row.extend(bits(g.view(d2)));
        row.extend(bits(g.view(out)));
        rows.push(row);
    }
    rows
}

/// One source fans out to `branches` independent gate → record / gate → count
/// chains with per-branch thresholds; returns each branch's recorded values and
/// pulse count.
fn run_fanout(workers: usize, branches: usize, path: &[f64]) -> Vec<(Vec<f64>, usize)> {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let probes: Vec<_> = (0..branches)
        .map(|k| {
            let threshold = k as f64 * 15.0;
            let gate = b.segment(
                signal::filter(move |a: ArrayView<f64, 0>| a.to_contiguous()[0] > threshold),
                srcv,
            );
            (
                b.segment(series::record_all(), (gate, srcv.1)),
                b.segment(count::<0>(), (gate, srcv.1)),
            )
        })
        .collect();
    let mut g = b.build();
    let mut pool = Pool::new(workers);

    for (i, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(i as i64 + 1));
    }
    probes
        .iter()
        .map(|&(rec, hits)| (series_vals(g.view(rec)), g.view(hits)))
        .collect()
}

/// `branches` concurrent *stateful* chains (gate → rolling mean → pulse
/// counter), each gating on a different divisor, over `gens` generations.
fn run_stateful_stress(workers: usize, branches: usize, gens: usize) -> Vec<usize> {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let counters: Vec<_> = (0..branches)
        .map(|k| {
            let divisor = k + 2;
            let gate = b.segment(
                signal::filter(move |a: ArrayView<f64, 0>| {
                    (a.to_contiguous()[0] as usize).is_multiple_of(divisor)
                }),
                srcv,
            );
            let _smoothed = b.segment(rolling::mean(3, 1), (gate, srcv.1));
            b.segment(count::<0>(), (gate, srcv.1))
        })
        .collect();
    let mut g = b.build();
    let mut pool = Pool::new(workers);

    for i in 1..=gens {
        *g.state_mut(src) = (i as f64).into();
        g.stabilize(&mut pool, &nano(i as i64));
    }
    counters.iter().map(|&c| g.view(c)).collect()
}

/// The same wiring produces bit-identical output single-threaded and on up to
/// eight worker threads.
///
/// This is the cleanest statement of the parallelism invariant: worker count is
/// a scheduling detail, never a semantic one. The cone deliberately mixes a
/// fused node, an unfused chain, a gate whose pulse decides whether a
/// downstream lag advances, and a rejoin — so a missing dependency edge, or a
/// node run before its producer, shows up as a differing bit pattern rather than
/// as a crash.
#[test]
fn pool_size_does_not_change_results() {
    let reference = run_mixed_cone(0, 60);
    assert_eq!(reference.len(), 60);
    // Guard against a vacuous pass: a cone stuck on one value would be equal
    // under every pool size for the wrong reason.
    assert!(
        reference.iter().any(|r| *r != reference[0]),
        "the cone must actually move between ticks"
    );
    for workers in [1usize, 2, 4, 8] {
        let parallel = run_mixed_cone(workers, 60);
        for (i, (want, got)) in reference.iter().zip(&parallel).enumerate() {
            assert_eq!(want, got, "tick {i}: Pool::new({workers}) diverged");
        }
    }
}

/// A wide fan-out of independent chains agrees with the sequential result, and
/// each branch sees exactly the values its own gate admits.
///
/// Sixteen sibling cones with *different* thresholds run concurrently off one
/// source: if a worker ever scattered a result into the wrong branch's slot, or
/// if two branches shared a gate's state, the per-branch expectations below
/// could not all hold at once.
#[test]
fn parallel_fanout_matches_sequential() {
    const BRANCHES: usize = 16;
    let path = quarter_path(99, 48);

    let sequential = run_fanout(0, BRANCHES, &path);
    let parallel = run_fanout(8, BRANCHES, &path);
    assert_eq!(
        sequential, parallel,
        "parallel run diverged from sequential"
    );

    for (k, (recorded, hits)) in parallel.iter().enumerate() {
        let threshold = k as f64 * 15.0;
        let expected: Vec<f64> = path.iter().copied().filter(|&v| v > threshold).collect();
        assert_eq!(recorded, &expected, "branch {k}: recorded values");
        assert_eq!(*hits, expected.len(), "branch {k}: pulse count");
    }
    // The thresholds must actually bite, or every branch would be the same test.
    assert!(
        parallel[0].0.len() > parallel[BRANCHES - 1].0.len(),
        "thresholds must separate the branches"
    );
}

/// Many concurrent stateful nodes over hundreds of generations keep exactly
/// their own state: no lost, duplicated or cross-branch increments.
///
/// A pulse counter is the sharpest probe for a state race, because a single
/// misplaced run is a permanent off-by-one rather than a value that happens to
/// converge later. Twenty-four branches × 400 generations on eight workers give
/// the scheduler ample opportunity to double-run or skip a node, and the
/// expected counts differ per branch, so a race that copied one branch's state
/// over another would also be caught.
#[test]
fn parallel_stateful_counts_do_not_race() {
    const BRANCHES: usize = 24;
    const GENS: usize = 400;

    let parallel = run_stateful_stress(8, BRANCHES, GENS);
    let sequential = run_stateful_stress(0, BRANCHES, GENS);
    assert_eq!(parallel, sequential, "parallel counts diverged");

    for (k, &got) in parallel.iter().enumerate() {
        let divisor = k + 2;
        let want = (1..=GENS).filter(|i| i % divisor == 0).count();
        assert_eq!(got, want, "branch {k} (divisor {divisor}) raced");
    }
}
