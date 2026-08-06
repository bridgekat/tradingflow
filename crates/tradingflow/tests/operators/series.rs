//! Integration tests for the `series` operator module — the recorded-history
//! currency of the graph.
//!
//! A `Series<T, N>` is a growable ring of `(Instant, Array<T, N>)` rows that
//! operators read as a `SeriesView<T, N>`. Rows are appended by `record_on` at
//! the graph's event time, dropped from the front by a `Retention` bound, and read
//! back by `last` / `shift` / the view derivations. Every row keeps its
//! *logical* index across front compaction (`range()` is `base..base + len`),
//! which is what lets a downstream reader address a window whose front is being
//! dropped underneath it — a good half of the assertions here are about exactly
//! that.

use tradingflow::data::{Array, ArrayView, Duration, Instant, NewAxis, Series, SeriesView};
use tradingflow::graph::Pool;
use tradingflow::graph::typed::{Builder, Graph};
use tradingflow::operators::{series, signal};
use tradingflow::ports::SeriesPortHandle;

use crate::harness::*;

// ---------------------------------------------------------------------------
// record.rs — appending an array stream into a series
// ---------------------------------------------------------------------------

/// A row is appended per signal, stamped with the `Instant` handed to
/// `stabilize` (not with a tick counter), oldest row first.
#[test]
fn record_stamps_rows_with_the_event_time() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_all(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Before any tick: no rows, but the element shape is already known.
    let s = g.view(rec);
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
    assert_eq!(s.extents(), [0usize; 0]);
    assert_eq!(s.range(), 0..0);

    // Event times need not be consecutive: the stamp is whatever `stabilize`
    // is given.
    for (t, v) in [(1_i64, 10.0), (5, 20.0), (9, 30.0)] {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t));
    }

    let s = g.view(rec);
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &[nano(1), nano(5), nano(9)]);
    assert_eq!(series_vals(s), vec![10.0, 20.0, 30.0]);
    assert_eq!(s.range(), 0..3);
    assert_eq!(s.at(1).0, nano(5));
    assert_eq!(vals(s.at(1).1), vec![20.0]);
}

/// A generation without a signal appends nothing: the record re-lends
/// the series unchanged.
#[test]
fn record_appends_nothing_when_the_input_does_not_notify() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let gate = b.op(signal::filter(|a: ArrayView<'_, f64, 0>| *a > 3.0), srcv);
    let rec = b.op(series::record_all(), (gate, srcv.1));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 1.0), (2, 5.0), (3, 2.0), (4, 10.0)] {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t));
    }

    let s = g.view(rec);
    assert_eq!(series_vals(s), vec![5.0, 10.0]);
    assert_eq!(s.instants(), &[nano(2), nano(4)]);
    // Logical indices count appended rows, not generations.
    assert_eq!(s.range(), 0..2);
}

/// `record_all` is unbounded: nothing is ever trimmed, so the logical base
/// stays at 0 and the first row survives arbitrarily many ticks.
#[test]
fn record_all_keeps_every_row() {
    const TICKS: i64 = 40;

    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_all(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 1..=TICKS {
        *g.state_mut(src) = (t as f64).into();
        g.stabilize(&mut pool, &nano(t));
    }

    let s = g.view(rec);
    assert_eq!(s.len(), TICKS as usize);
    assert_eq!(s.range(), 0..TICKS as usize);
    assert_eq!(s.at(0).0, nano(1));
    assert_eq!(vals(s.at(0).1), vec![1.0]);
    assert_eq!(
        series_vals(s),
        (1..=TICKS).map(|t| t as f64).collect::<Vec<_>>()
    );
}

/// Rows keep the element extents of the input, packed row-major and appended
/// one whole element at a time.
#[test]
fn record_keeps_the_element_extents_of_its_input() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, vec![0.0_f64; 2].into());
    let rec = b.op(series::record_all(), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 1..=3_i64 {
        *g.state_mut(src) = [t as f64, -(t as f64)].into();
        g.stabilize(&mut pool, &nano(t));
    }

    let s = g.view(rec);
    assert_eq!(s.extents(), [2]);
    assert_eq!(s.len(), 3);
    assert_eq!(
        series_vals(s),
        vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
        "rows are packed row-major, oldest first"
    );
    assert_eq!(vals(s.at(2).1), vec![3.0, -3.0]);
}

/// `delayed` moves trimming from *after* the push to *before the next* push,
/// so the rows a bounded record is about to drop stay readable for one more
/// tick. `buffer(r)` is exactly `record_on(r, true)`.
#[test]
fn record_delayed_defers_trimming_by_one_tick() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let eager = b.op(series::record_on(2usize, false), srcv);
    let delayed = b.op(series::record_on(2usize, true), srcv);
    let buffered = b.op(series::buffer(2usize), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Trimming is amortized (it only fires once at least half the rows are
    // droppable), so a `count(2)` record grows to 4 rows before compacting
    // back to 2 — on the tick that overflows, or one tick later if `delayed`.
    let want_eager: [&[i64]; 6] = [&[1], &[1, 2], &[1, 2, 3], &[3, 4], &[3, 4, 5], &[5, 6]];
    let want_delayed: [&[i64]; 6] = [
        &[1],
        &[1, 2],
        &[1, 2, 3],
        &[1, 2, 3, 4],
        &[3, 4, 5],
        &[3, 4, 5, 6],
    ];

    for t in 1..=6_i64 {
        *g.state_mut(src) = (10.0 * t as f64).into();
        g.stabilize(&mut pool, &nano(t));

        let (e, d, f) = (g.view(eager), g.view(delayed), g.view(buffered));
        let want = |rows: &[i64]| rows.iter().map(|&n| nano(n)).collect::<Vec<_>>();
        assert_eq!(e.instants(), want(want_eager[t as usize - 1]), "eager @{t}");
        assert_eq!(
            d.instants(),
            want(want_delayed[t as usize - 1]),
            "delayed @{t}"
        );
        // `buffer(r)` is a shorthand for `record_on(r, true)`, row for row.
        assert_eq!(f.instants(), d.instants(), "buffer @{t}");
        assert_eq!(series_vals(f), series_vals(d), "buffer @{t}");
        // Whichever way it trims, the newest row is always the current tick.
        assert_eq!(e.at(e.range().end - 1).0, nano(t), "eager @{t}");
        assert_eq!(d.at(d.range().end - 1).0, nano(t), "delayed @{t}");
    }

    // On tick 6 the eager record has already dropped rows 3 and 4; the delayed
    // one still lends them, and its base lags by one compaction step.
    let (e, d) = (g.view(eager), g.view(delayed));
    assert_eq!(series_vals(e), vec![50.0, 60.0]);
    assert_eq!(series_vals(d), vec![30.0, 40.0, 50.0, 60.0]);
    assert_eq!(e.range(), 4..6);
    assert_eq!(d.range(), 2..6);
}

/// A count-bounded record drops old rows while keeping window-relative
/// addressing exact — logical index `i` is still row `i` after the front is
/// compacted — honours its retention floor, and keeps storage bounded.
#[test]
fn bounded_record_compacts_the_front_and_bounds_storage() {
    const RETAIN: usize = 8;
    const TICKS: i64 = 30;

    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_on(RETAIN, false), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 1..=TICKS {
        *g.state_mut(src) = (t as f64).into();
        g.stabilize(&mut pool, &nano(t));

        let s = g.view(rec);
        // Logical indices count every row ever pushed, trimmed or not.
        assert_eq!(s.range().end, t as usize, "tick {t}: logical end");
        // At least the retention window is retained ...
        assert!(
            s.len() >= (t as usize).min(RETAIN),
            "tick {t}: len {} is below the retention floor",
            s.len()
        );
        // ... and never unboundedly more: trimming is amortized, so the
        // physical buffer may reach twice the window but no more.
        assert!(
            s.len() <= 2 * RETAIN,
            "tick {t}: physical storage unbounded: {}",
            s.len()
        );
        // Window-relative addressing: logical index `i` is still row `i`.
        for i in s.range() {
            let (at, v) = s.at(i);
            assert_eq!(at, nano(i as i64 + 1), "tick {t}: instant at {i}");
            assert_eq!(vals(v), vec![i as f64 + 1.0], "tick {t}: value at {i}");
        }
    }

    let s = g.view(rec);
    assert!(s.range().start > 0, "expected front compaction");
    assert!((s.len() as i64) < TICKS, "expected front compaction");
    assert_eq!(
        s.at(s.range().end - 1).0,
        nano(TICKS),
        "latest stamp intact"
    );
    assert_eq!(
        vals(s.at(s.range().end - 1).1),
        vec![TICKS as f64],
        "latest value intact"
    );
}

/// A duration-bounded record retains every row stamped within the trailing
/// window of the newest row, and stays bounded across day-stamped ticks.
#[test]
fn duration_bounded_record_keeps_the_trailing_time_window() {
    const WINDOW: i64 = 3;
    const DAYS: i64 = 12;

    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_on(Duration::from_days(WINDOW), false), srcv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for n in 1..=DAYS {
        *g.state_mut(src) = (n as f64).into();
        g.stabilize(&mut pool, &day(n));

        let s = g.view(rec);
        assert_eq!(s.range().end, n as usize, "day {n}: logical end");
        // The cutoff is exclusive: rows stamped at or before `now - WINDOW`
        // may be dropped, everything newer must still be there.
        let kept: Vec<Instant> = ((n - WINDOW + 1).max(1)..=n).map(day).collect();
        assert!(
            s.instants().ends_with(&kept),
            "day {n}: {:?} must end with {kept:?}",
            s.instants()
        );
        assert!(
            s.len() <= 2 * WINDOW as usize,
            "day {n}: physical storage unbounded: {}",
            s.len()
        );
        // Whatever survived is the contiguous tail of the tick sequence.
        let first = s.range().start as i64 + 1;
        assert_eq!(
            series_vals(s),
            (first..=n).map(|k| k as f64).collect::<Vec<_>>(),
            "day {n}"
        );
        assert_eq!(s.at(s.range().end - 1).0, day(n), "day {n}: latest stamp");
    }

    assert!(
        g.view(rec).range().start > 0,
        "expected front compaction over {DAYS} days"
    );
}

// ---------------------------------------------------------------------------
// last.rs — reading the newest row back as an array
// ---------------------------------------------------------------------------

/// `last` and `last_or` differ only in what they emit for an empty series:
/// `last` is `last_or(NaN)`. Both the build-time (`init`) and the tick-time
/// (`compute`) empty paths fill.
#[test]
fn last_and_last_or_differ_only_on_an_empty_series() {
    let mut b = Builder::new();
    let (cell, sv) = b.source(series::constant(Series::new([3])));
    let nan_filled = b.op(series::last(), sv);
    let fill_filled = b.op(series::last_or(-1.0_f64), sv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Build time, before any tick: the fill is shaped like a row.
    assert_close(&vals(g.view(nan_filled)), &[f64::NAN; 3], "last @build");
    assert_eq!(vals(g.view(fill_filled)), vec![-1.0; 3], "last_or @build");

    // Tick time with the series still empty: same answers, other code path.
    let _ = g.state_mut(cell);
    g.stabilize(&mut pool, &nano(1));
    assert_close(&vals(g.view(nan_filled)), &[f64::NAN; 3], "last @empty");
    assert_eq!(vals(g.view(fill_filled)), vec![-1.0; 3], "last_or @empty");

    // Once there is a row to read, the two agree.
    *g.state_mut(cell) = Series::from_parts(
        [3],
        vec![nano(1), nano(2)],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        0,
    );
    g.stabilize(&mut pool, &nano(2));
    assert_eq!(vals(g.view(nan_filled)), vec![4.0, 5.0, 6.0]);
    assert_eq!(vals(g.view(fill_filled)), vec![4.0, 5.0, 6.0]);
}

/// `last_or` tracks the newest recorded row, and holds it across generations
/// in which the record appends nothing.
#[test]
fn last_or_tracks_the_newest_row_and_holds_it_across_idle_ticks() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let gate = b.op(signal::filter(|a: ArrayView<'_, f64, 0>| *a > 3.0), srcv);
    let rec = b.op(series::record_all(), (gate, srcv.1));
    let lst = b.op(series::last_or(0.0_f64), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // The gate drops ticks 1 and 3, so the record appends nothing there and
    // `last_or` must carry its previous output.
    for (t, v, want) in [
        (1_i64, 1.0, 0.0), // dropped: no row yet, so still the empty fill
        (2, 5.0, 5.0),
        (3, 2.0, 5.0), // dropped: carried
        (4, 10.0, 10.0),
    ] {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t));
        assert_eq!(vals(g.view(lst)), vec![want], "tick {t}");
    }
}

/// `last` addresses the newest row by logical index (`range().end - 1`), so it
/// keeps working once the record's base has advanced past 0.
#[test]
fn last_reads_the_newest_row_of_a_compacted_record() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_on(2usize, false), srcv);
    let lst = b.op(series::last(), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 1..=10_i64 {
        *g.state_mut(src) = (t as f64).into();
        g.stabilize(&mut pool, &nano(t));
        assert_eq!(vals(g.view(lst)), vec![t as f64], "tick {t}");
    }
    assert!(g.view(rec).range().start > 0, "expected front compaction");
}

// ---------------------------------------------------------------------------
// shift.rs — re-pairing values with other rows' instants
// ---------------------------------------------------------------------------

/// A positive shift lags the values: output row `j` carries value `j` paired
/// with the instant of row `j + n`, so the newest readable value is the one
/// from `n` rows back. Unpaired rows are dropped, not `NaN`-filled.
#[test]
fn shift_positive_lags_values_behind_their_instants() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_all(), srcv);
    let by1 = b.op(series::shift(1), rec);
    let by2 = b.op(series::shift(2), rec);
    let last2 = b.op(series::last_or(0.0_f64), by2);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0), (4, 40.0)] {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t));
    }

    let s = g.view(by1);
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &[nano(2), nano(3), nano(4)]);
    assert_eq!(series_vals(s), vec![10.0, 20.0, 30.0]);

    let s = g.view(by2);
    assert_eq!(s.len(), 2);
    assert_eq!(s.instants(), &[nano(3), nano(4)]);
    assert_eq!(series_vals(s), vec![10.0, 20.0]);
    assert_eq!(s.range(), 0..2, "the logical base is inherited");
    // Read through a downstream operator: the newest row of `shift(2)` is the
    // value from two rows back.
    assert_eq!(vals(g.view(last2)), vec![20.0]);
}

/// A negative shift leads the values: output row `j` carries value `j + |n|`
/// paired with the instant of row `j`, so the newest readable value is the
/// newest recorded one, stamped `|n|` rows earlier.
#[test]
fn shift_negative_leads_values_ahead_of_their_instants() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_all(), srcv);
    let by1 = b.op(series::shift(-1), rec);
    let by2 = b.op(series::shift(-2), rec);
    let last2 = b.op(series::last_or(0.0_f64), by2);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0), (4, 40.0)] {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t));
    }

    let s = g.view(by1);
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &[nano(1), nano(2), nano(3)]);
    assert_eq!(series_vals(s), vec![20.0, 30.0, 40.0]);

    let s = g.view(by2);
    assert_eq!(s.len(), 2);
    assert_eq!(s.instants(), &[nano(1), nano(2)]);
    assert_eq!(series_vals(s), vec![30.0, 40.0]);
    assert_eq!(s.range(), 0..2);
    assert_eq!(vals(g.view(last2)), vec![40.0]);
}

/// `shift(0)` is the identity, rows and stamps alike.
#[test]
fn shift_by_zero_is_the_identity() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_all(), srcv);
    let by0 = b.op(series::shift(0), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t));
    }

    let (r, s) = (g.view(rec), g.view(by0));
    assert_eq!(s.instants(), r.instants());
    assert_eq!(series_vals(s), series_vals(r));
    assert_eq!(s.range(), r.range());
}

/// Shifting by more rows than the series holds leaves nothing paired: the
/// output is empty in both directions, and a reader falls back to its fill.
#[test]
fn shift_past_the_window_yields_an_empty_series() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.op(series::record_all(), srcv);
    let ahead = b.op(series::shift(9), rec);
    let behind = b.op(series::shift(-9), rec);
    let last_ahead = b.op(series::last_or(-1.0_f64), ahead);
    let last_behind = b.op(series::last_or(-1.0_f64), behind);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, v) in [(1_i64, 10.0), (2, 20.0), (3, 30.0)] {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t));
    }

    for (s, ctx) in [(g.view(ahead), "shift(9)"), (g.view(behind), "shift(-9)")] {
        assert!(s.is_empty(), "{ctx}");
        assert_eq!(s.len(), 0, "{ctx}");
        assert!(s.instants().is_empty(), "{ctx}");
        assert_eq!(series_vals(s), Vec::<f64>::new(), "{ctx}");
    }
    assert_eq!(vals(g.view(last_ahead)), vec![-1.0]);
    assert_eq!(vals(g.view(last_behind)), vec![-1.0]);
}

// ---------------------------------------------------------------------------
// constant.rs — series-valued source cells
// ---------------------------------------------------------------------------

/// A `constant` series lends its rows unchanged, and is a pokeable source
/// cell: replacing its state republishes the new rows on the next generation.
#[test]
fn constant_series_lends_its_rows_and_is_pokeable() {
    let mut b = Builder::new();
    let (cell, sv) = b.source(series::constant(Series::from_parts(
        [2],
        vec![nano(1), nano(2)],
        vec![1.0_f64, 2.0, 3.0, 4.0],
        0,
    )));
    let lst = b.op(series::last_or(0.0_f64), sv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let s = g.view(sv);
    assert_eq!(s.extents(), [2]);
    assert_eq!(s.len(), 2);
    assert_eq!(s.instants(), &[nano(1), nano(2)]);
    assert_eq!(series_vals(s), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(vals(g.view(lst)), vec![3.0, 4.0]);

    *g.state_mut(cell) = Series::from_parts(
        [2],
        vec![nano(1), nano(2), nano(3)],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        0,
    );
    g.stabilize(&mut pool, &nano(3));
    assert_eq!(g.view(sv).len(), 3);
    assert_eq!(vals(g.view(lst)), vec![5.0, 6.0]);
}

/// `from_parts` takes the logical base of the first retained row: `range()` is
/// `base..base + len`, and readers address rows by that logical index.
#[test]
fn from_parts_series_keeps_its_logical_base() {
    let mut b = Builder::new();
    let sv = b.val(series::constant(Series::from_parts(
        [],
        vec![nano(1), nano(2), nano(3)],
        vec![10.0_f64, 20.0, 30.0],
        5,
    )));
    let lst = b.op(series::last_or(0.0_f64), sv);
    let g = b.build();

    let s = g.view(sv);
    assert_eq!(s.len(), 3);
    assert_eq!(s.range(), 5..8, "rows 0..5 are notionally already trimmed");
    assert_eq!(s.at(5).0, nano(1));
    assert_eq!(vals(s.at(5).1), vec![10.0]);
    assert_eq!(vals(s.at(7).1), vec![30.0]);
    // A reader resolves `range().end - 1`, not `len() - 1`.
    assert_eq!(vals(g.view(lst)), vec![30.0]);
}

/// `empty` carries the element extents with no rows at all.
#[test]
fn empty_series_has_no_rows_but_keeps_its_extents() {
    let mut b = Builder::new();
    let sv = b.val(series::constant(Series::<f64, _>::new([2, 3])));
    let g = b.build();

    let s = g.view(sv);
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
    assert_eq!(s.extents(), [2, 3]);
    assert_eq!(s.range(), 0..0);
    assert!(s.instants().is_empty());
    assert_eq!(series_vals(s), Vec::<f64>::new());
}

// ---------------------------------------------------------------------------
// view.rs — zero-copy derivations over the element axes
// ---------------------------------------------------------------------------

/// The `[2, 3]` panel recorded on tick `k`: element `(i, j)` is
/// `100k + 10i + j`, so every scalar names its own row and position.
fn panel(k: i64) -> Array<f64, 2> {
    let data = (0..2)
        .flat_map(|i| (0..3).map(move |j| (100 * k + 10 * i + j) as f64))
        .collect();
    Array::from_parts([2, 3], data)
}

/// The flat row-major scalars of all three recorded panels.
fn all_panel_vals() -> Vec<f64> {
    (1..=3).flat_map(|k| vals(panel(k).view())).collect()
}

/// Records the three panels at `ts(1..=3)` behind whatever `wire` builds on
/// top of the record, and returns the stabilized graph with `wire`'s handles.
fn panels_recorded<H>(
    wire: impl FnOnce(&mut Builder<Instant>, SeriesPortHandle<f64, 2>) -> H,
) -> (Graph<Instant>, H) {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, Array::zeros([2, 3]));
    let rec = b.op(series::record_all(), srcv);
    let out = wire(&mut b, rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for k in 1..=3_i64 {
        *g.state_mut(src) = panel(k);
        g.stabilize(&mut pool, &nano(k));
    }
    (g, out)
}

/// The first `n` panels as one owned series, stamped `ts(1..=n)`.
fn panel_series(n: i64) -> Series<f64, 2> {
    Series::from_parts(
        [2, 3],
        (1..=n).map(nano).collect(),
        (1..=n).flat_map(|k| vals(panel(k).view())).collect(),
        0,
    )
}

/// Wires `wire` on top of a *constant* series that grows from two to three
/// panels — the counterpart to [`panels_recorded`] for derivations that should
/// be exercised against rows already present at build time as well as rows
/// appended later.
fn panels_constant<H>(
    wire: impl FnOnce(&mut Builder<Instant>, SeriesPortHandle<f64, 2>) -> H,
) -> (Graph<Instant>, H) {
    let mut b = Builder::new();
    let (cell, sv) = b.source(series::constant(panel_series(2)));
    let out = wire(&mut b, sv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Append the third panel, so the derivation is read back through its
    // `compute` path and is seen to track a growing series.
    *g.state_mut(cell) = panel_series(3);
    g.stabilize(&mut pool, &nano(3));
    (g, out)
}

/// `slice` reshapes the element axes only: every row survives with its stamp,
/// the row stride still spans a whole panel, and a downstream reader sees the
/// sliced sub-region of the newest row through the strided path.
#[test]
fn slice_selects_element_axes_and_keeps_every_row() {
    let (g, (sliced, lst)) = panels_constant(|b, rows| {
        let sliced = b.op(series::slice((.., 1..3)), rows);
        let lst = b.op(series::last_or(0.0_f64), sliced);
        (sliced, lst)
    });

    let s = g.view(sliced);
    assert_eq!(s.extents(), [2, 2], "columns 1..3 of each row");
    assert_eq!(s.len(), 3, "the row axis is untouched");
    assert_eq!(s.instants(), &[nano(1), nano(2), nano(3)]);
    assert_eq!(s.stride(), 6, "rows still step by a whole panel");
    assert_eq!(
        series_vals(s),
        vec![
            101.0, 102.0, 111.0, 112.0, //
            201.0, 202.0, 211.0, 212.0, //
            301.0, 302.0, 311.0, 312.0,
        ]
    );
    assert_eq!(vals(s.at(1).1), vec![201.0, 202.0, 211.0, 212.0]);
    assert_eq!(vals(g.view(lst)), vec![301.0, 302.0, 311.0, 312.0]);
}

/// Slicing a series with no rows gives an empty sliced series rather than
/// panicking. A zero-row series has an empty scalar buffer while the slice's
/// element-axis offset is non-zero, so the narrowing has to tolerate an offset
/// past the end. This is load-bearing rather than a corner case: the builder
/// evaluates every operator's output once at build time, when a `record` is
/// always still empty — so without it, `series::slice` downstream of a
/// `record` would bring the graph down at `build()`.
#[test]
fn slicing_an_empty_series_is_empty() {
    let mut b = Builder::new();
    let sv = b.val(series::constant(Series::<f64, _>::new([2, 3])));
    let sliced = b.op(series::slice((.., 1..3)), sv);
    let projected = b.op(series::slice_reshape::<_, _, 1, _>((1usize, ..)), sv);
    let g = b.build();

    let s = g.view(sliced);
    assert!(s.is_empty());
    assert_eq!(s.extents(), [2, 2]);
    assert_eq!(series_vals(s), Vec::<f64>::new());
    assert!(g.view(projected).is_empty());
    assert_eq!(g.view(projected).extents(), [3]);
}

/// The same, wired the way it actually comes up: `slice` directly downstream
/// of a `record`, which is empty when the graph is built and fills as ticks
/// arrive.
#[test]
fn slicing_a_record_survives_the_empty_build() {
    let mut b = Builder::new();
    let (src, srcv) = event_src(&mut b, Array::zeros([2, 3]));
    let rec = b.op(series::record_all(), srcv);
    let sliced = b.op(series::slice((.., 1..3)), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    assert!(g.view(sliced).is_empty(), "empty at build");

    *g.state_mut(src) = Array::from_parts([2, 3], (1..=6).map(|i| i as f64).collect());
    g.stabilize(&mut pool, &nano(1));

    let s = g.view(sliced);
    assert_eq!(s.len(), 1);
    assert_eq!(s.extents(), [2, 2]);
    assert_eq!(series_vals(s), vec![2.0, 3.0, 5.0, 6.0]);
}

/// `permute_axes` permutes the element axes only — the row/time axis is not part
/// of the permutation — and a reader materializes the permuted newest row.
#[test]
fn permute_axes_permutes_element_axes_only() {
    let (g, (tr, lst)) = panels_recorded(|b, rec| {
        let tr = b.op(series::permute_axes([1, 0]), rec);
        let lst = b.op(series::last_or(0.0_f64), tr);
        (tr, lst)
    });

    let s = g.view(tr);
    assert_eq!(s.extents(), [3, 2]);
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &[nano(1), nano(2), nano(3)]);
    assert_eq!(s.stride(), 6);
    assert_eq!(
        series_vals(s),
        vec![
            100.0, 110.0, 101.0, 111.0, 102.0, 112.0, //
            200.0, 210.0, 201.0, 211.0, 202.0, 212.0, //
            300.0, 310.0, 301.0, 311.0, 302.0, 312.0,
        ]
    );
    assert_eq!(
        vals(g.view(lst)),
        vec![300.0, 310.0, 301.0, 311.0, 302.0, 312.0]
    );
}

/// `move_axis` likewise touches the element axes only, leaving the row/time
/// axis, the instants and the packing stride untouched.
#[test]
fn move_axis_moves_element_axes_only() {
    let (g, (moved, lst)) = panels_recorded(|b, rec| {
        let moved = b.op(series::move_axis(1, 0), rec);
        let lst = b.op(series::last_or(0.0_f64), moved);
        (moved, lst)
    });

    let s = g.view(moved);
    assert_eq!(s.extents(), [3, 2]);
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &[nano(1), nano(2), nano(3)]);
    assert_eq!(s.stride(), 6);
    // At rank 2 the only move is the transpose `permute_axes([1, 0])` spells.
    assert_eq!(
        series_vals(s),
        vec![
            100.0, 110.0, 101.0, 111.0, 102.0, 112.0, //
            200.0, 210.0, 201.0, 211.0, 202.0, 212.0, //
            300.0, 310.0, 301.0, 311.0, 302.0, 312.0,
        ]
    );
    assert_eq!(
        vals(g.view(lst)),
        vec![300.0, 310.0, 301.0, 311.0, 302.0, 312.0]
    );
}

/// `slice_reshape` projects an element axis away, dropping the element rank
/// while keeping every row. (The wrapper takes exactly `N` specifiers for a
/// rank-`N` input, so it can only drop axes; adding them needs `derive_view`
/// — see `derive_view_can_add_element_axes`.)
#[test]
fn slice_reshape_projects_an_element_axis_away() {
    let (g, (row1, lst)) = panels_constant(|b, rows| {
        let row1 = b.op(series::slice_reshape::<_, _, 1, _>((1usize, ..)), rows);
        let lst = b.op(series::last_or(0.0_f64), row1);
        (row1, lst)
    });

    let s = g.view(row1);
    assert_eq!(s.extents(), [3], "axis 0 projected at index 1");
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &[nano(1), nano(2), nano(3)]);
    assert_eq!(
        series_vals(s),
        vec![
            110.0, 111.0, 112.0, 210.0, 211.0, 212.0, 310.0, 311.0, 312.0
        ]
    );
    assert_eq!(vals(g.view(lst)), vec![310.0, 311.0, 312.0]);
}

/// A `derive_view` closure can do what the fixed wrappers cannot: insert new
/// element axes. The scalars and the rows are untouched.
#[test]
fn derive_view_can_add_element_axes() {
    fn spread(a: SeriesView<'_, f64, 2>) -> SeriesView<'_, f64, 4> {
        a.slice_reshape::<4, _>((.., NewAxis, .., NewAxis))
    }

    let (g, wide) = panels_recorded(|b, rec| b.op(series::derive_view(spread), rec));

    let s = g.view(wide);
    assert_eq!(s.extents(), [2, 1, 3, 1]);
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &[nano(1), nano(2), nano(3)]);
    assert_eq!(series_vals(s), all_panel_vals());
}

/// `pad_ndim` prepends unit *element* axes, leaving the row axis and the
/// stamps alone. Padding to the input's own rank is the identity.
#[test]
fn pad_ndim_prepends_unit_element_axes() {
    let (g, (same, padded)) = panels_recorded(|b, rec| {
        let same = b.op(series::pad_ndim::<_, 2, 2>(), rec);
        let padded = b.op(series::pad_ndim::<_, 2, 4>(), rec);
        (same, padded)
    });

    let s = g.view(same);
    assert_eq!(s.extents(), [2, 3]);
    assert_eq!(s.len(), 3);
    assert_eq!(series_vals(s), all_panel_vals());

    let s = g.view(padded);
    assert_eq!(s.extents(), [1, 1, 2, 3]);
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &[nano(1), nano(2), nano(3)]);
    assert_eq!(series_vals(s), all_panel_vals());
}

/// `derive_view` is also the escape hatch for the row axis: a window keeps the
/// logical indices of the rows it selects, so `range()` starts at the logical
/// index of the window's first row rather than at 0.
#[test]
fn derive_view_windows_the_row_axis_preserving_logical_indices() {
    fn last_two(a: SeriesView<'_, f64, 2>) -> SeriesView<'_, f64, 2> {
        let r = a.range();
        a.window(r.end.saturating_sub(2).max(r.start)..r.end)
    }

    let (g, (tail, lst)) = panels_recorded(|b, rec| {
        let tail = b.op(series::derive_view(last_two), rec);
        let lst = b.op(series::last_or(0.0_f64), tail);
        (tail, lst)
    });

    let s = g.view(tail);
    assert_eq!(s.len(), 2);
    assert_eq!(s.range(), 1..3, "logical indices survive windowing");
    assert_eq!(s.instants(), &[nano(2), nano(3)]);
    assert_eq!(s.at(1).0, nano(2));
    assert_eq!(vals(s.at(1).1), vals(panel(2).view()));
    assert_eq!(vals(g.view(lst)), vals(panel(3).view()));
}
