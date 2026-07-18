//! Integration tests for `Series` / `SeriesView`, retention, and iteration.

use tradingflow_data::{Array, ArrayView, Duration, Instant, Retention, Series, SeriesView, Shape};

fn ts(n: i64) -> Instant {
    Instant::from_nanos(n)
}

#[test]
fn series_push_and_access() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    assert!(s.is_empty());

    s.push_data(ts(100), &[1.0, 2.0]);
    s.push_data(ts(200), &[3.0, 4.0]);
    s.push_data(ts(300), &[5.0, 6.0]);

    assert_eq!(s.len(), 3);
    assert_eq!(s.elem_len(), 2);
    assert_eq!(s.timestamps(), &[ts(100), ts(200), ts(300)]);
    assert_eq!(s.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(s.last().unwrap().data(), &[5.0, 6.0]);
    assert_eq!(s.elem(0).data(), &[1.0, 2.0]);
    assert_eq!(s.elem(1).data(), &[3.0, 4.0]);
    assert_eq!(s.elem(1).to_vec(), vec![3.0, 4.0]);
    assert_eq!(s.elem_data(1), &[3.0, 4.0]);
}

#[test]
fn series_scalar() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    assert_eq!(s.elem_len(), 1);

    s.push_data(ts(1), &[10.0]);
    s.push_data(ts(2), &[20.0]);

    assert_eq!(s.len(), 2);
    assert_eq!(s.elem(0).data(), &[10.0]);
    assert_eq!(s.last().unwrap().data(), &[20.0]);
}

#[test]
fn series_asof() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    s.push_data(ts(100), &[1.0]);
    s.push_data(ts(200), &[2.0]);
    s.push_data(ts(300), &[3.0]);

    assert_eq!(s.asof(ts(50)).map(|v| v.data()), None);
    assert_eq!(s.asof(ts(100)).map(|v| v.data()), Some([1.0].as_slice()));
    assert_eq!(s.asof(ts(150)).map(|v| v.data()), Some([1.0].as_slice()));
    assert_eq!(s.asof(ts(200)).map(|v| v.data()), Some([2.0].as_slice()));
    assert_eq!(s.asof(ts(250)).map(|v| v.data()), Some([2.0].as_slice()));
    assert_eq!(s.asof(ts(300)).map(|v| v.data()), Some([3.0].as_slice()));
    assert_eq!(s.asof(ts(999)).map(|v| v.data()), Some([3.0].as_slice()));
}

#[test]
#[should_panic(expected = "push_data: expected 2 scalars, got 3")]
fn series_push_data_wrong_size() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push_data(ts(1), &[1.0, 2.0, 3.0]);
}

#[test]
fn series_elem_shape() {
    let s = Series::<f64, 2>::new_unbounded([3, 4]);
    assert_eq!(s.elem_extents(), [3, 4]);
    assert_eq!(s.elem_len(), 12);
    assert!(s.elem_shape().is_contiguous());
}

#[test]
fn last_timestamp() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    assert_eq!(s.last_timestamp(), None);

    s.push_data(ts(100), &[1.0]);
    assert_eq!(s.last_timestamp(), Some(ts(100)));

    s.push_data(ts(200), &[2.0]);
    assert_eq!(s.last_timestamp(), Some(ts(200)));
}

#[test]
fn window_data_over_logical_range() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push_data(ts(100), &[1.0, 2.0]);
    s.push_data(ts(200), &[3.0, 4.0]);
    s.push_data(ts(300), &[5.0, 6.0]);

    assert_eq!(s.window(0..2).data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(s.window(1..3).data(), &[3.0, 4.0, 5.0, 6.0]);
    assert_eq!(s.window(2..3).data(), &[5.0, 6.0]);
    assert_eq!(s.window(0..0).data(), &[] as &[f64]);
}

#[test]
fn push_matches_push_data() {
    let mut a = Series::<f64, 1>::new_unbounded([2]);
    let mut b = Series::<f64, 1>::new_unbounded([2]);
    let row = Array::from_vec([2], vec![1.0, 2.0]);
    a.push_data(ts(100), row.data());
    b.push(ts(100), row.view());
    assert_eq!(a, b);
}

#[test]
fn push_materializes_a_strided_view() {
    // A strided element must land packed row-major in the series.
    let panel = Array::from_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let col1 = ArrayView::from_parts(Shape::strided([2], [3]), &panel.data()[1..]);
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push(ts(100), col1);
    assert_eq!(s.data(), &[2.0, 5.0]);
}

#[test]
#[should_panic(expected = "push: element extents mismatch")]
fn push_wrong_extents() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    let row = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
    s.push(ts(1), row.view());
}

#[test]
fn search() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    s.push_data(ts(100), &[1.0]);
    s.push_data(ts(200), &[2.0]);
    s.push_data(ts(300), &[3.0]);

    assert_eq!(s.search(ts(50)), 0); // before all
    assert_eq!(s.search(ts(100)), 0); // exact first
    assert_eq!(s.search(ts(150)), 1); // between
    assert_eq!(s.search(ts(200)), 1); // exact second
    assert_eq!(s.search(ts(300)), 2); // exact last
    assert_eq!(s.search(ts(999)), 3); // after all
}

// -- SeriesView --------------------------------------------------------------

#[test]
fn view_window_tail_and_elements() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push_data(ts(100), &[1.0, 2.0]);
    s.push_data(ts(200), &[3.0, 4.0]);
    s.push_data(ts(300), &[5.0, 6.0]);

    let v = s.view();
    assert_eq!(v.len(), 3);
    assert_eq!(v.elem_len(), 2);
    assert_eq!(v.timestamps(), &[ts(100), ts(200), ts(300)]);
    assert_eq!(v.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(v.elem(1).data(), &[3.0, 4.0]);
    assert_eq!(v.elem(2).to_vec(), vec![5.0, 6.0]);
    assert_eq!(v.elem_data(1), &[3.0, 4.0]);
    assert_eq!(v.timestamp(0), ts(100));
    assert_eq!(v.last().unwrap().data(), &[5.0, 6.0]);
    assert_eq!(v.last_timestamp(), Some(ts(300)));

    let w = v.window(1..3);
    assert_eq!(w.len(), 2);
    assert_eq!(w.timestamps(), &[ts(200), ts(300)]);
    assert_eq!(w.data(), &[3.0, 4.0, 5.0, 6.0]);

    let t = s.tail(2);
    assert_eq!(t.timestamps(), &[ts(200), ts(300)]);
    assert_eq!(t.data(), &[3.0, 4.0, 5.0, 6.0]);
    // n > len returns all
    assert_eq!(s.tail(100).len(), 3);

    // Series::window takes logical indices.
    let sw = s.window(1..2);
    assert_eq!(sw.timestamps(), &[ts(200)]);
    assert_eq!(sw.data(), &[3.0, 4.0]);
}

#[test]
fn view_asof_and_search() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    s.push_data(ts(100), &[1.0]);
    s.push_data(ts(200), &[2.0]);
    let v = s.view();
    assert_eq!(v.asof(ts(50)).map(|v| v.data()), None);
    assert_eq!(v.asof(ts(150)).map(|v| v.data()), Some([1.0].as_slice()));
    assert_eq!(v.search(ts(150)), 1);
    assert_eq!(v.search(ts(999)), 2);
}

#[test]
fn view_as_array_view() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push_data(ts(100), &[1.0, 2.0]);
    s.push_data(ts(200), &[3.0, 4.0]);
    s.push_data(ts(300), &[5.0, 6.0]);

    // Whole window: [3, 2], contiguous.
    let av = s.view().as_array_view::<2>();
    assert_eq!(av.extents(), [3, 2]);
    assert!(av.as_slice().is_some());
    assert_eq!(av.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Sub-window converts too.
    let av = s.window(1..3).as_array_view::<2>();
    assert_eq!(av.extents(), [2, 2]);
    assert_eq!(av.to_vec(), vec![3.0, 4.0, 5.0, 6.0]);

    // Owned copy.
    let arr = s.tail(1).to_array::<2>();
    assert_eq!(arr.extents(), [1, 2]);
    assert_eq!(arr.data(), &[5.0, 6.0]);
}

#[test]
#[should_panic(expected = "M (3) must be N + 1 (2)")]
fn view_as_array_view_wrong_rank() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push_data(ts(100), &[1.0, 2.0]);
    let _ = s.view().as_array_view::<3>();
}

#[test]
fn view_from_parts_checks_len() {
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 3.0, 4.0];
    let v = SeriesView::<f64, 1>::from_parts(Shape::row_major([2]), &tss, &vals);
    assert_eq!(v.len(), 2);
    assert_eq!(v.elem(1).data(), &[3.0, 4.0]);
}

#[test]
#[should_panic(expected = "from_parts: 2 elements of 2 scalars expect 4 scalars, got 3")]
fn view_from_parts_wrong_len() {
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 3.0];
    let _ = SeriesView::<f64, 1>::from_parts(Shape::row_major([2]), &tss, &vals);
}

#[test]
fn view_to_series() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push_data(ts(100), &[1.0, 2.0]);
    s.push_data(ts(200), &[3.0, 4.0]);
    s.push_data(ts(300), &[5.0, 6.0]);

    // Whole-window copy of an unbounded series equals the original.
    let owned = s.view().to_series();
    assert_eq!(owned, s);

    // A sub-window copies just its elements into a fresh, same-rank series.
    let sub = s.window(1..3).to_series();
    assert_eq!(sub.elem_extents(), [2]);
    assert_eq!(sub.timestamps(), &[ts(200), ts(300)]);
    assert_eq!(sub.data(), &[3.0, 4.0, 5.0, 6.0]);
}

// -- Retention ---------------------------------------------------------------

#[test]
fn count_retention_bounds_storage_and_preserves_logical_reads() {
    // Keep the most recent 3 elements; push 10.
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10 {
        s.push_data(ts((i + 1) * 100), &[i as f64]);
    }
    // Logical length is the full count; physical storage is bounded (<= 2x
    // the window thanks to amortized compaction).
    assert_eq!(s.len(), 10);
    assert!(
        s.retained_len() <= 6,
        "retained {} > 2x window",
        s.retained_len()
    );
    assert!(s.retained_len() >= 3, "fewer than the window retained");
    assert_eq!(s.base(), s.len() - s.retained_len());

    // The required window [7, 10) reads identically to an unbounded series.
    assert_eq!(s.elem(7).data(), &[7.0]);
    assert_eq!(s.elem(8).data(), &[8.0]);
    assert_eq!(s.elem(9).data(), &[9.0]);
    assert_eq!(s.last().unwrap().data(), &[9.0]);
    assert_eq!(s.timestamp(9), ts(1000));
    assert_eq!(s.window(7..10).data(), &[7.0, 8.0, 9.0]);
}

#[test]
#[should_panic(expected = "out of retained range")]
fn count_retention_evicts_old_indices() {
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10 {
        s.push_data(ts((i + 1) * 100), &[i as f64]);
    }
    // Index 0 was dropped long ago.
    let _ = s.elem(0).data();
}

#[test]
fn duration_retention_keeps_time_window() {
    // Keep everything within 250ns of the latest; ticks are 100ns apart.
    let mut s = Series::<f64, 1>::new([1], Retention::duration(Duration::from_nanos(250)));
    for i in 0..10 {
        s.push_data(ts((i + 1) * 100), &[i as f64]);
    }
    // Latest ts = 1000; cutoff = 750 → keep ts in {800, 900, 1000} = indices 7,8,9.
    assert_eq!(s.len(), 10);
    assert_eq!(s.elem(9).data(), &[9.0]);
    assert_eq!(s.elem(8).data(), &[8.0]);
    assert_eq!(s.elem(7).data(), &[7.0]);
    assert!(s.base() <= 7, "kept window too small: base {}", s.base());
}

#[test]
fn asof_and_search_use_logical_indices_under_retention() {
    // Regression: `asof` once mixed a physical partition point into a
    // logical accessor, which broke as soon as retention evicted elements.
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10 {
        s.push_data(ts((i + 1) * 100), &[i as f64]);
    }
    assert!(s.base() > 0, "retention must have evicted something");

    assert_eq!(s.asof(ts(1000)).unwrap().data(), &[9.0]);
    assert_eq!(s.asof(ts(850)).unwrap().data(), &[7.0]);
    // Before the retained window: None, though older elements once matched.
    assert_eq!(s.asof(ts(100)).map(|v| v.data()), None);

    // `search` returns logical indices: the first ts >= 850 is t900,
    // logically element 8.
    assert_eq!(s.search(ts(850)), 8);
    assert_eq!(s.search(ts(9999)), s.len());
}

#[test]
fn bounded_matches_unbounded_within_window() {
    // A bounded series reads identically to an unbounded one for every index
    // that the bound retains — the equivalence the retention contract rests on.
    let window = 5usize;
    let mut bounded = Series::<f64, 1>::new([2], Retention::count(window));
    let mut unbounded = Series::<f64, 1>::new_unbounded([2]);
    for i in 0..40usize {
        let row = [i as f64, (i * 2) as f64];
        let t = ts((i as i64 + 1) * 10);
        bounded.push_data(t, &row);
        unbounded.push_data(t, &row);
        assert_eq!(bounded.len(), unbounded.len());
        let lo = bounded.base();
        for j in lo..bounded.len() {
            assert_eq!(bounded.elem(j).data(), unbounded.elem(j).data());
            assert_eq!(bounded.timestamp(j), unbounded.timestamp(j));
        }
        assert_eq!(
            bounded.last().unwrap().data(),
            unbounded.last().unwrap().data()
        );
    }
}

// -- Iteration ---------------------------------------------------------------

#[test]
fn series_iter_yields_timestamp_element_pairs() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push_data(ts(100), &[1.0, 2.0]);
    s.push_data(ts(200), &[3.0, 4.0]);
    s.push_data(ts(300), &[5.0, 6.0]);

    assert_eq!(s.iter().len(), 3);
    let collected: Vec<(i64, Vec<f64>)> =
        s.iter().map(|(t, v)| (t.as_nanos(), v.to_vec())).collect();
    assert_eq!(
        collected,
        vec![
            (100, vec![1.0, 2.0]),
            (200, vec![3.0, 4.0]),
            (300, vec![5.0, 6.0]),
        ]
    );

    // `for (t, v) in &s` — the borrowing `IntoIterator`.
    let mut n = 0;
    for (t, v) in &s {
        assert_eq!(v.len(), 2);
        let _ = t;
        n += 1;
    }
    assert_eq!(n, 3);
}

#[test]
fn series_view_into_iter_and_reversed() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    s.push_data(ts(1), &[10.0]);
    s.push_data(ts(2), &[20.0]);
    s.push_data(ts(3), &[30.0]);

    // `IntoIterator` on the `Copy` view (the "owned" entry point).
    let fwd: Vec<f64> = s.view().into_iter().map(|(_, v)| v.to_vec()[0]).collect();
    assert_eq!(fwd, vec![10.0, 20.0, 30.0]);

    // Double-ended: reversed by timestamp.
    let rev: Vec<i64> = s.iter().rev().map(|(t, _)| t.as_nanos()).collect();
    assert_eq!(rev, vec![3, 2, 1]);
}

#[test]
fn series_iter_walks_only_the_retained_window() {
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10 {
        s.push_data(ts((i + 1) * 100), &[i as f64]);
    }
    // Iterating covers exactly the retained window [base, len).
    let vals: Vec<f64> = s.iter().map(|(_, v)| v.to_vec()[0]).collect();
    assert_eq!(vals.len(), s.retained_len());
    assert_eq!(s.iter().count(), s.timestamps().len());
    // The most recent element (logical index 9) is always retained.
    assert_eq!(vals.last().copied(), Some(9.0));
    // Timestamps line up with the retained window.
    let iter_ts: Vec<i64> = s.iter().map(|(t, _)| t.as_nanos()).collect();
    let want_ts: Vec<i64> = s.timestamps().iter().map(|t| t.as_nanos()).collect();
    assert_eq!(iter_ts, want_ts);
}

#[test]
fn series_iter_empty() {
    let s = Series::<f64, 1>::new_unbounded([2]);
    assert_eq!(s.iter().count(), 0);
    assert!(s.iter().next().is_none());
    assert!(s.view().into_iter().next_back().is_none());
}

#[test]
fn series_into_iter_yields_owned_arrays() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push_data(ts(100), &[1.0, 2.0]);
    s.push_data(ts(200), &[3.0, 4.0]);
    s.push_data(ts(300), &[5.0, 6.0]);

    assert_eq!(s.clone().into_iter().len(), 3);
    // By value: each element is an owned `Array` moved out of the series.
    let collected: Vec<(i64, Vec<f64>)> = s
        .into_iter()
        .map(|(t, a)| (t.as_nanos(), a.data().to_vec()))
        .collect();
    assert_eq!(
        collected,
        vec![
            (100, vec![1.0, 2.0]),
            (200, vec![3.0, 4.0]),
            (300, vec![5.0, 6.0]),
        ]
    );
}

#[test]
fn series_into_iter_double_ended_over_retained_window() {
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10 {
        s.push_data(ts((i + 1) * 100), &[i as f64]);
    }
    let retained = s.retained_len();
    // Reversed owned iteration walks exactly the retained window.
    let rev: Vec<f64> = s.into_iter().rev().map(|(_, a)| a.data()[0]).collect();
    assert_eq!(rev.len(), retained);
    assert_eq!(rev.first().copied(), Some(9.0)); // most recent first
}

#[test]
fn series_into_iter_rank2_and_empty() {
    // Rank-2 elements round-trip their row-major scalars.
    let mut s = Series::<f64, 2>::new_unbounded([2, 2]);
    s.push_data(ts(1), &[1.0, 2.0, 3.0, 4.0]);
    let (t, a) = s.into_iter().next().unwrap();
    assert_eq!(t, ts(1));
    assert_eq!(a.extents(), [2, 2]);
    assert_eq!(a.data(), &[1.0, 2.0, 3.0, 4.0]);

    let empty = Series::<f64, 1>::new_unbounded([2]);
    assert_eq!(empty.into_iter().count(), 0);
}
