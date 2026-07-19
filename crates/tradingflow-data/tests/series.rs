use tradingflow_data::layout::Strided;
use tradingflow_data::{
    Array, ArrayView, Duration, Instant, Layout, Retention, Series, SeriesView,
};

fn ts(n: i64) -> Instant {
    Instant::from_offset(Duration::from_nanos(n))
}

#[test]
fn series_push_and_access() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    assert!(s.is_empty());

    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    assert_eq!(s.range(), 0..3);
    assert_eq!(s.len(), 3);
    assert_eq!(s.layout().len(), 2);
    assert_eq!(s.timestamps(), &[ts(100), ts(200), ts(300)]);
    assert_eq!(s.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(s.at(0).unwrap().1.data(), &[1.0, 2.0]);
    assert_eq!(
        s.at(1).unwrap(),
        (ts(200), ArrayView::from_slice([2], &[3.0, 4.0])),
    );
    assert_eq!(s.at(2).unwrap().1.data(), &[5.0, 6.0]);
    assert_eq!(s.at(3), None);
}

#[test]
fn series_scalar() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    assert_eq!(s.layout().len(), 1);

    s.push(ts(1), ArrayView::from_slice([], &[10.0]));
    s.push(ts(2), ArrayView::from_slice([], &[20.0]));

    assert_eq!(s.range(), 0..2);
    assert_eq!(s.at(0).unwrap().1[[]], 10.0);
    assert_eq!(s.at(1).unwrap().1[[]], 20.0);
    assert_eq!(s.timestamps().last(), Some(&ts(2)));
}

#[test]
fn series_asof() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    s.push(ts(100), ArrayView::from_slice([], &[1.0]));
    s.push(ts(200), ArrayView::from_slice([], &[2.0]));
    s.push(ts(300), ArrayView::from_slice([], &[3.0]));

    assert_eq!(s.asof(ts(50)).map(|v| v.data()), None);
    assert_eq!(s.asof(ts(100)).map(|v| v.data()), Some([1.0].as_slice()));
    assert_eq!(s.asof(ts(150)).map(|v| v.data()), Some([1.0].as_slice()));
    assert_eq!(s.asof(ts(200)).map(|v| v.data()), Some([2.0].as_slice()));
    assert_eq!(s.asof(ts(250)).map(|v| v.data()), Some([2.0].as_slice()));
    assert_eq!(s.asof(ts(300)).map(|v| v.data()), Some([3.0].as_slice()));
    assert_eq!(s.asof(ts(999)).map(|v| v.data()), Some([3.0].as_slice()));
}

#[test]
#[should_panic(expected = "push: extents mismatch")]
fn push_wrong_extents() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    let row = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    s.push(ts(1), row.view());
}

#[test]
fn element_layout() {
    let s = Series::<f64, 2>::new_unbounded([3, 4]);
    assert_eq!(s.layout().extents(), [3, 4]);
    assert_eq!(s.layout().len(), 12);
    assert!(s.layout().is_contiguous());
}

#[test]
fn from_parts_round_trips() {
    let s = Series::from_parts(
        [2],
        vec![ts(100), ts(200)],
        vec![1.0, 2.0, 3.0, 4.0],
        Retention::unbounded(),
    );
    assert_eq!(s.range(), 0..2);
    assert_eq!(s.timestamps().last(), Some(&ts(200)));
    assert_eq!(s.at(1).unwrap().1.data(), &[3.0, 4.0]);

    // `push` builds the same series.
    let mut p = Series::<f64, 1>::new_unbounded([2]);
    p.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    p.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    assert_eq!(p, s);
}

#[test]
#[should_panic(expected = "from_parts: 2 elements of stride 2 expect 4 scalars, got 3")]
fn from_parts_wrong_len() {
    let _ = Series::<f64, 1>::from_parts(
        [2],
        vec![ts(1), ts(2)],
        vec![1.0, 2.0, 3.0],
        Retention::unbounded(),
    );
}

#[test]
fn push_materializes_a_strided_view() {
    // A strided element must land packed row-major in the series.
    let panel = Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let col1 = ArrayView::from_parts(Strided::new([2], [3]), &panel.data()[1..]);
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push(ts(100), col1);
    assert_eq!(s.data(), &[2.0, 5.0]);
}

#[test]
fn search() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    s.push(ts(100), ArrayView::from_slice([], &[1.0]));
    s.push(ts(200), ArrayView::from_slice([], &[2.0]));
    s.push(ts(300), ArrayView::from_slice([], &[3.0]));

    assert_eq!(s.search(ts(50)), 0); // before all
    assert_eq!(s.search(ts(100)), 0); // exact first
    assert_eq!(s.search(ts(150)), 1); // between
    assert_eq!(s.search(ts(200)), 1); // exact second
    assert_eq!(s.search(ts(300)), 2); // exact last
    assert_eq!(s.search(ts(999)), 3); // after all
}

// -- SeriesView --------------------------------------------------------------

#[test]
fn view_window_and_elements() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    let v = s.view();
    assert_eq!(v.len(), 3);
    assert!(!v.is_empty());
    assert_eq!(v.layout().len(), 2);
    assert_eq!(v.timestamps(), &[ts(100), ts(200), ts(300)]);
    assert_eq!(v.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(v.at(1).unwrap().1.data(), &[3.0, 4.0]);
    assert_eq!(v.at(2).unwrap().0, ts(300));
    assert_eq!(v.at(3), None);

    // Sub-window.
    let w = v.window(1..3);
    assert_eq!(w.len(), 2);
    assert_eq!(w.timestamps(), &[ts(200), ts(300)]);
    assert_eq!(w.data(), &[3.0, 4.0, 5.0, 6.0]);

    // The tail: the last n elements.
    let t = v.window(v.len() - 2..v.len());
    assert_eq!(t.timestamps(), &[ts(200), ts(300)]);

    // An empty window.
    assert_eq!(v.window(0..0).len(), 0);
}

#[test]
fn view_asof_and_search() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    s.push(ts(100), ArrayView::from_slice([], &[1.0]));
    s.push(ts(200), ArrayView::from_slice([], &[2.0]));
    let v = s.view();
    assert_eq!(v.asof(ts(50)).map(|v| v.data()), None);
    assert_eq!(v.asof(ts(150)).map(|v| v.data()), Some([1.0].as_slice()));
    assert_eq!(v.search(ts(150)), 1);
    assert_eq!(v.search(ts(999)), 2);
}

#[test]
fn view_to_array_view() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    // Whole window: [3, 2], contiguous.
    let av = s.view().to_array_view::<2>();
    assert_eq!(av.extents(), [3, 2]);
    assert!(av.as_slice().is_some());
    assert_eq!(&*av.to_contiguous(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Sub-windows convert too.
    let av = s.view().window(1..3).to_array_view::<2>();
    assert_eq!(av.extents(), [2, 2]);
    assert_eq!(&*av.to_contiguous(), &[3.0, 4.0, 5.0, 6.0]);

    // Owned copies: through the view, or by consuming the series.
    let arr = s.view().window(2..3).to_array_view::<2>().to_array();
    assert_eq!(arr.extents(), [1, 2]);
    assert_eq!(arr.data(), &[5.0, 6.0]);
    let arr = s.to_array::<2>();
    assert_eq!(arr.extents(), [3, 2]);
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
#[should_panic(expected = "to_array_view: M (3) must be N + 1 (2)")]
fn view_to_array_view_wrong_rank() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    let _ = s.view().to_array_view::<3>();
}

#[test]
fn view_from_slice_checks_len() {
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 3.0, 4.0];
    let v = SeriesView::<f64, 1>::from_slice([2], &tss, &vals);
    assert_eq!(v.len(), 2);
    assert_eq!(v.at(1).unwrap().1.data(), &[3.0, 4.0]);
}

#[test]
#[should_panic(expected = "from_slice: 2 elements of stride 2 expect 4 scalars, got 3")]
fn view_from_slice_wrong_len() {
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 3.0];
    let _ = SeriesView::<f64, 1>::from_slice([2], &tss, &vals);
}

#[test]
fn view_from_parts_with_padding() {
    // Elements of extent [2] packed with stride 3 — one pad scalar after each.
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 9.0, 3.0, 4.0, 9.0];
    let v = SeriesView::<f64, 1>::from_parts(Strided::new([2], [1]), 3, &tss, &vals);
    assert_eq!(v.len(), 2);
    // Padded: not one flat slice, but elements read correctly...
    assert!(v.as_slice().is_none());
    assert_eq!(v.at(0).unwrap().1[[0]], 1.0);
    assert_eq!(v.at(1).unwrap().1[[1]], 4.0);
    // ...and `to_contiguous` re-packs, dropping the padding.
    assert_eq!(&*v.to_contiguous(), &[1.0, 2.0, 3.0, 4.0]);
    // So does `to_series`.
    let owned = v.to_series(Retention::unbounded());
    assert_eq!(owned.data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(owned.timestamps(), &[ts(1), ts(2)]);
}

#[test]
#[should_panic(expected = "from_parts: shape spans 2 scalars, got 1")]
fn view_from_parts_stride_too_small() {
    let tss = [ts(1)];
    let vals = [1.0];
    let _ = SeriesView::<f64, 1>::from_parts(Strided::new([2], [1]), 1, &tss, &vals);
}

#[test]
fn view_to_series() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    // Whole-window copy of an unbounded series equals the original.
    let owned = s.view().to_series(Retention::unbounded());
    assert_eq!(owned, s);

    // A sub-window copies just its elements into a fresh, same-rank series.
    let sub = s.view().window(1..3).to_series(Retention::unbounded());
    assert_eq!(sub.layout().extents(), [2]);
    assert_eq!(sub.timestamps(), &[ts(200), ts(300)]);
    assert_eq!(sub.data(), &[3.0, 4.0, 5.0, 6.0]);
}

// -- Retention ---------------------------------------------------------------

#[test]
fn count_retention_bounds_storage_and_preserves_logical_reads() {
    // Keep the most recent 3 elements; push 10.
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    // The logical range still ends at the full count; physical storage is
    // bounded (<= 2x the window thanks to amortized compaction).
    assert_eq!(s.range().end, 10);
    assert!(s.len() <= 6, "retained {} > 2x window", s.len());
    assert!(s.len() >= 3, "fewer than the window retained");
    assert_eq!(s.range().start, 10 - s.len());

    // The required window [7, 10) reads identically to an unbounded series.
    assert_eq!(s.at(7).unwrap().1.data(), &[7.0]);
    assert_eq!(s.at(8).unwrap().1.data(), &[8.0]);
    assert_eq!(
        s.at(9).unwrap(),
        (ts(1000), ArrayView::from_slice([1], &[9.0])),
    );
    // Evicted logical indices read as `None`.
    assert_eq!(s.at(0), None);
    // Logical windows through the view: local index = logical - range().start.
    let base = s.range().start;
    assert_eq!(
        s.view().window(7 - base..10 - base).data(),
        &[7.0, 8.0, 9.0]
    );
}

#[test]
fn duration_retention_keeps_time_window() {
    // Keep everything within 250ns of the latest; ticks are 100ns apart.
    let mut s = Series::<f64, 1>::new([1], Retention::duration(Duration::from_nanos(250)));
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    // Latest ts = 1000; cutoff = 750 → keep ts in {800, 900, 1000} = indices 7, 8, 9.
    assert_eq!(s.range().end, 10);
    assert_eq!(s.at(9).unwrap().1.data(), &[9.0]);
    assert_eq!(s.at(8).unwrap().1.data(), &[8.0]);
    assert_eq!(s.at(7).unwrap().1.data(), &[7.0]);
    assert!(
        s.range().start <= 7,
        "kept window too small: base {}",
        s.range().start
    );
}

#[test]
fn asof_and_search_use_logical_indices_under_retention() {
    // Regression: `asof` once mixed a physical partition point into a
    // logical accessor, which broke as soon as retention evicted elements.
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    assert!(s.range().start > 0, "retention must have evicted something");

    assert_eq!(s.asof(ts(1000)).unwrap().data(), &[9.0]);
    assert_eq!(s.asof(ts(850)).unwrap().data(), &[7.0]);
    // Before the retained window: None, though older elements once matched.
    assert_eq!(s.asof(ts(100)).map(|v| v.data()), None);

    // `search` returns logical indices: the first ts >= 850 is t900,
    // logically element 8.
    assert_eq!(s.search(ts(850)), 8);
    assert_eq!(s.search(ts(9999)), s.range().end);
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
        bounded.push(t, ArrayView::from_slice([2], &row));
        unbounded.push(t, ArrayView::from_slice([2], &row));
        assert_eq!(bounded.range().end, unbounded.range().end);
        for j in bounded.range() {
            assert_eq!(bounded.at(j), unbounded.at(j));
        }
        assert_eq!(
            bounded.iter().next_back().unwrap(),
            unbounded.iter().next_back().unwrap(),
        );
    }
}

// -- Iteration ---------------------------------------------------------------

#[test]
fn series_iter_yields_timestamp_element_pairs() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    assert_eq!(s.iter().len(), 3);
    let collected: Vec<(i64, Vec<f64>)> = s
        .iter()
        .map(|(t, v)| (t.as_offset().as_nanos(), v.iter().collect()))
        .collect();
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
    s.push(ts(1), ArrayView::from_slice([], &[10.0]));
    s.push(ts(2), ArrayView::from_slice([], &[20.0]));
    s.push(ts(3), ArrayView::from_slice([], &[30.0]));

    // `IntoIterator` on the `Copy` view (the "owned" entry point).
    let fwd: Vec<f64> = s.view().into_iter().map(|(_, v)| v[[]]).collect();
    assert_eq!(fwd, vec![10.0, 20.0, 30.0]);

    // Double-ended: reversed by timestamp.
    let rev: Vec<i64> = s
        .iter()
        .rev()
        .map(|(t, _)| t.as_offset().as_nanos())
        .collect();
    assert_eq!(rev, vec![3, 2, 1]);
}

#[test]
fn series_iter_walks_only_the_retained_window() {
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    // Iterating covers exactly the retained window.
    let vals: Vec<f64> = s.iter().map(|(_, v)| v[[0]]).collect();
    assert_eq!(vals.len(), s.len());
    assert_eq!(s.iter().count(), s.timestamps().len());
    // The most recent element (logical index 9) is always retained.
    assert_eq!(vals.last().copied(), Some(9.0));
    // Timestamps line up with the retained window.
    let iter_ts: Vec<i64> = s.iter().map(|(t, _)| t.as_offset().as_nanos()).collect();
    let want_ts: Vec<i64> = s
        .timestamps()
        .iter()
        .map(|t| t.as_offset().as_nanos())
        .collect();
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
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    assert_eq!(s.clone().into_iter().len(), 3);
    // By value: each element is an owned `Array` moved out of the series.
    let collected: Vec<(i64, Vec<f64>)> = s
        .into_iter()
        .map(|(t, a)| (t.as_offset().as_nanos(), a.data().to_vec()))
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
fn series_into_iter_double_ended() {
    // Rank-1 blocks so the scalar order within each element matters.
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    // Regression: stepping from the back must neither reverse the scalars
    // within a block nor consume the elements still ahead of the front.
    let mut it = s.clone().into_iter();
    let (t, a) = it.next_back().unwrap();
    assert_eq!((t, a.data()), (ts(300), [5.0, 6.0].as_slice()));
    let (t, a) = it.next().unwrap();
    assert_eq!((t, a.data()), (ts(100), [1.0, 2.0].as_slice()));
    let (t, a) = it.next_back().unwrap();
    assert_eq!((t, a.data()), (ts(200), [3.0, 4.0].as_slice()));
    assert!(it.next().is_none());
    assert!(it.next_back().is_none());

    // A full reverse matches the forward order reversed.
    let fwd: Vec<Vec<f64>> = s
        .clone()
        .into_iter()
        .map(|(_, a)| a.data().to_vec())
        .collect();
    let mut rev: Vec<Vec<f64>> = s
        .into_iter()
        .rev()
        .map(|(_, a)| a.data().to_vec())
        .collect();
    rev.reverse();
    assert_eq!(rev, fwd);
}

#[test]
fn series_into_iter_double_ended_over_retained_window() {
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    let retained = s.len();
    // Reversed owned iteration walks exactly the retained window, newest first.
    let rev: Vec<f64> = s.into_iter().rev().map(|(_, a)| a.data()[0]).collect();
    assert_eq!(rev.len(), retained);
    for (k, v) in rev.iter().enumerate() {
        assert_eq!(*v, 9.0 - k as f64);
    }
}

#[test]
fn series_into_iter_rank2_and_empty() {
    // Rank-2 elements round-trip their row-major scalars.
    let mut s = Series::<f64, 2>::new_unbounded([2, 2]);
    s.push(ts(1), ArrayView::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]));
    let (t, a) = s.into_iter().next().unwrap();
    assert_eq!(t, ts(1));
    assert_eq!(a.extents(), [2, 2]);
    assert_eq!(a.data(), &[1.0, 2.0, 3.0, 4.0]);

    let empty = Series::<f64, 1>::new_unbounded([2]);
    assert_eq!(empty.into_iter().count(), 0);
}
