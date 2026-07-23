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
    assert_eq!(&*s.at(0).unwrap().1.to_contiguous(), &[1.0, 2.0]);
    assert_eq!(
        s.at(1).unwrap(),
        (ts(200), ArrayView::from_slice([2], &[3.0, 4.0])),
    );
    assert_eq!(&*s.at(2).unwrap().1.to_contiguous(), &[5.0, 6.0]);
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

    // A rank-0 element holds its scalar at the empty index.
    assert_eq!(s.asof(ts(50)).map(|v| v[[]]), None);
    assert_eq!(s.asof(ts(100)).map(|v| v[[]]), Some(1.0));
    assert_eq!(s.asof(ts(150)).map(|v| v[[]]), Some(1.0));
    assert_eq!(s.asof(ts(200)).map(|v| v[[]]), Some(2.0));
    assert_eq!(s.asof(ts(250)).map(|v| v[[]]), Some(2.0));
    assert_eq!(s.asof(ts(300)).map(|v| v[[]]), Some(3.0));
    assert_eq!(s.asof(ts(999)).map(|v| v[[]]), Some(3.0));
}

#[test]
#[should_panic(expected = "extents mismatch")]
fn push_wrong_extents() {
    let mut s = Series::<f64, 1>::new_unbounded([2]);
    let row = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    s.push(ts(1), row.view());
}

#[test]
fn element_layout() {
    let s = Series::<f64, 2>::new_unbounded([3, 4]);
    assert_eq!(s.extents(), [3, 4]);
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
    assert_eq!(&*s.at(1).unwrap().1.to_contiguous(), &[3.0, 4.0]);

    // `push` builds the same series.
    let mut p = Series::<f64, 1>::new_unbounded([2]);
    p.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    p.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    assert_eq!(p, s);
}

#[test]
#[should_panic(expected = "expect 4 scalars, got 3")]
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
    assert_eq!(&*v.to_contiguous(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(&*v.at(1).unwrap().1.to_contiguous(), &[3.0, 4.0]);
    assert_eq!(v.at(2).unwrap().0, ts(300));
    assert_eq!(v.at(3), None);

    // Sub-window.
    let w = v.window(1..3);
    assert_eq!(w.len(), 2);
    assert_eq!(w.timestamps(), &[ts(200), ts(300)]);
    assert_eq!(&*w.to_contiguous(), &[3.0, 4.0, 5.0, 6.0]);

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
    assert_eq!(v.asof(ts(50)).map(|v| v[[]]), None);
    assert_eq!(v.asof(ts(150)).map(|v| v[[]]), Some(1.0));
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
#[should_panic(expected = "M (3) must be N + 1 (2)")]
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
    assert_eq!(&*v.at(1).unwrap().1.to_contiguous(), &[3.0, 4.0]);
}

#[test]
#[should_panic(expected = "expect 4 scalars, got 3")]
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
#[should_panic(expected = "span 5 scalars, got 4")]
fn view_from_parts_data_too_short() {
    // Two elements of extent [2] laid 3 apart address up to offset 4, so the
    // data must hold 5 scalars — the tail beyond that is what may be missing,
    // never the space the elements themselves address.
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 9.0, 3.0];
    let _ = SeriesView::<f64, 1>::from_parts(Strided::new([2], [1]), 3, &tss, &vals);
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
    assert_eq!(sub.extents(), [2]);
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
    assert_eq!(&*s.at(7).unwrap().1.to_contiguous(), &[7.0]);
    assert_eq!(&*s.at(8).unwrap().1.to_contiguous(), &[8.0]);
    assert_eq!(
        s.at(9).unwrap(),
        (ts(1000), ArrayView::from_slice([1], &[9.0])),
    );
    // Evicted logical indices read as `None`.
    assert_eq!(s.at(0), None);
    // Logical windows through the view: local index = logical - range().start.
    let base = s.range().start;
    assert_eq!(
        &*s.view().window(7 - base..10 - base).to_contiguous(),
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
    assert_eq!(&*s.at(9).unwrap().1.to_contiguous(), &[9.0]);
    assert_eq!(&*s.at(8).unwrap().1.to_contiguous(), &[8.0]);
    assert_eq!(&*s.at(7).unwrap().1.to_contiguous(), &[7.0]);
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

    assert_eq!(&*s.asof(ts(1000)).unwrap().to_contiguous(), &[9.0]);
    assert_eq!(&*s.asof(ts(850)).unwrap().to_contiguous(), &[7.0]);
    // Before the retained window: None, though older elements once matched.
    assert_eq!(s.asof(ts(100)).map(|v| v[[0]]), None);

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
            bounded.iter().last().unwrap(),
            unbounded.iter().last().unwrap()
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
        assert_eq!(v.layout().len(), 2);
        let _ = t;
        n += 1;
    }
    assert_eq!(n, 3);
}

#[test]
fn series_view_into_iter() {
    let mut s = Series::<f64, 0>::new_unbounded([]);
    s.push(ts(1), ArrayView::from_slice([], &[10.0]));
    s.push(ts(2), ArrayView::from_slice([], &[20.0]));
    s.push(ts(3), ArrayView::from_slice([], &[30.0]));

    // `IntoIterator` on the `Copy` view (the "owned" entry point).
    let fwd: Vec<f64> = s.view().into_iter().map(|(_, v)| v[[]]).collect();
    assert_eq!(fwd, vec![10.0, 20.0, 30.0]);

    // Timestamps come out in window order.
    let times: Vec<i64> = s.iter().map(|(t, _)| t.as_offset().as_nanos()).collect();
    assert_eq!(times, vec![1, 2, 3]);
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
    assert!(s.view().into_iter().next().is_none());
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
fn series_into_iter_over_retained_window() {
    let mut s = Series::<f64, 1>::new([1], Retention::count(3));
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    let retained = s.len();
    let newest = s.range().end - 1;
    // Owned iteration walks exactly the retained window, oldest first.
    let vals: Vec<f64> = s.into_iter().map(|(_, a)| a.data()[0]).collect();
    assert_eq!(vals.len(), retained);
    for (k, v) in vals.iter().enumerate() {
        assert_eq!(*v, (newest - retained + 1 + k) as f64);
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

#[test]
fn view_slicing_selects_element_sub_regions() {
    // Three elements of extent [2, 3], packed.
    let tss = [ts(100), ts(200), ts(300)];
    let vals: Vec<f64> = (0..18).map(f64::from).collect();
    let v = SeriesView::<f64, 2>::from_slice([2, 3], &tss, &vals);

    // Slicing the element axes keeps every timestamp.
    let s = v.slice((1..2, 1..3));
    assert_eq!(s.len(), 3);
    assert_eq!(s.timestamps(), &tss);
    assert_eq!(s.extents(), [1, 2]);
    // Element i now reads its own sub-block, from a re-based data slice.
    assert_eq!(&*s.at(0).unwrap().1.to_contiguous(), &[4.0, 5.0]);
    assert_eq!(&*s.at(1).unwrap().1.to_contiguous(), &[10.0, 11.0]);
    assert_eq!(&*s.at(2).unwrap().1.to_contiguous(), &[16.0, 17.0]);
    // Which agrees with slicing each element's view directly.
    for (i, (_, e)) in v.iter().enumerate() {
        assert_eq!(
            &*s.at(i).unwrap().1.to_contiguous(),
            &*e.slice((1..2, 1..3)).to_contiguous(),
        );
    }

    // Iteration walks the same sliced elements as indexed access.
    let walked: Vec<_> = s.iter().map(|(_, e)| e.to_contiguous().to_vec()).collect();
    assert_eq!(
        walked,
        vec![vec![4.0, 5.0], vec![10.0, 11.0], vec![16.0, 17.0]]
    );

    // Windows of a sliced view still read correctly.
    let w = s.window(1..3);
    assert_eq!(w.len(), 2);
    assert_eq!(&*w.at(1).unwrap().1.to_contiguous(), &[16.0, 17.0]);
    assert_eq!(&*w.to_contiguous(), &[10.0, 11.0, 16.0, 17.0]);
}

#[test]
fn view_slicing_reshapes_elements() {
    let tss = [ts(100), ts(200)];
    let vals: Vec<f64> = (0..12).map(f64::from).collect();
    let v = SeriesView::<f64, 2>::from_slice([2, 3], &tss, &vals);

    // An index drops an element axis: [2, 3] elements -> [3].
    let s: SeriesView<f64, 1> = v.slice_reshape((1, ..));
    assert_eq!(s.len(), 2);
    assert_eq!(s.extents(), [3]);
    assert_eq!(&*s.at(0).unwrap().1.to_contiguous(), &[3.0, 4.0, 5.0]);
    assert_eq!(&*s.at(1).unwrap().1.to_contiguous(), &[9.0, 10.0, 11.0]);

    // And `()` adds one: [2, 3] -> [2, 1, 3].
    let s: SeriesView<f64, 3> = v.slice_reshape((.., (), ..));
    assert_eq!(s.extents(), [2, 1, 3]);
    assert_eq!(
        &*s.at(1).unwrap().1.to_contiguous(),
        &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );

    // A sliced series copies into an owned one with the sliced extents.
    let owned = v.slice((.., 1..3)).to_series(Retention::unbounded());
    assert_eq!(owned.layout().extents(), [2, 2]);
    assert_eq!(owned.data(), &[1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0]);
    assert_eq!(owned.timestamps(), &tss);
}

#[test]
fn series_view_eq_compares_timestamps_and_elements() {
    let tss = [ts(1), ts(2)];
    let packed = SeriesView::<f64, 1>::from_slice([2], &tss, &[1.0, 2.0, 3.0, 4.0]);

    // The same elements laid 3 apart, with a pad scalar after each and a
    // trailing scalar the index space never reaches.
    let padded_vals = [1.0, 2.0, 9.0, 3.0, 4.0, 9.0, 9.0];
    let padded = SeriesView::from_parts(Strided::new([2], [1]), 3, &tss, &padded_vals);
    assert_eq!(packed, padded);

    // Differing timestamps, or differing elements, make them unequal.
    let other_ts = [ts(1), ts(3)];
    assert_ne!(
        packed,
        SeriesView::<f64, 1>::from_slice([2], &other_ts, &[1.0, 2.0, 3.0, 4.0]),
    );
    assert_ne!(
        packed,
        SeriesView::<f64, 1>::from_slice([2], &tss, &[1.0, 2.0, 3.0, 5.0]),
    );
}
