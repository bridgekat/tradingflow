use tradingflow_data::layout::Strided;
use tradingflow_data::{Array, ArrayView, Duration, Instant, Layout, Series, SeriesView};

fn ts(n: i64) -> Instant {
    Instant::from_offset(Duration::from_nanos(n))
}

#[test]
fn series_push_and_access() {
    let mut s = Series::<f64, 1>::new([2]);
    assert!(s.is_empty());

    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    assert_eq!(s.range(), 0..3);
    assert_eq!(s.len(), 3);
    assert_eq!(s.layout().len(), 2);
    assert_eq!(s.instants(), &[ts(100), ts(200), ts(300)]);
    assert_eq!(s.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(&*s.at(0).1.to_contiguous(), &[1.0, 2.0]);
    assert_eq!(s.at(1), (ts(200), ArrayView::from_slice([2], &[3.0, 4.0])));
    assert_eq!(&*s.at(2).1.to_contiguous(), &[5.0, 6.0]);
}

#[test]
#[should_panic(expected = "out of retained range")]
fn series_at_past_end_panics() {
    let mut s = Series::<f64, 1>::new([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    let _ = s.at(1);
}

#[test]
fn series_scalar() {
    let mut s = Series::<f64, 0>::new([]);
    assert_eq!(s.layout().len(), 1);

    s.push(ts(1), ArrayView::from_slice([], &[10.0]));
    s.push(ts(2), ArrayView::from_slice([], &[20.0]));

    assert_eq!(s.range(), 0..2);
    assert_eq!(s.at(0).1[[]], 10.0);
    assert_eq!(s.at(1).1[[]], 20.0);
    assert_eq!(s.instants().last(), Some(&ts(2)));
}

#[test]
fn series_asof() {
    let mut s = Series::<f64, 0>::new([]);
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
    let mut s = Series::<f64, 1>::new([2]);
    let row = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    s.push(ts(1), row.view());
}

#[test]
fn element_layout() {
    let s = Series::<f64, 2>::new([3, 4]);
    assert_eq!(s.extents(), [3, 4]);
    assert_eq!(s.layout().len(), 12);
    assert!(s.layout().is_contiguous());
}

#[test]
fn from_parts_round_trips() {
    let s = Series::from_parts([2], vec![ts(100), ts(200)], vec![1.0, 2.0, 3.0, 4.0], 0);
    assert_eq!(s.range(), 0..2);
    assert_eq!(s.instants().last(), Some(&ts(200)));
    assert_eq!(&*s.at(1).1.to_contiguous(), &[3.0, 4.0]);

    // `push` builds the same series.
    let mut p = Series::<f64, 1>::new([2]);
    p.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    p.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    assert_eq!(p, s);
}

#[test]
fn from_parts_with_base() {
    // The rows are logical elements [4, 6): a reconstructed tail of a longer
    // stream (e.g. deserialized after trims).
    let s = Series::from_parts([1], vec![ts(500), ts(600)], vec![5.0, 6.0], 4);
    assert_eq!(s.range(), 4..6);
    assert_eq!(s.len(), 2);
    assert_eq!(s.at(4), (ts(500), ArrayView::from_slice([1], &[5.0])));
    assert_eq!(s.at(5).1[[0]], 6.0);
    assert_eq!(s.view().range(), 4..6);
}

#[test]
#[should_panic(expected = "expect 4 scalars, got 3")]
fn from_parts_wrong_len() {
    let _ = Series::<f64, 1>::from_parts([2], vec![ts(1), ts(2)], vec![1.0, 2.0, 3.0], 0);
}

#[test]
fn push_materializes_a_strided_view() {
    // A strided element must land packed row-major in the series.
    let panel = Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let col1 = ArrayView::from_parts(Strided::new([2], [3]), &panel.data()[1..]);
    let mut s = Series::<f64, 1>::new([2]);
    s.push(ts(100), col1);
    assert_eq!(s.data(), &[2.0, 5.0]);
}

// -- Trim --------------------------------------------------------------------

#[test]
fn trim_advances_range_and_preserves_logical_reads() {
    let mut full = Series::<f64, 1>::new([1]);
    let mut trimmed = Series::<f64, 1>::new([1]);
    for i in 0..10i64 {
        let t = ts((i + 1) * 100);
        full.push(t, ArrayView::from_slice([1], &[i as f64]));
        trimmed.push(t, ArrayView::from_slice([1], &[i as f64]));
    }
    trimmed.trim(4);

    // The logical range keeps its end; the start advances; storage shrinks.
    assert_eq!(trimmed.range(), 4..10);
    assert_eq!(trimmed.len(), 6);
    assert_eq!(trimmed.data().len(), 6);
    assert_eq!(trimmed.instants().first(), Some(&ts(500)));

    // Every retained logical index reads identically to the untrimmed twin.
    for j in trimmed.range() {
        assert_eq!(trimmed.at(j), full.at(j));
    }
}

#[test]
fn trim_is_cumulative() {
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(2);
    s.trim(3);
    assert_eq!(s.range(), 5..10);
    assert_eq!(s.at(5).1[[0]], 5.0);
}

#[test]
fn trim_zero_and_trim_all() {
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..3i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(0);
    assert_eq!(s.range(), 0..3);

    // Trimming everything empties the series but keeps the logical position.
    s.trim(3);
    assert!(s.is_empty());
    assert_eq!(s.range(), 3..3);

    // A later push continues the logical stream.
    s.push(ts(400), ArrayView::from_slice([1], &[3.0]));
    assert_eq!(s.range(), 3..4);
    assert_eq!(s.at(3), (ts(400), ArrayView::from_slice([1], &[3.0])));
}

#[test]
#[should_panic(expected = "trim: count 4 > len 3")]
fn trim_more_than_len_panics() {
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..3i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(4);
}

#[test]
#[should_panic(expected = "out of retained range")]
fn series_at_below_range_panics_after_trim() {
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..5i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(2);
    let _ = s.at(1); // evicted
}

#[test]
fn asof_after_trim() {
    // Regression: `asof` once mixed a window-local partition point into the
    // logical accessor, reading the wrong element as soon as `base > 0`.
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(4);
    assert_eq!(s.range(), 4..10);

    assert_eq!(&*s.asof(ts(1000)).unwrap().to_contiguous(), &[9.0]);
    assert_eq!(&*s.asof(ts(850)).unwrap().to_contiguous(), &[7.0]);
    assert_eq!(&*s.asof(ts(500)).unwrap().to_contiguous(), &[4.0]);
    // Before the retained window: None, though older elements once matched.
    assert_eq!(s.asof(ts(499)).map(|v| v[[0]]), None);

    // The view agrees.
    let v = s.view();
    assert_eq!(&*v.asof(ts(850)).unwrap().to_contiguous(), &[7.0]);
}

// -- SeriesView --------------------------------------------------------------

#[test]
fn view_window_and_elements() {
    let mut s = Series::<f64, 1>::new([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    let v = s.view();
    assert_eq!(v.len(), 3);
    assert!(!v.is_empty());
    assert_eq!(v.range(), 0..3);
    assert_eq!(v.layout().len(), 2);
    assert_eq!(v.instants(), &[ts(100), ts(200), ts(300)]);
    assert_eq!(&*v.to_contiguous(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(&*v.at(1).1.to_contiguous(), &[3.0, 4.0]);
    assert_eq!(v.at(2).0, ts(300));

    // Sub-window, addressed by logical indices, which it preserves.
    let w = v.window(1..3);
    assert_eq!(w.len(), 2);
    assert_eq!(w.range(), 1..3);
    assert_eq!(w.instants(), &[ts(200), ts(300)]);
    assert_eq!(&*w.to_contiguous(), &[3.0, 4.0, 5.0, 6.0]);
    assert_eq!(w.at(1), v.at(1));

    // The tail: the last n elements.
    let t = v.window(v.range().end - 2..v.range().end);
    assert_eq!(t.instants(), &[ts(200), ts(300)]);

    // An empty window.
    assert_eq!(v.window(0..0).len(), 0);
}

#[test]
#[should_panic(expected = "out of retained range")]
fn view_at_past_end_panics() {
    let mut s = Series::<f64, 1>::new([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    let _ = s.view().at(1);
}

#[test]
fn view_carries_base_after_trim() {
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(4);

    let v = s.view();
    assert_eq!(v.range(), 4..10);
    assert_eq!(v.len(), 6);
    assert_eq!(v.at(7), (ts(800), ArrayView::from_slice([1], &[7.0])));

    // Windows take logical bounds directly — no manual base arithmetic.
    let w = v.window(7..10);
    assert_eq!(w.range(), 7..10);
    assert_eq!(&*w.to_contiguous(), &[7.0, 8.0, 9.0]);
    // Nested windows stay in the same logical frame.
    let n = w.window(8..9);
    assert_eq!(n.range(), 8..9);
    assert_eq!(n.at(8).1[[0]], 8.0);
    // Empty windows at either edge of the retained range.
    assert_eq!(v.window(4..4).len(), 0);

    // Element-wise slicing preserves the logical frame too.
    let sliced = v.slice((0..1,));
    assert_eq!(sliced.range(), 4..10);
    assert_eq!(sliced.at(9).1[[0]], 9.0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn view_window_below_base_panics() {
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..5i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(2);
    let _ = s.view().window(1..4); // starts before the retained range
}

#[test]
#[should_panic(expected = "out of bounds")]
#[allow(clippy::reversed_empty_ranges)]
fn view_window_inverted_panics() {
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..5i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    let _ = s.view().window(3..1);
}

#[test]
fn view_asof() {
    let mut s = Series::<f64, 0>::new([]);
    s.push(ts(100), ArrayView::from_slice([], &[1.0]));
    s.push(ts(200), ArrayView::from_slice([], &[2.0]));
    let v = s.view();
    assert_eq!(v.asof(ts(50)).map(|v| v[[]]), None);
    assert_eq!(v.asof(ts(150)).map(|v| v[[]]), Some(1.0));
    assert_eq!(v.asof(ts(999)).map(|v| v[[]]), Some(2.0));
}

#[test]
fn view_to_array_view() {
    let mut s = Series::<f64, 1>::new([2]);
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
    let mut s = Series::<f64, 1>::new([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    let _ = s.view().to_array_view::<3>();
}

#[test]
fn view_from_slice_checks_len() {
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 3.0, 4.0];
    let v = SeriesView::<f64, 1>::from_slice([2], &tss, &vals, 0);
    assert_eq!(v.len(), 2);
    assert_eq!(&*v.at(1).1.to_contiguous(), &[3.0, 4.0]);
}

#[test]
#[should_panic(expected = "expect 4 scalars, got 3")]
fn view_from_slice_wrong_len() {
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 3.0];
    let _ = SeriesView::<f64, 1>::from_slice([2], &tss, &vals, 0);
}

#[test]
fn view_from_parts_with_padding() {
    // Elements of extent [2] packed with stride 3 — one pad scalar after each.
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 9.0, 3.0, 4.0, 9.0];
    let v = SeriesView::<f64, 1>::from_parts(Strided::new([2], [1]), 3, &tss, &vals, 0);
    assert_eq!(v.len(), 2);
    // Padded: not one flat slice, but elements read correctly...
    assert!(v.as_slice().is_none());
    assert_eq!(v.at(0).1[[0]], 1.0);
    assert_eq!(v.at(1).1[[1]], 4.0);
    // ...and `to_contiguous` re-packs, dropping the padding.
    assert_eq!(&*v.to_contiguous(), &[1.0, 2.0, 3.0, 4.0]);
    // So does `to_series`.
    let owned = v.to_series();
    assert_eq!(owned.data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(owned.instants(), &[ts(1), ts(2)]);
}

#[test]
#[should_panic(expected = "span 5 scalars, got 4")]
fn view_from_parts_data_too_short() {
    // Two elements of extent [2] laid 3 apart address up to offset 4, so the
    // data must hold 5 scalars — the tail beyond that is what may be missing,
    // never the space the elements themselves address.
    let tss = [ts(1), ts(2)];
    let vals = [1.0, 2.0, 9.0, 3.0];
    let _ = SeriesView::<f64, 1>::from_parts(Strided::new([2], [1]), 3, &tss, &vals, 0);
}

#[test]
fn view_to_series() {
    let mut s = Series::<f64, 1>::new([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));
    s.push(ts(300), ArrayView::from_slice([2], &[5.0, 6.0]));

    // Whole-window copy equals the original.
    let owned = s.view().to_series();
    assert_eq!(owned, s);

    // A sub-window copies just its elements into a fresh, same-rank series,
    // preserving the logical frame.
    let sub = s.view().window(1..3).to_series();
    assert_eq!(sub.extents(), [2]);
    assert_eq!(sub.range(), 1..3);
    assert_eq!(sub.instants(), &[ts(200), ts(300)]);
    assert_eq!(sub.data(), &[3.0, 4.0, 5.0, 6.0]);
    assert_eq!(sub.at(2), s.at(2));

    // `From` is the same copy.
    let owned: Series<f64, 1> = s.view().into();
    assert_eq!(owned, s);
}

// -- Iteration ---------------------------------------------------------------

#[test]
fn series_iter_yields_instant_element_pairs() {
    let mut s = Series::<f64, 1>::new([2]);
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
    let mut s = Series::<f64, 0>::new([]);
    s.push(ts(1), ArrayView::from_slice([], &[10.0]));
    s.push(ts(2), ArrayView::from_slice([], &[20.0]));
    s.push(ts(3), ArrayView::from_slice([], &[30.0]));

    // `IntoIterator` on the `Copy` view (the "owned" entry point).
    let fwd: Vec<f64> = s.view().into_iter().map(|(_, v)| v[[]]).collect();
    assert_eq!(fwd, vec![10.0, 20.0, 30.0]);

    // Instants come out in window order.
    let times: Vec<i64> = s.iter().map(|(t, _)| t.as_offset().as_nanos()).collect();
    assert_eq!(times, vec![1, 2, 3]);
}

#[test]
fn series_iter_walks_only_the_retained_window() {
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(7);
    // Iterating covers exactly the retained window, oldest first.
    let vals: Vec<f64> = s.iter().map(|(_, v)| v[[0]]).collect();
    assert_eq!(vals, vec![7.0, 8.0, 9.0]);
    assert_eq!(s.iter().count(), s.len());
    // Instants line up with the retained window.
    let iter_ts: Vec<i64> = s.iter().map(|(t, _)| t.as_offset().as_nanos()).collect();
    let want_ts: Vec<i64> = s
        .instants()
        .iter()
        .map(|t| t.as_offset().as_nanos())
        .collect();
    assert_eq!(iter_ts, want_ts);
}

#[test]
fn series_iter_empty() {
    let s = Series::<f64, 1>::new([2]);
    assert_eq!(s.iter().count(), 0);
    assert!(s.iter().next().is_none());
    assert!(s.view().into_iter().next().is_none());
}

#[test]
fn series_into_iter_yields_owned_arrays() {
    let mut s = Series::<f64, 1>::new([2]);
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
    let mut s = Series::<f64, 1>::new([1]);
    for i in 0..10i64 {
        s.push(ts((i + 1) * 100), ArrayView::from_slice([1], &[i as f64]));
    }
    s.trim(7);
    // Owned iteration walks exactly the retained window, oldest first.
    let vals: Vec<f64> = s.into_iter().map(|(_, a)| a.data()[0]).collect();
    assert_eq!(vals, vec![7.0, 8.0, 9.0]);
}

#[test]
fn series_into_iter_rank2_and_empty() {
    // Rank-2 elements round-trip their row-major scalars.
    let mut s = Series::<f64, 2>::new([2, 2]);
    s.push(ts(1), ArrayView::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]));
    let (t, a) = s.into_iter().next().unwrap();
    assert_eq!(t, ts(1));
    assert_eq!(a.extents(), [2, 2]);
    assert_eq!(a.data(), &[1.0, 2.0, 3.0, 4.0]);

    let empty = Series::<f64, 1>::new([2]);
    assert_eq!(empty.into_iter().count(), 0);
}

// -- Element-wise slicing ----------------------------------------------------

#[test]
fn view_slicing_selects_element_sub_regions() {
    // Three elements of extent [2, 3], packed.
    let tss = [ts(100), ts(200), ts(300)];
    let vals: Vec<f64> = (0..18).map(f64::from).collect();
    let v = SeriesView::<f64, 2>::from_slice([2, 3], &tss, &vals, 0);

    // Slicing the element axes keeps every instant.
    let s = v.slice((1..2, 1..3));
    assert_eq!(s.len(), 3);
    assert_eq!(s.instants(), &tss);
    assert_eq!(s.extents(), [1, 2]);
    // Element i now reads its own sub-block, from a re-based data slice.
    assert_eq!(&*s.at(0).1.to_contiguous(), &[4.0, 5.0]);
    assert_eq!(&*s.at(1).1.to_contiguous(), &[10.0, 11.0]);
    assert_eq!(&*s.at(2).1.to_contiguous(), &[16.0, 17.0]);
    // Which agrees with slicing each element's view directly.
    for (i, (_, e)) in v.iter().enumerate() {
        assert_eq!(
            &*s.at(i).1.to_contiguous(),
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
    assert_eq!(&*w.at(1).1.to_contiguous(), &[10.0, 11.0]);
    assert_eq!(&*w.to_contiguous(), &[10.0, 11.0, 16.0, 17.0]);
}

#[test]
fn view_slicing_reshapes_elements() {
    let tss = [ts(100), ts(200)];
    let vals: Vec<f64> = (0..12).map(f64::from).collect();
    let v = SeriesView::<f64, 2>::from_slice([2, 3], &tss, &vals, 0);

    // An index drops an element axis: [2, 3] elements -> [3].
    let s: SeriesView<f64, 1> = v.slice_reshape((1, ..));
    assert_eq!(s.len(), 2);
    assert_eq!(s.extents(), [3]);
    assert_eq!(&*s.at(0).1.to_contiguous(), &[3.0, 4.0, 5.0]);
    assert_eq!(&*s.at(1).1.to_contiguous(), &[9.0, 10.0, 11.0]);

    // And `()` adds one: [2, 3] -> [2, 1, 3].
    let s: SeriesView<f64, 3> = v.slice_reshape((.., (), ..));
    assert_eq!(s.extents(), [2, 1, 3]);
    assert_eq!(
        &*s.at(1).1.to_contiguous(),
        &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );

    // A sliced series copies into an owned one with the sliced extents.
    let owned = v.slice((.., 1..3)).to_series();
    assert_eq!(owned.layout().extents(), [2, 2]);
    assert_eq!(owned.data(), &[1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0]);
    assert_eq!(owned.instants(), &tss);
}

#[test]
fn series_view_eq_compares_instants_and_elements() {
    let tss = [ts(1), ts(2)];
    let packed = SeriesView::<f64, 1>::from_slice([2], &tss, &[1.0, 2.0, 3.0, 4.0], 0);

    // The same elements laid 3 apart, with a pad scalar after each and a
    // trailing scalar the index space never reaches.
    let padded_vals = [1.0, 2.0, 9.0, 3.0, 4.0, 9.0, 9.0];
    let padded = SeriesView::from_parts(Strided::new([2], [1]), 3, &tss, &padded_vals, 0);
    assert_eq!(packed, padded);

    // Differing instants, or differing elements, make them unequal.
    let other_ts = [ts(1), ts(3)];
    assert_ne!(
        packed,
        SeriesView::<f64, 1>::from_slice([2], &other_ts, &[1.0, 2.0, 3.0, 4.0], 0),
    );
    assert_ne!(
        packed,
        SeriesView::<f64, 1>::from_slice([2], &tss, &[1.0, 2.0, 3.0, 5.0], 0),
    );
}

#[test]
fn view_pad_ndim_prepends_extent_1_element_axes() {
    let mut s = Series::<f64, 1>::new([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    s.push(ts(200), ArrayView::from_slice([2], &[3.0, 4.0]));

    let v = s.view().pad_ndim::<3>();
    assert_eq!(v.extents(), [1, 1, 2]);
    // The time axis, window and packing are untouched.
    assert_eq!(v.range(), 0..2);
    assert_eq!(v.instants(), s.instants());
    assert_eq!(v.as_slice(), s.view().as_slice());
    // Elements come out as rank-3 views of the same scalars.
    assert_eq!(v.at(1).0, ts(200));
    assert_eq!(v.at(1).1[[0, 0, 1]], 4.0);
    // Padding to the same rank is the identity.
    assert_eq!(s.view().pad_ndim::<1>(), s.view());
}

#[test]
#[should_panic(expected = "must be at least")]
fn view_pad_ndim_below_rank() {
    let mut s = Series::<f64, 1>::new([2]);
    s.push(ts(100), ArrayView::from_slice([2], &[1.0, 2.0]));
    let _ = s.view().pad_ndim::<0>();
}
