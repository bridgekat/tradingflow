use tradingflow_data::array::{apply_binary, apply_unary};
use tradingflow_data::layout::{Slice, Strided};
use tradingflow_data::{Array, ArrayView, Layout};

#[test]
fn full_and_zeros() {
    let a = Array::full([2, 3], 1.0_f64);
    assert_eq!(a.extents(), [2, 3]);
    assert_eq!(a.layout().len(), 6);
    assert_eq!(a.data(), &[1.0; 6]);

    let b = Array::<f64, 1>::zeros([4]);
    assert_eq!(b.data(), &[0.0; 4]);
}

#[test]
fn scalar() {
    let a = Array::scalar(42.0_f64);
    assert_eq!(a.extents(), [] as [usize; 0]);
    assert_eq!(a.layout().len(), 1);
    // A rank-0 array holds its one scalar at the empty index.
    assert_eq!(a[[]], 42.0);
}

#[test]
fn from_slice_matches_from_parts() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let v = ArrayView::from_slice([2, 3], &data);
    assert_eq!(v.extents(), [2, 3]);
    assert!(v.layout().is_contiguous());
    assert_eq!(v.to_array(), Array::from_parts([2, 3], data.clone().into()));
}

#[test]
#[should_panic(expected = "from_slice: extents [2, 3] expect 6 scalars, got 5")]
fn from_slice_wrong_len() {
    let _ = ArrayView::<f64, 2>::from_slice([2, 3], &[0.0; 5]);
}

#[test]
#[should_panic(expected = "from_parts: extents [2, 3] expect 6 scalars, got 5")]
fn array_from_parts_wrong_len() {
    let _ = Array::<f64, 2>::from_parts([2, 3], vec![0.0; 5].into());
}

#[test]
#[should_panic(expected = "from_parts: shape spans 5 scalars, got 4")]
fn view_from_parts_data_too_short() {
    // A [3]-extent, stride-2 column addresses offsets {0, 2, 4}.
    let _ = ArrayView::<f64, 1>::from_parts(Strided::new([3], [2]), &[0.0; 4]);
}

#[test]
fn assign_and_index_mut() {
    let mut a = Array::<f64, 1>::zeros([3]);
    let b = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    a.assign(b.view());
    assert_eq!(a.data(), &[1.0, 2.0, 3.0]);
    a[[1]] = 20.0;
    assert_eq!(a.data(), &[1.0, 20.0, 3.0]);
}

#[test]
fn assign_materializes_a_strided_view() {
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    // Column 1: extent 3, stride 2, from index 1.
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let mut a = Array::<f64, 1>::zeros([3]);
    a.assign(col1);
    assert_eq!(a.data(), &[2.0, 4.0, 6.0]);
}

#[test]
fn assign_squeezes_a_stride3_column() {
    let panel = Array::from_parts([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into());
    // Column 2: extent 2, stride 3, from index 2 -> [2.0, 5.0].
    let col2 = ArrayView::from_parts(Strided::new([2], [3]), &panel.data()[2..]);
    let mut dst = Array::<f64, 1>::zeros([2]);
    dst.assign(col2);
    assert_eq!(dst.data(), &[2.0, 5.0]);
}

#[test]
#[should_panic(expected = "assign: extents mismatch")]
fn assign_wrong_extents() {
    let mut a = Array::<f64, 1>::zeros([3]);
    let b = Array::<f64, 1>::zeros([2]);
    a.assign(b.view());
}

#[test]
fn index_is_per_axis() {
    let mut a = Array::from_parts([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into());
    assert_eq!(a[[0, 0]], 0.0);
    assert_eq!(a[[0, 2]], 2.0);
    assert_eq!(a[[1, 0]], 3.0);
    assert_eq!(a[[1, 2]], 5.0);
    a[[1, 1]] = 40.0;
    assert_eq!(a.data(), &[0.0, 1.0, 2.0, 3.0, 40.0, 5.0]);
}

#[test]
#[should_panic(expected = "index [0, 3] out of bounds for extents [2, 3]")]
fn index_out_of_bounds_per_axis() {
    // Flat offset 3 is inside the buffer, but [0, 3] is off the end of a row.
    let a = Array::<f64, 2>::zeros([2, 3]);
    let _ = a[[0, 3]];
}

#[test]
fn view_index_resolves_strides() {
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    assert_eq!(panel.view()[[2, 0]], 5.0);
    // Column 1: extent 3, stride 2, from index 1.
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    assert_eq!(col1[[0]], 2.0);
    assert_eq!(col1[[1]], 4.0);
    assert_eq!(col1[[2]], 6.0);
}

#[test]
fn reshape_same_rank_and_cross_rank() {
    // Rank-preserving reshape.
    let a = Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let a = a.reshape([3, 2]);
    assert_eq!(a.extents(), [3, 2]);
    assert_eq!(a.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Rank-changing reshape: flatten and back.
    let flat = a.reshape([6]);
    assert_eq!(flat.extents(), [6]);
    let back = flat.reshape([2, 3]);
    assert_eq!(back[[1, 2]], 6.0);
}

#[test]
#[should_panic(expected = "reshape")]
fn reshape_wrong_size() {
    let a = Array::<f64, 2>::zeros([2, 3]);
    let _ = a.reshape([2, 2]);
}

#[test]
fn view_is_copy_and_inline() {
    use std::mem::size_of;
    let word = size_of::<usize>();
    let fatptr = size_of::<&[f64]>(); // 2 words
    // data(&[T]) + extents[N] + strides[N] — all inline.
    assert_eq!(size_of::<ArrayView<f64, 1>>(), fatptr + word * 2);
    assert_eq!(size_of::<ArrayView<f64, 2>>(), fatptr + word * 4);
    fn assert_copy<T: Copy>() {}
    assert_copy::<ArrayView<f64, 3>>();
}

#[test]
fn as_slice_and_strided_column() {
    let panel = Array::from_parts(
        [3, 4],
        vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0,
        ]
        .into(),
    );
    assert!(panel.view().as_slice().is_some());

    // Column 1: extent 3, stride 4, from index 1 — strided.
    let col1 = ArrayView::from_parts(Strided::new([3], [4]), &panel.data()[1..]);
    assert!(col1.as_slice().is_none());
    assert_eq!(col1.iter().collect::<Vec<_>>(), vec![1.0, 5.0, 9.0]);
    assert_eq!(&*col1.to_contiguous(), &[1.0, 5.0, 9.0]);
}

#[test]
fn unary_contiguous_and_strided_agree() {
    let x = Array::from_parts([3], vec![1.0, 4.0, 9.0].into());
    let mut out = Array::<f64, 1>::zeros([3]);
    apply_unary(&mut out, x.view(), |v: f64| v.sqrt());
    assert_eq!(out.data(), &[1.0, 2.0, 3.0]);

    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let mut out = Array::<f64, 1>::zeros([3]);
    apply_unary(&mut out, col1, |v: f64| v * 10.0);
    assert_eq!(out.data(), &[20.0, 40.0, 60.0]);
}

#[test]
fn binary_mixed_contiguous_and_strided() {
    let a = Array::from_parts([3], vec![100.0, 200.0, 300.0].into());
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let bcol = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let mut out = Array::<f64, 1>::zeros([3]);
    apply_binary(&mut out, a.view(), bcol, |x, y| x + y);
    assert_eq!(out.data(), &[102.0, 204.0, 306.0]);
}

#[test]
fn partial_eq_includes_layout() {
    let a = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    let b = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    let c = Array::from_parts([3], vec![1.0, 2.0, 4.0].into());
    assert_eq!(a, b);
    assert_ne!(a, c);
    // Same scalars, different extents: unequal.
    let d = Array::from_parts([2, 3], vec![0.0; 6].into());
    let e = Array::from_parts([3, 2], vec![0.0; 6].into());
    assert_ne!(d, e);
}

// -- Iteration ---------------------------------------------------------------

#[test]
fn array_into_iter_moves_row_major() {
    let a = Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    assert_eq!(a.clone().into_iter().len(), 6);
    let collected: Vec<f64> = a.into_iter().collect();
    assert_eq!(collected, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn array_iter_borrows_and_clones() {
    let a = Array::from_parts([2, 2], vec![1.0, 2.0, 3.0, 4.0].into());
    assert_eq!(a.iter().len(), 4);
    assert_eq!(a.iter().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0]);
    // Borrowed: `a` is still usable, and `for x in &a` works.
    let sum: f64 = (&a).into_iter().sum();
    assert_eq!(sum, 10.0);
    assert_eq!(a.layout().len(), 4);
}

#[test]
fn arrayview_iter_honours_strides() {
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    // Column 1: extent 3, stride 2, from index 1 -> {2, 4, 6}.
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    assert_eq!(col1.iter().collect::<Vec<_>>(), vec![2.0, 4.0, 6.0]);
    // The view is `Copy`, so `IntoIterator` consumes a copy; the strided walk
    // matches `to_contiguous`.
    assert_eq!(
        col1.into_iter().collect::<Vec<_>>(),
        col1.to_contiguous().into_owned(),
    );
}

#[test]
fn array_iter_rank0_yields_one_scalar() {
    let s = Array::scalar(42.0_f64);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![42.0]);
    assert_eq!(s.into_iter().collect::<Vec<_>>(), vec![42.0]);
}

#[test]
fn view_slicing_selects_sub_regions() {
    // A [3, 4] panel of 0..12.
    let data: Vec<f64> = (0..12).map(f64::from).collect();
    let v = ArrayView::from_slice([3, 4], &data);

    // A rectangular sub-region reads the elements it selects.
    let s = v.slice((1..3, 1..3));
    assert_eq!(s.extents(), [2, 2]);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![5.0, 6.0, 9.0, 10.0]);
    assert_eq!(s[[0, 0]], 5.0);
    assert_eq!(s[[1, 1]], 10.0);

    // Stepping and open bounds compose in one call.
    let s = v.slice(((.., 2), (1.., 2)));
    assert_eq!(s.extents(), [2, 2]);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![1.0, 3.0, 9.0, 11.0]);

    // A single axis, the rest kept whole. An index keeps the axis here —
    // only `slice_reshape` drops it — so the row stays rank 2.
    let row = v.slice_along_axis(0, 2..3);
    assert_eq!(row.extents(), [1, 4]);
    assert_eq!(row.iter().collect::<Vec<_>>(), vec![8.0, 9.0, 10.0, 11.0]);

    // Slices of slices compose.
    assert_eq!(
        v.slice((1.., ..))
            .slice((..1, 2..))
            .iter()
            .collect::<Vec<_>>(),
        vec![6.0, 7.0]
    );
}

#[test]
fn view_slicing_reshapes_and_broadcasts() {
    let data: Vec<f64> = (0..12).map(f64::from).collect();
    let v = ArrayView::from_slice([3, 4], &data);

    // An index drops its axis, `()` adds one: [3, 4] -> [1, 2].
    let s: ArrayView<f64, 2> = v.slice_reshape((1, (), 1..3));
    assert_eq!(s.extents(), [1, 2]);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![5.0, 6.0]);

    // Indexing every axis leaves a rank-0 view of one scalar.
    let s: ArrayView<f64, 0> = v.slice_reshape((2, 3));
    assert_eq!(s[[]], 11.0);

    // A new axis broadcast with step 0 repeats the row it was inserted over.
    let b: ArrayView<f64, 3> = v.slice_reshape((.., (), ..));
    let b = b.slice_along_axis(1, Slice::new(0, Some(2), 0));
    assert_eq!(b.extents(), [3, 2, 4]);
    assert_eq!(b[[1, 0, 0]], b[[1, 1, 0]]);
    assert_eq!(b.to_array().data().len(), 24);

    // An empty selection is a valid, empty view.
    let e = v.slice((3.., ..));
    assert_eq!(e.extents(), [0, 4]);
    assert_eq!(e.iter().count(), 0);
}

#[test]
fn view_eq_compares_only_the_index_space() {
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    // Column 1: extent 3, stride 2, from index 1 -> [2, 4, 6].
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);

    // A strided view equals a packed one presenting the same elements.
    assert_eq!(col1, ArrayView::from_slice([3], &[2.0, 4.0, 6.0]));

    // Scalars the index space never reaches do not participate — neither the
    // stride gaps above nor the trailing scalar here.
    let trailing = [2.0, 4.0, 6.0, 99.0];
    assert_eq!(
        col1,
        ArrayView::from_parts(Strided::new([3], [1]), &trailing)
    );

    // Same scalars in the same order, different extents: unequal.
    let wide = ArrayView::from_slice([2, 3], panel.data());
    let tall = ArrayView::from_slice([3, 2], panel.data());
    assert_ne!(wide, tall);
    assert_eq!(
        wide.iter().collect::<Vec<_>>(),
        tall.iter().collect::<Vec<_>>()
    );

    // One differing element is enough.
    assert_ne!(col1, ArrayView::from_slice([3], &[2.0, 4.0, 7.0]));
}
