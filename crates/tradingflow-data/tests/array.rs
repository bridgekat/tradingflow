use tradingflow_data::array::{
    binary_map, binary_map_into, concat, concat_into, map, map_into, select, select_into,
    select_mask, select_mask_into, split, stack, stack_into, unstack,
};
use tradingflow_data::layout::Strided;
use tradingflow_data::{Array, ArrayView, Layout, NewAxis, Slice};

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
#[should_panic(expected = "expect 6 scalars, got 5")]
fn from_slice_wrong_len() {
    let _ = ArrayView::<f64, 2>::from_slice([2, 3], &[0.0; 5]);
}

#[test]
#[should_panic(expected = "expect 6 scalars, got 5")]
fn array_from_parts_wrong_len() {
    let _ = Array::<f64, 2>::from_parts([2, 3], vec![0.0; 5].into());
}

#[test]
#[should_panic(expected = "spans 5 scalars, got 4")]
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
#[should_panic(expected = "extents mismatch")]
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
#[should_panic(expected = "out of bounds")]
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
#[should_panic]
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
    assert_eq!(
        col1.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 5.0, 9.0]
    );
    assert_eq!(&*col1.to_contiguous(), &[1.0, 5.0, 9.0]);
}

#[test]
fn unary_contiguous_and_strided_agree() {
    let x = Array::from_parts([3], vec![1.0, 4.0, 9.0].into());
    let out = map(x.view(), |&v: &f64| v.sqrt());
    assert_eq!(out, Array::from_parts([3], vec![1.0, 2.0, 3.0].into()));

    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let out = map(col1, |&v: &f64| v * 10.0);
    assert_eq!(out, Array::from_parts([3], vec![20.0, 40.0, 60.0].into()));
}

#[test]
fn binary_mixed_contiguous_and_strided() {
    let a = Array::from_parts([3], vec![100.0, 200.0, 300.0].into());
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let bcol = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let out = binary_map(a.view(), bcol, |x, y| x + y);
    assert_eq!(
        out,
        Array::from_parts([3], vec![102.0, 204.0, 306.0].into())
    );
}

#[test]
fn apply_into_reuses_a_row_major_buffer() {
    let x = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    let mut out = [0.0; 3];
    map_into(&mut out, x.view(), |&v: &f64| v + 0.5);
    assert_eq!(out, [1.5, 2.5, 3.5]);
    binary_map_into(&mut out, x.view(), x.view(), |a, b| a * b);
    assert_eq!(out, [1.0, 4.0, 9.0]);
}

#[test]
#[should_panic(expected = "not broadcast-compatible")]
fn binary_extents_mismatch() {
    let a = Array::<f64, 1>::zeros([3]);
    let b = Array::<f64, 1>::zeros([2]);
    let _ = binary_map(a.view(), b.view(), |x, y| x + y);
}

#[test]
fn broadcast_row_against_column() {
    // [2, 1] + [1, 3] -> [2, 3], an outer sum.
    let a = Array::from_parts([2, 1], vec![10.0, 20.0].into());
    let b = Array::from_parts([1, 3], vec![1.0, 2.0, 3.0].into());
    let out = binary_map(a.view(), b.view(), |x, y| x + y);
    assert_eq!(
        out,
        Array::from_parts([2, 3], vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0].into())
    );
    // Broadcasting is symmetric in the extents.
    let out = binary_map(b.view(), a.view(), |x, y| x + y);
    assert_eq!(out.extents(), [2, 3]);
    assert_eq!(out.data(), &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
}

#[test]
fn broadcast_equal_extents_matches_map_binary() {
    let a = Array::from_parts([2, 2], vec![1.0, 2.0, 3.0, 4.0].into());
    let b = Array::from_parts([2, 2], vec![5.0, 6.0, 7.0, 8.0].into());
    let out = binary_map(a.view(), b.view(), |x, y| x * y);
    assert_eq!(out, binary_map(a.view(), b.view(), |x, y| x * y));
}

#[test]
fn broadcast_strided_operand() {
    // Column 1 of a [3, 2] panel as a [3, 1] view (extent 3 stride 2, then a
    // dangling extent-1 axis), broadcast against a [3, 3] panel.
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let col1 = ArrayView::from_parts(Strided::new([3, 1], [2, 1]), &panel.data()[1..]);
    let ones = Array::full([3, 3], 1.0);
    let out = binary_map(col1, ones.view(), |x, y| x * y);
    assert_eq!(
        out,
        Array::from_parts(
            [3, 3],
            vec![2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0].into()
        )
    );
}

#[test]
fn broadcast_into_reuses_a_row_major_buffer() {
    let a = Array::from_parts([2, 1], vec![1.0, 2.0].into());
    let b = Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let mut out = [0.0; 6];
    binary_map_into(&mut out, a.view(), b.view(), |x, y| x + y);
    assert_eq!(out, [2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);
}

#[test]
fn broadcast_zero_extent_is_empty() {
    // [1, 0] against [2, 1] -> [2, 0]: empty, no calls to `f`.
    let a = Array::<f64, 2>::zeros([1, 0]);
    let b = Array::from_parts([2, 1], vec![1.0, 2.0].into());
    let out = binary_map(a.view(), b.view(), |_, _| -> f64 { unreachable!() });
    assert_eq!(out.extents(), [2, 0]);
}

#[test]
#[should_panic(expected = "not broadcast-compatible")]
fn broadcast_incompatible_extents() {
    let a = Array::<f64, 2>::zeros([2, 3]);
    let b = Array::<f64, 2>::zeros([2, 2]);
    let _ = binary_map(a.view(), b.view(), |x, y| x + y);
}

#[test]
fn pad_ndim_prepends_extent_1_axes() {
    let a = Array::from_parts([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into());
    let v = a.view().pad_ndim::<4>();
    assert_eq!(v.extents(), [1, 1, 2, 3]);
    // Contiguity (and thus the elementwise fast path) is preserved.
    assert!(v.layout().is_contiguous());
    assert_eq!(v.as_slice(), a.view().as_slice());
    assert_eq!(v[[0, 0, 1, 2]], 5.0);
    // Padding to the same rank is the identity.
    assert_eq!(a.view().pad_ndim::<2>(), a.view());
}

#[test]
fn pad_ndim_strided_operand() {
    // Column 1 of a [3, 2] panel: extent 3, stride 2 — not contiguous.
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let v = col1.pad_ndim::<2>();
    assert_eq!(v.extents(), [1, 3]);
    assert!(v.as_slice().is_none());
    assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn pad_ndim_enables_cross_rank_broadcast() {
    // Full NumPy alignment: a rank-1 [3] against a rank-2 [2, 3], with the
    // rank promotion opted into at the call site.
    let a = Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let b = Array::from_parts([3], vec![10.0, 20.0, 30.0].into());
    let out = binary_map(a.view(), b.view().pad_ndim(), |x, y| x + y);
    assert_eq!(
        out,
        Array::from_parts([2, 3], vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0].into())
    );
}

#[test]
#[should_panic(expected = "must be at least")]
fn pad_ndim_below_rank() {
    let a = Array::<f64, 2>::zeros([2, 3]);
    let _ = a.view().pad_ndim::<1>();
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

// -- Concatenation and stacking -----------------------------------------------

#[test]
fn concat_rank1_sums_the_axis() {
    let a = Array::from_parts([2], vec![1.0, 2.0].into());
    let b = Array::from_parts([3], vec![3.0, 4.0, 5.0].into());
    let c = concat(&[a.view(), b.view()], 0);
    assert_eq!(
        c,
        Array::from_parts([5], vec![1.0, 2.0, 3.0, 4.0, 5.0].into())
    );
}

#[test]
fn concat_rank2_along_each_axis() {
    let a = Array::from_parts([2, 2], vec![0.0, 1.0, 2.0, 3.0].into());
    let b = Array::from_parts([1, 2], vec![4.0, 5.0].into());
    // Axis 0: rows of `b` follow rows of `a`.
    let c = concat(&[a.view(), b.view()], 0);
    assert_eq!(
        c,
        Array::from_parts([3, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into())
    );

    // Axis 1: inputs may be ragged along the concat axis.
    let b = Array::from_parts([2, 1], vec![4.0, 5.0].into());
    let c = concat(&[a.view(), b.view()], 1);
    assert_eq!(
        c,
        Array::from_parts([2, 3], vec![0.0, 1.0, 4.0, 2.0, 3.0, 5.0].into())
    );
}

#[test]
fn concat_materializes_strided_views() {
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    // Column 1: extent 3, stride 2, from index 1 -> [2, 4, 6].
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let c = concat(&[col1, ArrayView::from_slice([1], &[8.0])], 0);
    assert_eq!(c, Array::from_parts([4], vec![2.0, 4.0, 6.0, 8.0].into()));
}

#[test]
fn concat_of_empty_extents_is_empty() {
    let a = Array::<f64, 2>::zeros([0, 3]);
    let c = concat(&[a.view(), a.view()], 0);
    assert_eq!(c.extents(), [0, 3]);
    let c = concat(&[a.view(), a.view()], 1);
    assert_eq!(c.extents(), [0, 6]);
}

#[test]
fn concat_into_reuses_a_row_major_buffer() {
    let a = Array::from_parts([2], vec![1.0, 2.0].into());
    let b = Array::from_parts([2], vec![3.0, 4.0].into());
    let mut out = [0.0; 4];
    concat_into(&mut out, &[a.view(), b.view()], 0);
    assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    stack_into(&mut out, &[a.view(), b.view()], 1);
    assert_eq!(out, [1.0, 3.0, 2.0, 4.0]);
    // Combining no views writes nothing.
    let mut out = [7.0; 2];
    concat_into::<f64, 1>(&mut out, &[], 0);
    stack_into::<f64, 1>(&mut out, &[], 0);
    assert_eq!(out, [7.0, 7.0]);
}

#[test]
#[should_panic(expected = "requires at least one view")]
fn concat_no_views() {
    let _ = concat::<f64, 1>(&[], 0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn concat_axis_out_of_bounds() {
    let a = Array::<f64, 2>::zeros([2, 2]);
    let _ = concat(&[a.view()], 2);
}

#[test]
#[should_panic(expected = "incompatible with")]
fn concat_off_axis_extents_mismatch() {
    let a = Array::<f64, 2>::zeros([2, 3]);
    let b = Array::<f64, 2>::zeros([2, 2]);
    let _ = concat(&[a.view(), b.view()], 0);
}

#[test]
fn stack_rank1_along_each_axis() {
    let a = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    let b = Array::from_parts([3], vec![4.0, 5.0, 6.0].into());

    // Axis 0: the new axis enumerates the inputs.
    let s: Array<f64, 2> = stack(&[a.view(), b.view()], 0);
    assert_eq!(
        s,
        Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into())
    );

    // Axis 1: inputs are interleaved element-wise.
    let s = stack::<f64, 1, 2>(&[a.view(), b.view()], 1);
    assert_eq!(
        s,
        Array::from_parts([3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0].into())
    );
}

#[test]
fn stack_scalars_into_rank1() {
    let a = Array::scalar(1.0_f64);
    let b = Array::scalar(2.0_f64);
    let s: Array<f64, 1> = stack(&[a.view(), b.view()], 0);
    assert_eq!(s, Array::from_parts([2], vec![1.0, 2.0].into()));
}

#[test]
fn stack_materializes_strided_views() {
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    // Column 1: extent 3, stride 2, from index 1 -> [2, 4, 6].
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let s: Array<f64, 2> = stack(&[col1, ArrayView::from_slice([3], &[7.0, 8.0, 9.0])], 1);
    assert_eq!(
        s,
        Array::from_parts([3, 2], vec![2.0, 7.0, 4.0, 8.0, 6.0, 9.0].into())
    );
}

#[test]
#[should_panic(expected = "M (3) must be N + 1 (2)")]
fn stack_wrong_output_rank() {
    let a = Array::<f64, 1>::zeros([2]);
    let _: Array<f64, 3> = stack(&[a.view()], 0);
}

#[test]
#[should_panic(expected = "requires at least one view")]
fn stack_no_views() {
    let _: Array<f64, 2> = stack::<f64, 1, 2>(&[], 0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn stack_axis_out_of_bounds() {
    let a = Array::<f64, 1>::zeros([2]);
    let _: Array<f64, 2> = stack(&[a.view()], 2);
}

#[test]
#[should_panic(expected = "extents mismatch")]
fn stack_extents_mismatch() {
    let a = Array::<f64, 1>::zeros([2]);
    let b = Array::<f64, 1>::zeros([3]);
    let _: Array<f64, 2> = stack(&[a.view(), b.view()], 0);
}

// -- Splitting -----------------------------------------------------------------

#[test]
fn split_rank1_by_lengths() {
    let a = Array::from_parts([6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    let parts = split(a.view(), &[1, 2, 3], 0);
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], ArrayView::from_slice([1], &[1.0]));
    assert_eq!(parts[1], ArrayView::from_slice([2], &[2.0, 3.0]));
    assert_eq!(parts[2], ArrayView::from_slice([3], &[4.0, 5.0, 6.0]));
}

#[test]
fn split_rank2_along_each_axis() {
    let data: Vec<f64> = (0..8).map(f64::from).collect();
    let a = Array::from_parts([2, 4], data.into());

    // Axis 0: whole rows, still contiguous.
    let parts = split(a.view(), &[1, 1], 0);
    assert_eq!(
        parts[0],
        ArrayView::from_slice([1, 4], &[0.0, 1.0, 2.0, 3.0])
    );
    assert_eq!(
        parts[1],
        ArrayView::from_slice([1, 4], &[4.0, 5.0, 6.0, 7.0])
    );
    assert!(parts[0].as_slice().is_some());

    // Axis 1: column blocks become strided views — no copying.
    let parts = split(a.view(), &[3, 1], 1);
    assert_eq!(
        parts[0],
        ArrayView::from_slice([2, 3], &[0.0, 1.0, 2.0, 4.0, 5.0, 6.0])
    );
    assert_eq!(parts[1], ArrayView::from_slice([2, 1], &[3.0, 7.0]));
    assert!(parts[0].as_slice().is_none());
}

#[test]
fn split_is_zero_copy() {
    let a = Array::from_parts([4], vec![1.0, 2.0, 3.0, 4.0].into());
    let parts = split(a.view(), &[2, 2], 0);
    // Each part borrows the original buffer at its offset.
    assert_eq!(parts[0].data().as_ptr(), a.data().as_ptr());
    assert_eq!(parts[1].data().as_ptr(), a.data()[2..].as_ptr());
}

#[test]
fn split_roundtrips_with_concat() {
    let data: Vec<f64> = (0..8).map(f64::from).collect();
    let a = Array::from_parts([2, 4], data.into());
    let parts = split(a.view(), &[1, 1], 0);
    assert_eq!(concat(&parts, 0), a);
    let parts = split(a.view(), &[1, 3], 1);
    assert_eq!(concat(&parts, 1), a);
}

#[test]
fn split_of_a_strided_view() {
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    // Column 1: extent 3, stride 2, from index 1 -> [2, 4, 6].
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let parts = split(col1, &[1, 2], 0);
    assert_eq!(parts[0], ArrayView::from_slice([1], &[2.0]));
    assert_eq!(parts[1], ArrayView::from_slice([2], &[4.0, 6.0]));
}

#[test]
fn split_allows_empty_parts() {
    let a = Array::from_parts([4], vec![1.0, 2.0, 3.0, 4.0].into());
    let parts = split(a.view(), &[0, 4, 0], 0);
    assert_eq!(parts[0].extents(), [0]);
    assert_eq!(parts[1], a.view());
    assert_eq!(parts[2].extents(), [0]);
    // No lengths at all is valid only for an empty axis.
    let e = Array::<f64, 1>::zeros([0]);
    assert!(split(e.view(), &[], 0).is_empty());
}

#[test]
#[should_panic(expected = "out of bounds")]
fn split_axis_out_of_bounds() {
    let a = Array::<f64, 1>::zeros([4]);
    let _ = split(a.view(), &[2, 2], 1);
}

#[test]
#[should_panic(expected = "expected extent 4")]
fn split_lengths_sum_mismatch() {
    let a = Array::<f64, 1>::zeros([4]);
    let _ = split(a.view(), &[1, 2], 0);
}

#[test]
fn unstack_drops_the_axis() {
    let a = Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());

    // Axis 0: rows.
    let rows: Vec<ArrayView<f64, 1>> = unstack(a.view(), 0);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], ArrayView::from_slice([3], &[1.0, 2.0, 3.0]));
    assert_eq!(rows[1], ArrayView::from_slice([3], &[4.0, 5.0, 6.0]));

    // Axis 1: columns — strided views over the same buffer.
    let cols: Vec<ArrayView<f64, 1>> = unstack(a.view(), 1);
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[1], ArrayView::from_slice([2], &[2.0, 5.0]));
    assert!(cols[1].as_slice().is_none());
}

#[test]
fn unstack_roundtrips_with_stack() {
    let a = Array::from_parts([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    for axis in 0..2 {
        let parts: Vec<ArrayView<f64, 1>> = unstack(a.view(), axis);
        assert_eq!(stack::<f64, 1, 2>(&parts, axis), a);
    }
}

#[test]
fn unstack_rank1_into_scalars() {
    let a = Array::from_parts([2], vec![1.0, 2.0].into());
    let scalars: Vec<ArrayView<f64, 0>> = unstack(a.view(), 0);
    assert_eq!(*scalars[0], 1.0);
    assert_eq!(*scalars[1], 2.0);
}

#[test]
#[should_panic(expected = "M (1) must be N (3) - 1")]
fn unstack_wrong_output_rank() {
    let a = Array::<f64, 3>::zeros([2, 2, 2]);
    let _: Vec<ArrayView<f64, 1>> = unstack(a.view(), 0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn unstack_axis_out_of_bounds() {
    let a = Array::<f64, 2>::zeros([2, 2]);
    let _: Vec<ArrayView<f64, 1>> = unstack(a.view(), 2);
}

// -- Select ------------------------------------------------------------------

#[test]
fn select_reorders_and_repeats() {
    let a = Array::from_parts([4], vec![1.0, 2.0, 3.0, 4.0].into());
    let t = select(a.view(), &[3, 0, 0, 2], 0);
    assert_eq!(t, Array::from_parts([4], vec![4.0, 1.0, 1.0, 3.0].into()));
}

#[test]
fn select_rank2_along_each_axis() {
    let data: Vec<f64> = (0..6).map(f64::from).collect();
    let a = Array::from_parts([2, 3], data.into());

    // Axis 0: whole rows.
    let t = select(a.view(), &[1, 0], 0);
    assert_eq!(
        t,
        Array::from_parts([2, 3], vec![3.0, 4.0, 5.0, 0.0, 1.0, 2.0].into())
    );

    // Axis 1: columns.
    let t = select(a.view(), &[2, 0], 1);
    assert_eq!(
        t,
        Array::from_parts([2, 2], vec![2.0, 0.0, 5.0, 3.0].into())
    );
}

#[test]
fn select_no_indices_empties_the_axis() {
    let a = Array::from_parts([2, 3], vec![0.0; 6].into());
    let t = select(a.view(), &[], 1);
    assert_eq!(t.extents(), [2, 0]);
}

#[test]
fn select_of_a_strided_view() {
    let panel = Array::from_parts([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into());
    // Column 1: extent 3, stride 2, from index 1 -> [2, 4, 6].
    let col1 = ArrayView::from_parts(Strided::new([3], [2]), &panel.data()[1..]);
    let t = select(col1, &[2, 0], 0);
    assert_eq!(t, Array::from_parts([2], vec![6.0, 2.0].into()));
}

#[test]
#[should_panic(expected = "out of bounds")]
fn select_axis_out_of_bounds() {
    let a = Array::<f64, 2>::zeros([2, 3]);
    let _ = select(a.view(), &[0], 2);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn select_index_out_of_bounds() {
    let a = Array::<f64, 2>::zeros([2, 3]);
    let _ = select(a.view(), &[0, 3], 1);
}

#[test]
fn select_mask_keeps_masked_entries() {
    let data: Vec<f64> = (0..6).map(f64::from).collect();
    let a = Array::from_parts([2, 3], data.into());
    let c = select_mask(a.view(), &[true, false, true], 1);
    assert_eq!(
        c,
        Array::from_parts([2, 2], vec![0.0, 2.0, 3.0, 5.0].into())
    );
    // An all-false mask empties the axis.
    let c = select_mask(a.view(), &[false, false], 0);
    assert_eq!(c.extents(), [0, 3]);
}

#[test]
fn select_into_reuses_a_row_major_buffer() {
    let a = Array::from_parts([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into());
    let mut out = [0.0; 4];
    select_into(&mut out, a.view(), &[2, 0], 1);
    assert_eq!(out, [2.0, 0.0, 5.0, 3.0]);
    select_mask_into(&mut out, a.view(), &[true, false, true], 1);
    assert_eq!(out, [0.0, 2.0, 3.0, 5.0]);
    // Selecting nothing writes nothing.
    let mut out = [7.0; 2];
    select_into(&mut out, a.view(), &[], 1);
    select_mask_into(&mut out, a.view(), &[false, false, false], 1);
    assert_eq!(out, [7.0, 7.0]);
}

#[test]
#[should_panic(expected = "mask length")]
fn select_mask_length_mismatch() {
    let a = Array::<f64, 2>::zeros([2, 3]);
    let _ = select_mask(a.view(), &[true, false], 1);
}

// -- Transposition -------------------------------------------------------------

#[test]
fn transpose_view_is_zero_copy() {
    let data: Vec<f64> = (0..6).map(f64::from).collect();
    let a = Array::from_parts([2, 3], data.into());
    let t = a.view().transpose([1, 0]);
    assert_eq!(t.extents(), [3, 2]);
    // The transposed view borrows the same buffer...
    assert_eq!(t.data().as_ptr(), a.data().as_ptr());
    // ...and addresses the same element under the swapped index.
    assert_eq!(t[[2, 1]], a[[1, 2]]);
    assert_eq!(
        t.iter().copied().collect::<Vec<_>>(),
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
    // The identity permutation keeps contiguity; a real swap loses it.
    assert!(a.view().transpose([0, 1]).as_slice().is_some());
    assert!(t.as_slice().is_none());
}

#[test]
fn transpose_rank3_permutation() {
    let data: Vec<f64> = (0..24).map(f64::from).collect();
    let a = Array::from_parts([2, 3, 4], data.into());
    // Axis d of the result is axis perm[d] of the input: [2, 3, 4] -> [4, 2, 3].
    let t = a.view().transpose([2, 0, 1]);
    assert_eq!(t.extents(), [4, 2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(t[[k, i, j]], a[[i, j, k]]);
            }
        }
    }
}

#[test]
#[should_panic(expected = "not a permutation")]
fn transpose_rejects_non_permutation() {
    let a = Array::<f64, 2>::zeros([2, 3]);
    let _ = a.view().transpose([0, 0]);
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
    assert_eq!(
        a.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
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
    assert_eq!(
        col1.iter().copied().collect::<Vec<_>>(),
        vec![2.0, 4.0, 6.0]
    );
    // The view is `Copy`, so `IntoIterator` consumes a copy; the strided walk
    // matches `to_contiguous`.
    assert_eq!(
        col1.into_iter().copied().collect::<Vec<_>>(),
        col1.to_contiguous().into_owned(),
    );
}

#[test]
fn array_iter_rank0_yields_one_scalar() {
    let s = Array::scalar(42.0_f64);
    assert_eq!(s.iter().copied().collect::<Vec<_>>(), vec![42.0]);
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
    assert_eq!(
        s.iter().copied().collect::<Vec<_>>(),
        vec![5.0, 6.0, 9.0, 10.0]
    );
    assert_eq!(s[[0, 0]], 5.0);
    assert_eq!(s[[1, 1]], 10.0);

    // Stepping and open bounds compose in one call.
    let s = v.slice(((.., 2), (1.., 2)));
    assert_eq!(s.extents(), [2, 2]);
    assert_eq!(
        s.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 3.0, 9.0, 11.0]
    );

    // A single axis, the rest kept whole. An index keeps the axis here —
    // only `slice_reshape` drops it — so the row stays rank 2.
    let row = v.slice((2..3, ..));
    assert_eq!(row.extents(), [1, 4]);
    assert_eq!(
        row.iter().copied().collect::<Vec<_>>(),
        vec![8.0, 9.0, 10.0, 11.0]
    );

    // Slices of slices compose.
    assert_eq!(
        v.slice((1.., ..))
            .slice((..1, 2..))
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![6.0, 7.0]
    );
}

#[test]
fn view_slicing_reshapes_and_broadcasts() {
    let data: Vec<f64> = (0..12).map(f64::from).collect();
    let v = ArrayView::from_slice([3, 4], &data);

    // An index drops its axis, `()` adds one: [3, 4] -> [1, 2].
    let s: ArrayView<f64, 2> = v.slice_reshape((1, NewAxis, 1..3));
    assert_eq!(s.extents(), [1, 2]);
    assert_eq!(s.iter().copied().collect::<Vec<_>>(), vec![5.0, 6.0]);

    // Indexing every axis leaves a rank-0 view of one scalar.
    let s: ArrayView<f64, 0> = v.slice_reshape((2, 3));
    assert_eq!(s[[]], 11.0);

    // A new axis broadcast with step 0 repeats the row it was inserted over.
    let b: ArrayView<f64, 3> = v.slice_reshape((.., NewAxis, ..));
    let b = b.slice((.., Slice::new(0, Some(2), 0), ..));
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
