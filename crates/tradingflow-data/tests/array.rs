//! Integration tests for `Array` / `ArrayView` and the element-wise kernels.

use tradingflow_data::array::{apply_binary, apply_unary};
use tradingflow_data::{Array, ArrayView, Shape};

#[test]
fn full_and_zeros() {
    let a = Array::full([2, 3], 1.0_f64);
    assert_eq!(a.extents(), [2, 3]);
    assert_eq!(a.len(), 6);
    assert_eq!(a.data(), &[1.0; 6]);

    let b = Array::<f64, 1>::zeros([4]);
    assert_eq!(b.data(), &[0.0; 4]);
}

#[test]
fn scalar() {
    let a = Array::scalar(42.0_f64);
    assert_eq!(a.extents(), [] as [usize; 0]);
    assert_eq!(a.len(), 1);
    // A rank-0 array holds its one scalar at the empty index.
    assert_eq!(a[[]], 42.0);
}

#[test]
fn from_slice_matches_from_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let v = ArrayView::from_slice([2, 3], &data);
    assert_eq!(v.extents(), [2, 3]);
    assert!(v.shape().is_contiguous());
    assert_eq!(v.to_array(), Array::from_vec([2, 3], data.clone()));
}

#[test]
#[should_panic(expected = "from_slice: extents [2, 3] expect 6 scalars, got 5")]
fn from_slice_wrong_len() {
    let _ = ArrayView::<f64, 2>::from_slice([2, 3], &[0.0; 5]);
}

#[test]
#[should_panic(expected = "from_parts: shape spans 5 scalars, got 4")]
fn from_parts_data_too_short() {
    // A [3]-extent, stride-2 column addresses offsets {0, 2, 4}.
    let _ = ArrayView::<f64, 1>::from_parts(Shape::strided([3], [2]), &[0.0; 4]);
}

#[test]
fn assign_and_index_mut() {
    let mut a = Array::<f64, 1>::zeros([3]);
    let b = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
    a.assign(b.view());
    assert_eq!(a.data(), &[1.0, 2.0, 3.0]);
    a[[1]] = 20.0;
    assert_eq!(a.data(), &[1.0, 20.0, 3.0]);
}

#[test]
fn assign_materializes_a_strided_view() {
    let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // Column 1: extent 3, stride 2, from index 1.
    let col1 = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
    let mut a = Array::<f64, 1>::zeros([3]);
    a.assign(col1);
    assert_eq!(a.data(), &[2.0, 4.0, 6.0]);
}

#[test]
fn assign_squeezes_a_stride3_column() {
    // `assign` materializes a strided view row-major — the public path through
    // the internal `write_row_major` squeeze.
    let panel = Array::from_vec([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    // Column 2: extent 2, stride 3, from index 2 -> [2.0, 5.0].
    let col2 = ArrayView::from_parts(Shape::strided([2], [3]), &panel.data()[2..]);
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
    let mut a = Array::from_vec([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
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
    let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(panel.view()[[2, 0]], 5.0);
    // Column 1: extent 3, stride 2, from index 1.
    let col1 = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
    assert_eq!(col1[[0]], 2.0);
    assert_eq!(col1[[1]], 4.0);
    assert_eq!(col1[[2]], 6.0);
}

#[test]
fn reshape() {
    // Same-rank reshape (rank is compile-time fixed at `N`).
    let mut a = Array::from_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    a.reshape([3, 2]);
    assert_eq!(a.extents(), [3, 2]);
    assert_eq!(a.len(), 6);
}

#[test]
#[should_panic(expected = "reshape")]
fn reshape_wrong_size() {
    let mut a = Array::<f64, 2>::zeros([2, 3]);
    a.reshape([2, 2]);
}

#[test]
fn view_is_copy_and_inline() {
    use std::mem::size_of;
    let word = size_of::<usize>();
    let fatptr = size_of::<&[f64]>(); // 2 words
    // data(&[T]) + extents[N] + strides[N] — all inline (no offset).
    assert_eq!(size_of::<ArrayView<f64, 1>>(), fatptr + word * 2);
    assert_eq!(size_of::<ArrayView<f64, 2>>(), fatptr + word * 4);
    fn assert_copy<T: Copy>() {}
    assert_copy::<ArrayView<f64, 3>>();
}

#[test]
fn as_slice_and_strided_column() {
    let panel = Array::from_vec(
        [3, 4],
        vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0,
        ],
    );
    assert!(panel.view().as_slice().is_some());

    // Column 1: extent 3, stride 4, from index 1 — strided.
    let col1 = ArrayView::from_parts(Shape::strided([3], [4]), &panel.data()[1..]);
    assert!(col1.as_slice().is_none());
    assert_eq!(col1.to_vec(), vec![1.0, 5.0, 9.0]);
    assert_eq!(&*col1.to_contiguous(), &[1.0, 5.0, 9.0]);
}

#[test]
fn unary_contiguous_and_strided_agree() {
    let x = Array::from_vec([3], vec![1.0, 4.0, 9.0]);
    let mut out = Array::<f64, 1>::zeros([3]);
    apply_unary(&mut out, x.view(), |v: f64| v.sqrt());
    assert_eq!(out.data(), &[1.0, 2.0, 3.0]);

    let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let col1 = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
    let mut out = Array::<f64, 1>::zeros([3]);
    apply_unary(&mut out, col1, |v: f64| v * 10.0);
    assert_eq!(out.data(), &[20.0, 40.0, 60.0]);
}

#[test]
fn binary_mixed_contiguous_and_strided() {
    let a = Array::from_vec([3], vec![100.0, 200.0, 300.0]);
    let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bcol = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
    let mut out = Array::<f64, 1>::zeros([3]);
    apply_binary(&mut out, a.view(), bcol, |x, y| x + y);
    assert_eq!(out.data(), &[102.0, 204.0, 306.0]);
}

#[test]
fn partial_eq_includes_shape() {
    let a = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
    let b = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
    let c = Array::from_vec([3], vec![1.0, 2.0, 4.0]);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// -- Iteration ---------------------------------------------------------------

#[test]
fn array_into_iter_moves_row_major() {
    let a = Array::from_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(a.clone().into_iter().len(), 6);
    let collected: Vec<f64> = a.into_iter().collect();
    assert_eq!(collected, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn array_iter_borrows_and_clones() {
    let a = Array::from_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a.iter().len(), 4);
    assert_eq!(a.iter().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0]);
    // Borrowed: `a` is still usable, and `for x in &a` works.
    let sum: f64 = (&a).into_iter().sum();
    assert_eq!(sum, 10.0);
    assert_eq!(a.len(), 4);
}

#[test]
fn arrayview_iter_honours_strides() {
    let panel = Array::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // Column 1: extent 3, stride 2, from index 1 -> {2, 4, 6}.
    let col1 = ArrayView::from_parts(Shape::strided([3], [2]), &panel.data()[1..]);
    assert_eq!(col1.iter().collect::<Vec<_>>(), vec![2.0, 4.0, 6.0]);
    // The view is `Copy`, so `IntoIterator` consumes a copy and the strided
    // walk matches `to_vec`.
    assert_eq!(col1.into_iter().collect::<Vec<_>>(), col1.to_vec());
}

#[test]
fn array_iter_rank0_yields_one_scalar() {
    let s = Array::scalar(42.0_f64);
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![42.0]);
    assert_eq!(s.into_iter().collect::<Vec<_>>(), vec![42.0]);
}

#[test]
fn array_into_iter_double_ended() {
    let a = Array::from_vec([4], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        a.into_iter().rev().collect::<Vec<_>>(),
        vec![4.0, 3.0, 2.0, 1.0]
    );
}
