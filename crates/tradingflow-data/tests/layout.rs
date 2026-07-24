use tradingflow_data::layout::{ColMajor, RowMajor, Strided};
use tradingflow_data::{Layout, NewAxis, Slice, SliceReshape};

#[test]
fn row_major_strides_and_len() {
    let l = RowMajor::new([2, 3, 4]);
    assert_eq!(l.extents(), [2, 3, 4]);
    assert_eq!(l.strides(), [12, 4, 1]);
    assert_eq!(l.len(), 24);
    assert_eq!(l.ndim(), 3);
    assert!(l.is_contiguous());
    assert!(!l.is_empty());

    // The empty product: a rank-0 layout holds exactly one element.
    let l = RowMajor::new([]);
    assert_eq!(l.len(), 1);
    assert!(l.is_contiguous());

    // A zero extent makes the layout empty but keeps it contiguous.
    let l = RowMajor::new([2, 0]);
    assert_eq!(l.len(), 0);
    assert!(l.is_empty());
}

#[test]
fn col_major_strides_and_contiguity() {
    let l = ColMajor::new([2, 3, 4]);
    assert_eq!(l.strides(), [1, 2, 6]);
    assert_eq!(l.len(), 24);
    // Column-major is not *row-major* contiguous beyond rank 1.
    assert!(!l.is_contiguous());
    assert!(ColMajor::new([5]).is_contiguous());
    assert!(ColMajor::new([]).is_contiguous());

    // Offsets still walk in logical row-major order, through the strides.
    assert_eq!(
        ColMajor::new([2, 3]).iter().collect::<Vec<_>>(),
        vec![0, 2, 4, 1, 3, 5],
    );
}

#[test]
fn strided_detects_row_major() {
    // Canonical row-major strides for [2, 3] are [3, 1].
    assert!(Strided::new([2, 3], [3, 1]).is_contiguous());
    assert!(!Strided::new([2, 3], [1, 2]).is_contiguous());

    // The policy conversions preserve extents and strides.
    let rm: Strided<2> = RowMajor::new([2, 3]).into();
    assert_eq!(rm, Strided::new([2, 3], [3, 1]));
    let cm: Strided<2> = ColMajor::new([2, 3]).into();
    assert_eq!(cm, Strided::new([2, 3], [1, 2]));
    assert!(!cm.is_contiguous());
}

#[test]
fn offset_is_the_stride_dot_product() {
    let l = RowMajor::new([2, 3]);
    assert_eq!(l.offset([0, 0]), 0);
    assert_eq!(l.offset([0, 2]), 2);
    assert_eq!(l.offset([1, 0]), 3);
    assert_eq!(l.offset([1, 2]), 5);

    // A rank-0 layout addresses its element at the empty index.
    assert_eq!(RowMajor::new([]).offset([]), 0);

    // Strided: a column of the [2, 3] panel above.
    let col = Strided::new([2], [3]);
    assert_eq!(col.offset([0]), 0);
    assert_eq!(col.offset([1]), 3);
}

#[test]
fn offset_bounds_are_per_axis() {
    let l = RowMajor::new([2, 3]);
    assert!(l.offset_contains([1, 2]));
    // Flat offset 3 is in the buffer, but [0, 3] runs off the end of a row.
    assert!(!l.offset_contains([0, 3]));
    assert_eq!(l.offset_checked([0, 3]), None);
    assert_eq!(l.offset_checked([2, 0]), None);
    assert_eq!(l.offset_checked([1, 2]), Some(5));

    // Nothing is in bounds in an empty layout; everything is in a rank-0 one.
    assert!(!RowMajor::new([0]).offset_contains([0]));
    assert!(RowMajor::new([]).offset_contains([]));
}

#[test]
#[should_panic(expected = "out of bounds")]
fn offset_panics_out_of_bounds() {
    let _ = RowMajor::new([2, 3]).offset([2, 0]);
}

#[test]
fn offsets_walk_row_major() {
    // Contiguous: offsets are just 0..len.
    let l = RowMajor::new([2, 3]);
    assert_eq!(l.iter().collect::<Vec<_>>(), (0..6).collect::<Vec<_>>());
    assert_eq!(l.iter().len(), 6);

    // Strided: a column, stride 3.
    let col = Strided::new([2], [3]);
    assert_eq!(col.iter().collect::<Vec<_>>(), vec![0, 3]);

    // Transposed [2, 3] -> [3, 2]: row-major over swapped axes.
    let t = Strided::new([3, 2], [1, 3]);
    assert_eq!(t.iter().collect::<Vec<_>>(), vec![0, 3, 1, 4, 2, 5]);

    // A rank-0 layout yields its single element; an empty one yields nothing.
    assert_eq!(RowMajor::new([]).iter().collect::<Vec<_>>(), vec![0]);
    assert_eq!(
        RowMajor::new([0, 2]).iter().collect::<Vec<_>>(),
        vec![] as Vec<usize>,
    );
}

#[test]
fn offsets_agree_with_offset() {
    // The traversal's incremental carry must match the direct dot product,
    // whatever the policy behind the trait.
    fn check(l: impl Layout<3>) {
        let e = l.extents();
        let direct: Vec<_> = (0..e[0])
            .flat_map(|i| (0..e[1]).flat_map(move |j| (0..e[2]).map(move |k| [i, j, k])))
            .map(|idx| l.offset(idx))
            .collect();
        assert_eq!(l.iter().collect::<Vec<_>>(), direct);
    }
    check(RowMajor::new([2, 3, 2]));
    check(ColMajor::new([2, 3, 2]));
    check(Strided::new([2, 3, 2], [1, 8, 3]));
}

#[test]
fn offsets_are_exhausted_once() {
    let mut it = RowMajor::new([2]).iter();
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None); // fused
}

#[test]
fn slice_restricts_extents_and_keeps_strides() {
    let l = RowMajor::new([4, 5]);
    let (offset, s) = l.slice([1..3, 2..5]);
    assert_eq!(offset, l.offset([1, 2]));
    assert_eq!(s, Strided::new([2, 3], [5, 1]));
    // An interior slice of a row-major panel is no longer contiguous...
    assert!(!s.is_contiguous());
    // ...but a slice restricting only the leading axis still is.
    let (offset, rows) = l.slice([1..3, 0..5]);
    assert_eq!(offset, 5);
    assert!(rows.is_contiguous());

    // The full slice is the identity.
    let (offset, full) = l.slice([0..4, 0..5]);
    assert_eq!(offset, 0);
    assert_eq!(full, l.into());

    // A rank-0 layout has exactly one (empty) slice: itself.
    let full: [Slice; 0] = [];
    assert_eq!(RowMajor::new([]).slice(full), (0, Strided::new([], [])));
}

#[test]
fn slice_offsets_address_the_sub_region() {
    // Offsets of the sub-layout, shifted by the base offset, must agree with
    // direct offsets of the sub-indices — whatever the policy behind the trait.
    fn check(l: impl Layout<2>) {
        let (base, s) = l.slice([1..3, 1..4]);
        let direct: Vec<_> = (1..3)
            .flat_map(|i| (1..4).map(move |j| [i, j]))
            .map(|idx| l.offset(idx))
            .collect();
        assert_eq!(s.iter().map(|o| base + o).collect::<Vec<_>>(), direct);
    }
    check(RowMajor::new([3, 4]));
    check(ColMajor::new([3, 4]));
    check(Strided::new([3, 4], [1, 3]));
}

#[test]
fn slice_bounds_are_per_axis() {
    let l = RowMajor::new([2, 3]);
    // A range may end at the extent, but not beyond it.
    assert!(l.slice_checked([0..2, 0..3]).is_some());
    assert!(l.slice_checked([0..2, 0..4]).is_none());
    assert!(l.slice_checked([0..3, 0..3]).is_none());
    // A decreasing range selects nothing rather than failing.
    #[allow(clippy::reversed_empty_ranges)]
    let backwards = l.slice_checked([1..0, 0..3]);
    assert_eq!(backwards, Some((3, Strided::new([0, 3], [3, 1]))));

    // An empty range yields an empty layout; the offset is still the
    // stride dot product of the starts.
    let (offset, s) = l.slice([1..1, 0..3]);
    assert_eq!(offset, 3);
    assert!(s.is_empty());
    assert_eq!(s.extents(), [0, 3]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn slice_panics_out_of_bounds() {
    let _ = RowMajor::new([2, 3]).slice([(1..3, 2), (0..4, 1)]);
}

#[test]
fn slice_bounds_are_optional() {
    let l = RowMajor::new([4, 5]);
    // `..` keeps an axis in full; `a..` and `..b` default the other bound.
    assert_eq!(l.slice([.., ..]), (0, l.into()));
    assert_eq!(l.slice((1.., ..)), l.slice([1..4, 0..5]));
    assert_eq!(l.slice((.., ..3)), l.slice([0..4, 0..3]));
    // An unbounded end resolves to the axis extent, also under a step.
    assert_eq!(l.slice((.., (.., 2))), l.slice([(0..4, 1), (0..5, 2)]));
    assert_eq!(l.slice((.., (1.., 3))), l.slice([(0..4, 1), (1..5, 3)]));
    // A start at the extent is an empty tail; past it is out of bounds.
    assert!(l.slice_checked([4.., 0..]).is_some());
    assert!(l.slice_checked([5.., 0..]).is_none());
}

#[test]
fn stepped_slice_decimates_extents_and_scales_strides() {
    let l = RowMajor::new([5, 6]);
    let (offset, s) = l.slice([(0..5, 2), (0..6, 3)]);
    assert_eq!(offset, 0);
    assert_eq!(s, Strided::new([3, 2], [12, 3]));
    // Element [i, j] of the stepped layout is [2i, 3j] of the original,
    // and the span never grows.
    assert_eq!(s.offset([1, 1]), l.offset([2, 3]));
    assert!(s.span() <= l.span());

    // A partial final step still counts its first element: 1, 4 below 6.
    let (offset, s) = l.slice((.., (1..6, 3)));
    assert_eq!(offset, 1);
    assert_eq!(s.extents(), [5, 2]);

    // Step 1 on every axis is the identity; an empty axis stays empty.
    assert_eq!(l.slice([(0..5, 1), (0..6, 1)]), (0, l.into()));
    let (_, empty) = RowMajor::new([0, 4]).slice([(0..0, 2), (0..4, 2)]);
    assert!(empty.is_empty());
}

#[test]
fn stepped_slice_offsets_address_the_sub_region() {
    // start..end with a step in one call: addresses 1, 4, 7.
    let l = RowMajor::new([10]);
    let (offset, s) = l.slice([(1..8, 3)]);
    assert_eq!(s.extents(), [3]);
    assert_eq!(
        s.iter().map(|o| offset + o).collect::<Vec<_>>(),
        vec![1, 4, 7],
    );

    // Mixed steps agree with direct offsets — whatever the policy.
    fn check(l: impl Layout<2>) {
        let (base, s) = l.slice([(1..3, 1), (0..4, 2)]);
        let direct: Vec<_> = (1..3)
            .flat_map(|i| (0..4).step_by(2).map(move |j| [i, j]))
            .map(|idx| l.offset(idx))
            .collect();
        assert_eq!(s.iter().map(|o| base + o).collect::<Vec<_>>(), direct);
    }
    check(RowMajor::new([3, 4]));
    check(ColMajor::new([3, 4]));
    check(Strided::new([3, 4], [1, 3]));
}

#[test]
fn broadcast_axis_repeats_offsets() {
    // A step-0 slice pins an axis to one index, repeated `count` times.
    let l = RowMajor::new([2, 3]);
    let (base, s) = l.slice((Slice::new(1, Some(4), 0), ..));
    assert_eq!(base, 3);
    assert_eq!(s, Strided::new([4, 3], [0, 1]));
    // The physical span does not grow with the broadcast count...
    assert_eq!(s.span(), 3);
    // ...and offsets repeat the pinned row.
    assert_eq!(
        s.iter().map(|o| base + o).collect::<Vec<_>>(),
        vec![3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5],
    );

    // The broadcast source index must itself be in bounds; the count is
    // deliberately unconstrained by the extent.
    let oob = [Slice::new(2, Some(4), 0), Slice::from(..)];
    assert!(l.slice_checked(oob).is_none());
}

#[test]
fn slice_reshape_mixes_slices_indices_and_new_axes() {
    let l = RowMajor::new([2, 3, 4]);
    // A tuple mixes specifier types: an index collapses its axis, a range
    // keeps it, and `()` adds one — [2, 3, 4] -> [3, 1, 2].
    let (offset, s): (usize, Strided<3>) = l.slice_reshape((1, .., NewAxis, (0..4, 2)));
    assert_eq!(offset, l.offset([1, 0, 0]));
    // The new axis takes the row-major stride at its position (the product of
    // produced extents to its right), not 1.
    assert_eq!(s, Strided::new([3, 1, 2], [4, 2, 2]));

    // That stride choice makes purely rank-raising reshapes keep contiguity.
    let (_, c): (usize, Strided<4>) = l.slice_reshape((NewAxis, .., .., ..));
    assert!(c.is_contiguous());
    assert_eq!(c, l.pad_ndim());

    // All-index specifiers collapse to rank 0, addressing a single element.
    let (offset, scalar): (usize, Strided<0>) = l.slice_reshape((1, 2, 3));
    assert_eq!(
        (offset, scalar),
        (l.offset([1, 2, 3]), Strided::new([], []))
    );

    // Only new axes: rank 0 rises to rank 2, still one element.
    let (offset, up): (usize, Strided<2>) = RowMajor::new([]).slice_reshape((NewAxis, NewAxis));
    assert_eq!((offset, up), (0, Strided::new([1, 1], [1, 1])));

    // An array of one specifier type still works.
    let (offset, s): (usize, Strided<0>) =
        l.slice_reshape([SliceReshape::Index(1), SliceReshape::Index(2), 3.into()]);
    assert_eq!((offset, s), (l.offset([1, 2, 3]), Strided::new([], [])));
}

#[test]
fn slice_reshape_agrees_with_the_single_axis_operations() {
    let l = RowMajor::new([2, 3, 4]);
    // A reshape built from one operation matches that operation alone.
    let sliced: (usize, Strided<3>) = l.slice_reshape((.., 1..3, ..));
    assert_eq!(sliced, l.slice((.., 1..3, ..)));

    // Offsets of a reshaped layout still address the selected elements.
    let (base, s): (usize, Strided<2>) = l.slice_reshape((1, .., (0..4, 2)));
    let direct: Vec<_> = (0..3)
        .flat_map(|j| (0..4).step_by(2).map(move |k| [1, j, k]))
        .map(|idx| l.offset(idx))
        .collect();
    assert_eq!(s.iter().map(|o| base + o).collect::<Vec<_>>(), direct);
}

#[test]
fn slice_reshape_bounds_are_per_axis() {
    let l = RowMajor::new([2, 3]);
    let ok: Option<(usize, Strided<1>)> = l.slice_reshape_checked((1, ..));
    assert!(ok.is_some());
    // A collapsing index must be in bounds, as must a slice.
    let bad_index: Option<(usize, Strided<1>)> = l.slice_reshape_checked((2, ..));
    assert!(bad_index.is_none());
    let bad_slice: Option<(usize, Strided<2>)> = l.slice_reshape_checked((.., 0..4));
    assert!(bad_slice.is_none());
}

#[test]
#[should_panic]
fn slice_reshape_panics_on_wrong_consumed_rank() {
    let _: (usize, Strided<2>) = RowMajor::new([2, 3, 4]).slice_reshape((.., ..));
}

#[test]
#[should_panic]
fn slice_reshape_panics_on_wrong_produced_rank() {
    let _: (usize, Strided<3>) = RowMajor::new([2, 3]).slice_reshape((.., ..));
}

#[test]
fn slices_accept_tuples_and_arrays_alike() {
    let l = RowMajor::new([4, 5]);
    // A tuple may mix range forms that an array cannot hold together.
    assert_eq!(l.slice((1..3, ..)), l.slice([1..3, 0..5]));
    assert_eq!(l.slice((.., 2..)), l.slice([0..4, 2..5]));
    assert_eq!(l.slice(((1..4, 2), ..)), l.slice([(1..4, 2), (0..5, 1)]));
    // Explicit `Slice` values mix in too.
    assert_eq!(
        l.slice((Slice::new(1, Some(2), 1), ..)),
        l.slice((1..3, ..))
    );
    // The empty tuple slices a rank-0 layout.
    assert_eq!(RowMajor::new([]).slice(()), (0, Strided::new([], [])));
}

#[test]
fn transpose_permutes_extents_and_strides() {
    let l = RowMajor::new([2, 3, 4]);
    // Axis d of the result is axis perm[d] of the input.
    let t = l.transpose([2, 0, 1]);
    assert_eq!(t, Strided::new([4, 2, 3], [1, 12, 4]));
    // The transposed layout addresses the same element under the permuted
    // index, and the span never changes.
    assert_eq!(t.offset([3, 1, 2]), l.offset([1, 2, 3]));
    assert_eq!(t.span(), l.span());
    // The identity permutation is a no-op; the inverse roundtrips.
    assert_eq!(l.transpose([0, 1, 2]), l.into());
    assert_eq!(t.transpose([1, 2, 0]), l.into());
    // A rank-0 layout has exactly one (empty) permutation: itself.
    assert_eq!(RowMajor::new([]).transpose([]), Strided::new([], []));
}

#[test]
#[should_panic(expected = "not a permutation")]
fn transpose_rejects_repeated_axes() {
    let _ = RowMajor::new([2, 3, 4]).transpose([1, 2, 1]);
}

#[test]
#[should_panic(expected = "not a permutation")]
fn transpose_rejects_out_of_range_axis() {
    let _ = RowMajor::new([2, 3]).transpose([0, 3]);
}

#[test]
fn span_is_the_last_offset_plus_one() {
    // Contiguous: span == len.
    assert_eq!(RowMajor::new([2, 3]).span(), 6);
    assert_eq!(RowMajor::new([]).span(), 1);
    // Strided: a column of a [3, 4] panel touches offsets {0, 4, 8}.
    assert_eq!(Strided::new([3], [4]).span(), 9);
    // Empty: addresses nothing.
    assert_eq!(RowMajor::new([2, 0]).span(), 0);
    // The span is 1 + the largest offset `offsets` ever yields.
    let l = Strided::new([2, 3], [1, 2]);
    assert_eq!(l.span(), 1 + l.iter().max().unwrap());
}
